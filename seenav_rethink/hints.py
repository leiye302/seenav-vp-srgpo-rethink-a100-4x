"""Bounded train-only OpenAI-compatible vision requests, grouped by trajectory."""

from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from PIL import Image

SYSTEM = Path(__file__).with_name("local_hint_system.txt")
MAX_HINT_CHARS = 320
MAX_BODY_BYTES = 2 * 1024**2
HINT = re.compile(r"Next move: [0-7]\. [^\r\n<>]+\Z")


@dataclass(frozen=True)
class Settings:
    endpoint: str
    model: str
    key: str = field(repr=False)
    timeout: float = 45
    attempts: int = 2
    parallel: int = 4
    images_per_request: int = 4
    max_tokens: int = 2048
    update_budget: int = 180


def from_env():
    endpoint = os.environ.get("SEENAV_HINT_API_URL", "").strip()
    url = urlsplit(endpoint)
    model = os.environ.get("SEENAV_HINT_MODEL", "").strip()
    key = os.environ.get("SEENAV_HINT_API_KEY", "").strip()
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or not url.path.endswith("/chat/completions") or not model or not key):
        raise ValueError("Set SEENAV_HINT_API_URL (HTTPS /chat/completions), SEENAV_HINT_MODEL and SEENAV_HINT_API_KEY.")

    def option(name, default, low, high):
        try:
            n = int(os.environ.get("SEENAV_HINT_" + name, default))
        except ValueError:
            raise ValueError(f"SEENAV_HINT_{name} must be an integer") from None
        if not low <= n <= high:
            raise ValueError(f"SEENAV_HINT_{name} must be in [{low}, {high}]")
        return n

    return Settings(endpoint, model, key, option("TIMEOUT", 45, 5, 120),
                    option("ATTEMPTS", 2, 1, 3), option("PARALLEL", 4, 1, 16),
                    option("IMAGES_PER_REQUEST", 4, 1, 8), option("MAX_TOKENS", 2048, 256, 8192),
                    option("UPDATE_BUDGET", 180, 20, 900))


def image_url(anchor):
    # Native EBNav anchor and observation reference the same BEV+FV tensor.
    # Reuse the native process_image after its collector has converted CHW to HWC.
    from agent_system.multi_turn_rollout.utils import process_image

    a = np.asarray(anchor)
    if a.ndim != 3 or a.shape[-1] != 3 or a.dtype == object:
        # Some collate versions coerce nested arrays to object dtype.
        a = np.asarray(anchor, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError("invalid native image anchor")
    image = process_image(a)
    if not isinstance(image, Image.Image) or image.size != (1000, 500):
        raise ValueError("unexpected native EBNav image geometry")
    out = BytesIO()
    image.save(out, format="PNG", compress_level=1)
    return "data:image/png;base64," + b64encode(out.getvalue()).decode("ascii")


def messages(timeline, selected):
    result = "success" if any(row["success"] for row in timeline) else "not successful"
    summary = [{"step": x["step"], "executed_action": x["executed_action"], "feedback": x["feedback"],
                "terminal": x["done"]} for x in timeline]
    content = [{"type": "text", "text": f"Episode outcome: {result}. Actual execution trace (hindsight): "
                + json.dumps(summary, ensure_ascii=False)}]
    for step, ctx, reasoning, move, url in selected:
        content.append({"type": "text", "text": json.dumps({
            "selected_step": step, "actor_visible_prompt": ctx, "actor_reasoning": reasoning,
            "actor_proposed_move": move}, ensure_ascii=False)})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "system", "content": SYSTEM.read_text(encoding="utf-8")},
            {"role": "user", "content": content}]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post(settings, payload):
    req = Request(settings.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                  headers={"Authorization": "Bearer " + settings.key, "Content-Type": "application/json",
                           "Accept": "application/json"}, method="POST")
    with build_opener(NoRedirect()).open(req, timeout=settings.timeout) as response:
        body = response.read(MAX_BODY_BYTES+1)
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("oversized response")
        return json.loads(body)


def parse_reply(body, steps):
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("truncated completion")
    content = choice["message"]["content"]
    if isinstance(content, list):
        content = "".join(x["text"] for x in content if x.get("type") == "text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty completion")
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1][:-3].strip()

    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate reply key")
            result[k] = v
        return result

    values = json.loads(content, object_pairs_hook=unique)["hints"]
    if not isinstance(values, dict):
        raise ValueError("invalid hints object")
    accepted, rejected = {}, {}
    for step in steps:
        value = values.get(str(step))
        if isinstance(value, str) and len(value) <= MAX_HINT_CHARS and HINT.fullmatch(value):
            accepted[step] = value
        else:
            rejected[step] = "missing_or_invalid_hint"
    return accepted, rejected


def one_request(settings, timeline, selected, deadline):
    if time.monotonic() >= deadline:
        return {}, {x[0]: "budget" for x in selected}, 0
    # Isolate a bad frame instead of losing every otherwise valid step.
    viable, frame_errors = [], {}
    for entry in selected:
        try:
            step, ctx, reasoning, move, anchor = entry
            viable.append((step, ctx, reasoning, move, image_url(anchor)))
        except Exception:
            frame_errors[entry[0]] = "image"
    if not viable:
        return {}, frame_errors, 0
    steps = [x[0] for x in viable]
    # Pass already encoded frames, so an episode request does not re-encode them
    # on a retry. No prompt, body, image or key is written to a log or cache.
    payload = {"model": settings.model, "max_tokens": settings.max_tokens, "temperature": 0,
               "messages": messages(timeline, viable)}
    reason = "api_error"
    attempts = 0
    for i in range(settings.attempts):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            reason = "budget"
            break
        try:
            attempts += 1
            accepted, rejected = parse_reply(post(replace(settings, timeout=min(settings.timeout, remaining)),
                                                   payload), steps)
            if accepted or i+1 == settings.attempts:
                return accepted, {**frame_errors, **rejected}, attempts
            reason = "missing_or_invalid_hint"
        except HTTPError as exc:
            reason = f"http_{exc.code}"
            exc.close()
            if exc.code not in (408, 429) and exc.code < 500:
                break
        except (URLError, TimeoutError, OSError):
            reason = "transport"
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            reason = "bad_reply"
        except Exception:
            reason = "api_error"
        if i+1 < settings.attempts:
            time.sleep(max(0, min(0.5 * (2**i), deadline-time.monotonic())))
    return {}, {**frame_errors, **{s: reason for s in steps}}, attempts


def request_hints(settings, trajectories, selections):
    jobs = []
    for uid, steps in selections.items():
        selected = sorted(steps.values(), key=lambda x: x[0])
        timeline = sorted(trajectories[uid].values(), key=lambda x: x["step"])
        k = settings.images_per_request
        jobs.extend((uid, timeline, selected[i:i+k]) for i in range(0, len(selected), k))
    if not jobs:
        return {}, {}, 0
    deadline = time.monotonic() + settings.update_budget

    def request(job):
        uid, timeline, selected = job
        try:
            good, failed, attempts = one_request(settings, timeline, selected, deadline)
        except Exception:
            good, failed, attempts = {}, {x[0]: "request_failure" for x in selected}, 0
        return uid, good, failed, attempts

    hints, errors, attempts = {}, {}, 0
    with ThreadPoolExecutor(max_workers=min(len(jobs), settings.parallel)) as pool:
        for uid, good, failed, n in pool.map(request, jobs):
            hints.update({(uid, s): value for s, value in good.items()})
            errors.update({(uid, s): value for s, value in failed.items()})
            attempts += n
    return hints, errors, attempts
