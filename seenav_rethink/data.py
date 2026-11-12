"""Verified native SeeNav H/R/action rows; the dataset contains no executable code."""

from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

from reproduce import LOCK, sha
from .protocol import EOS, MARKER, parse_decision

BASE_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ROUTE_COUNTS = {"good": 5, "bad": 3}
CONTEXT_LIMIT = 8192
NATIVE_MAX_PROMPT = 3072
ACTION = re.compile(r"<action>\s*([0-7])\s*</action>\Z")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_bytes())


def load_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def member(root, name):
    path = (root / name).resolve()
    check(not Path(name).is_absolute() and path.is_relative_to(root.resolve()), "unsafe package path")
    return path


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def decode_frame(raw, digest):
    from PIL import Image

    check(digest_bytes(raw) == digest, "changed native observation image")
    with Image.open(BytesIO(raw)) as image:
        check(image.format == "PNG" and image.size == (1000, 500), "expected native BEV+FV PNG")
        return image.convert("RGB")


def native_template(source):
    # Read only the string literal from the official pinned source.
    import ast

    path = source / "verl-agent/agent_system/environments/prompts/ebnav.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "EBNAV_TEMPLATE_VP" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Native VP tag prompt not found")


def prompt_pattern(template):
    return re.compile(re.escape(template).replace(re.escape("{user_instruction}"), r".+?")
                      .replace(re.escape("{history}"), r".*?"), re.S)


def verify_package(root, template, *, verify_frames=False):
    manifest = load_json(root / "manifest.json")
    contract = manifest["contract"]
    check(manifest["schema"] == "seenav_native_route_sft_v1"
          and manifest["seenav_commit"] == LOCK["commit"]
          and manifest["base_model"] == LOCK["base_model"] and manifest["base_revision"] == BASE_REVISION
          and manifest["route_per_batch"] == ROUTE_COUNTS and manifest["context_limit"] == CONTEXT_LIMIT
          and contract["marker"] == MARKER and contract["reasoning_ce"] is False
          and contract["route_lambda"] == 0.1 and contract["reasoning_kl_coef"] == 0.0,
          "dataset does not match the pinned SeeNav Route-SFT recipe")
    for name, spec in manifest["files"].items():
        path = member(root, name)
        check(path.stat().st_size == spec["size"] and sha(path) == spec["sha256"], f"changed file: {name}")
    rows, frame_files, identities = {}, defaultdict(dict), set()
    marker, eos = contract["marker_token_id_on_pinned_processor"], contract["eos_token_id"]
    pattern = prompt_pattern(template)
    for split in ("train", "validation"):
        names = f"{split}.jsonl", f"{split}_batches.jsonl"
        check(set(names) <= manifest["files"].keys(), "unsealed rows or batch order")
        rs, batches = (load_rows(member(root, n)) for n in names)
        summary = manifest["splits"][split]
        check(0 < len(rs) == 8 * len(batches) == summary["rows"] and len(batches) == summary["batches"],
              f"wrong split size: {split}")
        for i, batch in enumerate(batches):
            group = rs[8*i:8*i+8]
            check(batch == {"batch": i + 1, "sample_ids": [x["sample_id"] for x in group]}
                  and Counter(x["route"] for x in group) == ROUTE_COUNTS, f"wrong batch order/quota: {split}")
        for row in rs:
            rid, route = row["sample_id"], int(row["route"] == "bad")
            r, suffix = row["assistant_prefix_ids"], row["target_ids"]
            action = ACTION.fullmatch(row["response"].removeprefix(MARKER))
            parsed = parse_decision(row["assistant_prefix"] + row["response"])
            digest = row["image_sha256"]
            check(isinstance(rid, str) and rid and rid not in identities
                  and row["route_label"] == route and action is not None and parsed is not None
                  and row["response"].startswith(MARKER) == bool(route)
                  and parsed[1:] == (bool(route), row["supervision_action"])
                  and pattern.fullmatch(row["prompt_text"]) is not None and MARKER not in row["prompt_text"]
                  and marker not in r and len(suffix) > 1 and suffix[-1] == eos
                  and (suffix[0] == marker) == bool(route) and suffix.count(marker) == route
                  and row["suffix_start"] == row["prompt_tokens"] + len(r)
                  and row["total_tokens"] == row["suffix_start"] + len(suffix) <= CONTEXT_LIMIT
                  and re.fullmatch(r"[0-9a-f]{64}", digest)
                  and row["image"] == f"frames/{digest[:2]}/{digest}.png"
                  and row["image_archive"] in manifest["files"], f"invalid row: {rid}")
            frame_files[row["image_archive"]][row["image"]] = digest
            identities.add(rid)
        rows[split] = rs
    for field in ("scene", "episode", "image_sha256", "source_input_sha256"):
        check({r[field] for r in rows["train"]}.isdisjoint(r[field] for r in rows["validation"]),
              f"training/validation leakage: {field}")
    check(sorted({r["scene"] for r in rows["validation"]}) == manifest["holdout_scenes"]
          and sum(map(len, frame_files.values())) == manifest["unique_frames"], "wrong scene/frame coverage")
    if verify_frames:
        for archive, entries in frame_files.items():
            with ZipFile(member(root, archive)) as z:
                check(sorted(z.namelist()) == sorted(entries), f"unexpected ZIP members: {archive}")
                for name, digest in entries.items():
                    decode_frame(z.read(name), digest)
    return manifest, rows


def register_marker(processor, manifest):
    tokenizer = processor.tokenizer
    expected = manifest["contract"]["marker_token_id_on_pinned_processor"]
    if MARKER not in tokenizer.all_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [MARKER]},
                                     replace_additional_special_tokens=False)
    check(tokenizer.encode(MARKER, add_special_tokens=False) == [expected]
          and expected in tokenizer.all_special_ids
          and tokenizer.eos_token == EOS and tokenizer.eos_token_id == manifest["contract"]["eos_token_id"],
          "tokenizer does not match the sealed parent")
    return expected


def render_native_prompt(tokenizer, row):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt_text"]}],
                                           tokenize=False, add_generation_prompt=True)
    prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    expected = row["rendered_prompt_sha256"]
    if digest_bytes(prompt.encode()) != expected:
        # The sealed legacy pack used the structured HF image-first renderer.
        # It removes exactly one leading newline in this Qwen chat. Verify
        # that legacy digest but train on the actual online SRGPO prompt.
        needle = "<|im_start|>user\n\n<|vision_start|>"
        legacy = prompt.replace(needle, "<|im_start|>user\n<|vision_start|>", 1)
        check(prompt.count(needle) == 1 and digest_bytes(legacy.encode()) == expected,
              "native chat template differs beyond the known image-boundary newline")
    return prompt


def prepare_inputs(root, processor, row, manifest):
    import torch

    with ZipFile(member(root, row["image_archive"])) as z:
        image = decode_frame(z.read(row["image"]), row["image_sha256"])
    # Same chat rendering as the native rollout: one user message, explicit
    # image boundary, native assistant-generation prompt. No rewrite of H.
    tokenizer = processor.tokenizer
    prompt = render_native_prompt(tokenizer, row)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt",
                       truncation=False, add_special_tokens=False)
    h = inputs["input_ids"]
    check(h.shape == (1, row["prompt_tokens"]) and h.shape[-1] <= NATIVE_MAX_PROMPT
          and inputs["image_grid_thw"].tolist() == [manifest["vision_grid_thw"]], "native image/prompt grid changed")
    r, suffix = row["assistant_prefix_ids"], row["target_ids"]
    check(tokenizer.decode(r, skip_special_tokens=False, clean_up_tokenization_spaces=False) == row["assistant_prefix"]
          and tokenizer.encode(row["response"] + EOS, add_special_tokens=False) == suffix, "changed suffix boundary")
    ids = torch.cat((h, h.new_tensor([r + suffix])), dim=1)
    start = h.shape[-1] + len(r)
    check(start == row["suffix_start"] and ids.shape[-1] == row["total_tokens"], "incorrect mask start")
    inputs["input_ids"], inputs["attention_mask"] = ids, torch.ones_like(ids)
    return inputs, start


def route_loss(logits, ids, start, label, marker, *, weight=0.1):
    import torch
    from torch.nn import functional as F

    check(logits.ndim == 2 and ids.ndim == 1 and len(ids) == logits.shape[0]
          and 0 < start < len(ids) and label in (0, 1)
          and (int(ids[start]) == marker) == bool(label), "wrong route supervision position")
    suffix_logits = logits[start-1:-1].float()
    ce = F.cross_entropy(suffix_logits, ids[start:], reduction="mean")
    z = suffix_logits[0]
    binary_logit = z[marker] - torch.logsumexp(torch.cat((z[:marker], z[marker+1:])), dim=0)
    bce = F.binary_cross_entropy_with_logits(binary_logit, binary_logit.new_tensor(float(label)))
    return {"loss": ce + weight * bce, "suffix_ce": ce, "marker_bce": bce}
