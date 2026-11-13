"""Plan row-local Teacher context without changing the student rollout."""

from collections import Counter, defaultdict
from time import monotonic

import torch

from seenav_opsd.teacher import SKILL_FILE, pack_teacher_tensors
from . import hints
from .protocol import decode_decision, marker_token_id


def tag_prefix(tokenizer):
    skill = SKILL_FILE.read_text(encoding="utf-8").strip()
    old = "in the original JSON format."
    if skill.count(old) != 1:
        raise ValueError("expected the pinned static SeeNav skill")
    skill = skill.replace(old, "inside the native <action> tag.")
    ids = tokenizer.encode(f"[Privileged Skill Information]\n{skill}\n\n", add_special_tokens=False)
    if not ids or len(ids) > 256 or set(ids) & set(tokenizer.all_special_ids):
        raise ValueError("expected a short static tag Teacher prefix")
    return ids


def prefixes(trainer, batch, *, enabled):
    tokenizer = trainer.tokenizer
    static = tag_prefix(tokenizer)
    if not enabled:
        return static, {}, {}

    marker = marker_token_id(tokenizer)
    if marker is None:
        raise ValueError("trained Actor must preserve the single special <RETHINK> token")
    native = batch.batch
    rows = len(native["responses"])
    width = native["responses"].shape[-1]
    active = native["attention_mask"][:, -width:].bool()
    marked = (native["responses"].eq(marker) & active)
    candidates = marked.any(dim=-1).nonzero(as_tuple=True)[0].tolist()
    stats = {"rethink/response_rows": rows, "rethink/marker_rows": len(candidates),
             "rethink/eligible_rows": 0, "rethink/unique_routed": 0, "rethink/hint_requests": 0,
             "rethink/rows_with_local_hint": 0, "rethink/unique_with_local_hint": 0,
             "rethink/unique_static_fallback": 0, "rethink/malformed_marker_rows": 0,
             "rethink/execution_mismatch_rows": 0}
    if not candidates:
        return static, {}, stats
    other = batch.non_tensor_batch
    evidence = other.get("rethink_evidence")
    anchors = other.get("anchor_obs")
    uids = other.get("traj_uid")
    if any(x is None or len(x) != rows for x in (evidence, anchors, uids)):
        raise ValueError("missing pinned SeeNav rollout evidence; verify the method overlay")
    timeline, selected, locations = defaultdict(dict), defaultdict(dict), defaultdict(list)
    # The native divisibility adjustment may copy or drop rows; use the
    # trajectory and physical rollout index, not a mutable batch row number.
    for i in candidates:
        ids = native["responses"][i, active[i]].tolist()
        decoded = decode_decision(tokenizer, ids, marker)
        if not decoded.routed:
            stats["rethink/malformed_marker_rows"] += 1
            continue
        reasoning, _, move = decoded.parsed
        ctx = evidence[i]
        # Physical action failure (e.g. a collision) still carries the action
        # ID and is valid hindsight. Do not route a step with no matching move.
        if ctx["executed_action"] != move:
            stats["rethink/execution_mismatch_rows"] += 1
            continue
        uid, step = str(uids[i]), int(ctx["step"])
        selected[uid].setdefault(step, (step, ctx["prompt"], reasoning, move, anchors[i]))
        locations[uid, step].append(i)
    stats["rethink/eligible_rows"] = sum(map(len, locations.values()))
    stats["rethink/unique_routed"] = len(locations)
    if not locations:
        return static, {}, stats
    for uid, ctx in zip(uids, evidence):
        uid = str(uid)
        if uid in selected:
            timeline[uid].setdefault(int(ctx["step"]), ctx)

    started = monotonic()
    try:
        settings = hints.from_env()
    except ValueError:
        # Launch checks configuration before allocating GPUs. If the Ray
        # driver did not inherit it, keep the native update and surface it.
        stats["rethink/unique_static_fallback"] = len(locations)
        stats["rethink/fallback_api_config"] = len(locations)
        return static, {}, stats
    received, errors, requests = hints.request_hints(settings, timeline, selected)
    stats["rethink/hint_elapsed_s"] = monotonic() - started
    stats["rethink/hint_requests"] = requests
    config = trainer.config.actor_rollout_ref.rollout
    limit = int(config.max_model_len)
    local, failures = {}, Counter()
    for key, indices in locations.items():
        hint = received.get(key)
        if hint is None:
            failures[errors.get(key, "missing_hint")] += 1
            continue
        extra = tokenizer.encode(f"[Local Navigation Hint]\n{hint}\n\n", add_special_tokens=False)
        if (not extra or len(extra) > 96 or set(extra) & set(tokenizer.all_special_ids)
                or native["input_ids"].shape[-1] + len(static) + len(extra) > limit):
            failures["prefix_contract"] += 1
            continue
        for i in indices:
            local[i] = static + extra
        stats["rethink/unique_with_local_hint"] += 1
    stats["rethink/rows_with_local_hint"] = len(local)
    stats["rethink/unique_static_fallback"] = sum(failures.values())
    stats.update({f"rethink/fallback_{name}": n for name, n in failures.items()})
    return static, local, stats


def pack_prefixes(student, static, local, max_model_len, pad_id):
    if not local:
        return pack_teacher_tensors(student, static, max_model_len)

    ids, pos = student["input_ids"], student["position_ids"]
    result_width = ids.shape[-1] + max(len(static), *(len(v) for v in local.values()))
    groups = defaultdict(list)
    for i in range(len(ids)):
        groups[tuple(local.get(i, static))].append(i)
    result = {"input_ids": ids.new_full((len(ids), result_width), pad_id),
              "attention_mask": student["attention_mask"].new_zeros((len(ids), result_width)),
              "position_ids": pos.new_ones((*pos.shape[:-1], result_width)),
              "responses": student["responses"]}
    for prefix, indices in groups.items():
        ix = ids.new_tensor(indices)
        subset = {k: student[k].index_select(0, ix) for k in result}
        packed = pack_teacher_tensors(subset, prefix, max_model_len)
        for name in ("input_ids", "attention_mask", "position_ids"):
            result[name][ix, ..., -packed[name].shape[-1]:] = packed[name]
    return result
