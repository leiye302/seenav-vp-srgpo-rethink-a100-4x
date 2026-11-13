"""Train SeeNav Route-SFT, then native VP + SRGPO + OPSD with Actor-routed hints."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import reproduce as base
import rethink
from seenav_rethink.assets import model_files, stage1_binding, verify_parent
from seenav_rethink.data import check, load_json

STAGE2_UPDATES = 150  # one native training batch per epoch, 150 epochs
HINT_COUNTERS = (
    "unique_routed", "unique_with_local_hint", "unique_static_fallback", "hint_requests",
)


def reviewed_dataset(dataset):
    spec = load_json(base.ROOT / "dataset_release.json")
    manifest = dataset / "manifest.json"
    digest = base.sha(manifest)
    check(digest == spec["manifest_sha256"], "The pipeline needs the reviewed GitHub Release dataset.")
    steps = spec["splits"]["train"]["batches"]
    check(type(steps) is int and steps > 0, "Invalid released training schedule.")
    return digest, steps


def commands(dataset, source, model, receipt, output, devices, digest, steps, save_state):
    stage1_dir, stage2_dir = output / "stage1", output / "stage2"
    checkpoint = stage1_dir / f"global_step_{steps}"
    first = [sys.executable, str(base.ROOT / "train_route_sft.py"), "train",
             "--source", str(source), "--dataset", str(dataset), "--model", str(model),
             "--output", str(stage1_dir)]
    if receipt is not None:
        first.extend(["--model-receipt", str(receipt)])
    if save_state:
        first.append("--save-training-state")
    second = [sys.executable, str(base.ROOT / "rethink.py"), "train", "--variant", "rethink",
              "--source", str(source), "--stage1", str(checkpoint),
              "--expect-stage1-dataset", digest, "--expect-stage1-step", str(steps),
              "--gpus", ",".join(devices), "--output", str(stage2_dir)]
    return first, second, checkpoint


def stage2_result(output, updates=STAGE2_UPDATES):
    """Check the native completion markers and summarize applied hints, not just requests."""
    check(load_json(output / "exit.json")["returncode"] == 0, "Stage 2 did not exit cleanly.")
    ckpt_root = output / "checkpoints"
    last = (ckpt_root / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip()
    check(last == str(updates), "Stage 2 did not save the expected final update.")
    final = ckpt_root / f"global_step_{updates}"
    actor = final / "actor"
    required = [final / "data.pt"]
    for rank in range(base.GPU_COUNT):
        for name in ("model", "optim", "extra_state"):
            required.append(actor / f"{name}_world_size_{base.GPU_COUNT}_rank_{rank}.pt")
    check(all(p.is_file() and p.stat().st_size > 0 for p in required), "Stage-2 checkpoint is incomplete.")
    export = actor / "huggingface"
    weights = model_files(export)
    check(all((export / name).stat().st_size > 0 for name in weights), "Stage-2 HF export is incomplete.")

    totals = dict.fromkeys(HINT_COUNTERS, 0)
    for step in range(1, updates + 1):
        metrics = load_json(output / "metrics" / f"metrics_{step}.json")
        check(metrics["training/global_step"] == step, "Stage-2 metrics are not aligned with updates.")
        counts = {k: metrics[f"rethink/{k}"] for k in HINT_COUNTERS}
        check(all(type(n) is int and n >= 0 for n in counts.values())
              and counts["unique_routed"] == counts["unique_with_local_hint"] + counts["unique_static_fallback"],
              "Stage-2 hint counters are inconsistent.")
        for k, n in counts.items():
            totals[k] += n
    routed, applied = totals["unique_routed"], totals["unique_with_local_hint"]
    status = "applied" if applied else ("all_routed_fell_back" if routed else "no_actor_triggers")
    return {
        "updates": updates,
        "native_checkpoint": final.relative_to(output.parent).as_posix(),
        "hf_export": export.relative_to(output.parent).as_posix(),
        "hint_counts": totals,
        "hint_coverage": applied / routed if routed else None,
        "local_hint_applied": applied > 0,
        "local_hint_status": status,
    }


def run_stage(argv, logfile, env):
    with logfile.open("xb") as log:
        result = subprocess.run(argv, cwd=base.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"Training process exited with code {result.returncode}; see {logfile}")


def record(path, payload):
    base.write_new(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run(dataset, source, model, receipt, output, devices, digest, steps, save_state):
    first, second, checkpoint = commands(dataset, source, model, receipt, output, devices, digest, steps, save_state)
    output.mkdir(parents=True, exist_ok=False)
    # The vision API is needed only by Stage 2. Its settings are not command
    # line arguments and are not written by this pipeline.
    sft_env = {k: v for k, v in os.environ.items() if not k.startswith("SEENAV_HINT_")}
    sft_env["CUDA_VISIBLE_DEVICES"] = devices[0]
    print(f"Stage 1: Route-SFT; follow {output / 'stage1.log'}", flush=True)
    run_stage(first, output / "stage1.log", sft_env)

    manifest, _ = stage1_binding(checkpoint, expected_dataset=digest, expected_step=steps)
    record(output / "handoff.json", {
        "stage1": checkpoint.relative_to(output).as_posix(),
        "stage1_manifest_sha256": base.sha(manifest),
        "dataset_manifest_sha256": digest,
        "stage1_updates": steps,
        "stage2_variant": "rethink",
        "stage2_updates_expected": STAGE2_UPDATES,
    })
    # The Stage-2 entry also checks the actual weight/processor digests and
    # marker token before starting its GPU workers.
    print(f"Stage 2: native SRGPO + OPSD + routed hints; follow {output / 'stage2/train.log'}", flush=True)
    run_stage(second, output / "stage2.log", {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices)})
    result = stage2_result(output / "stage2")
    record(output / "complete.json", result)
    if not result["local_hint_applied"]:
        print("WARNING: Stage 2 finished without an applied local hint; this run does not demonstrate hint training.",
              flush=True)
    print(f"Verified both stages; see {output / 'complete.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("check", "train"))
    p.add_argument("--dataset", type=Path, default=base.ROOT / "data/seenav-reviewed-v2")
    p.add_argument("--source", type=Path, default=base.DEFAULT_SOURCE)
    p.add_argument("--model", type=Path, help="Pinned local Qwen2.5-VL-3B-Instruct snapshot")
    p.add_argument("--model-receipt", type=Path, help="SHA-256 receipt for a relocated base model")
    p.add_argument("--output", type=Path, help="New directory for both stages")
    p.add_argument("--gpus", help="Four allocated A100 80 GB devices, or use CUDA_VISIBLE_DEVICES")
    p.add_argument("--save-training-state", action="store_true", help="Also save the Stage-1 optimizer/scheduler/RNG")
    args = p.parse_args()
    if args.command == "train" and (args.model is None or args.output is None):
        p.error("train requires --model and --output")
    if args.model_receipt is not None and args.model is None:
        p.error("--model-receipt requires --model")
    source, dataset = base.verify_source(args.source.resolve()), args.dataset.resolve()
    runtime = rethink.verify_runtime(rethink.RUNTIME)
    digest, steps = reviewed_dataset(dataset)
    if args.command == "check":
        cmd = [sys.executable, str(base.ROOT / "train_route_sft.py"), "check",
               "--source", str(source), "--dataset", str(dataset)]
        if args.model:
            cmd.extend(["--model", str(args.model.resolve())])
        if args.model_receipt:
            cmd.extend(["--model-receipt", str(args.model_receipt.resolve())])
        subprocess.run(cmd, cwd=base.ROOT, check=True)
        print(json.dumps({"stage1_updates": steps, "stage2_updates_expected": STAGE2_UPDATES,
                          "stage2": "SRGPO + gated OPSD + Actor-routed local hints",
                          "gpu_or_api_used": False}, indent=2))
        return
    check(sys.platform.startswith("linux"), "The full pipeline needs Linux/CUDA and the native SeeNav environment.")
    out, model = args.output.resolve(), args.model.resolve()
    base.check_run_path(out)
    check(not out.exists() and not any(out.is_relative_to(x) for x in (source, runtime, dataset, model)),
          "Use a new output directory outside the source, model and dataset directories.")
    devices = base.gpu_ids(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    from seenav_rethink.hints import from_env

    from_env()  # verify syntax only, before committing to Stage 1; no request
    base.hardware_check({**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices)})
    receipt = args.model_receipt.resolve() if args.model_receipt else None
    verify_parent(model, receipt)
    run(dataset, source, model, receipt, out, devices, digest, steps, args.save_training_state)


if __name__ == "__main__":
    main()
