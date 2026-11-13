"""Stage 2: native SeeNav SRGPO + OPSD, optionally with Actor-routed local hints."""

import argparse
import json
from pathlib import Path
import re
import tempfile

import reproduce as base

RUNTIME = base.ROOT / ".deps" / "SeeNav-Agent-rethink-protocol-v2"
PATCH = base.ROOT / "patches" / "rethink-rollout.patch"
ROLLOUT = "verl-agent/agent_system/multi_turn_rollout/rollout_loop.py"
EXTRA_FILES = {ROLLOUT: "227a379b406a2c050b2b10a5b9629a5d33cf4261ddca73a46b9601c2b426877c"}
VARIANTS = ("rethink", "tag-control", "sft-static")


def verify_runtime(source):
    base.verify_opsd_source(source, extra_files=EXTRA_FILES)
    base.git(source, "apply", "--unidiff-zero", "--reverse", "--check", str(PATCH))
    return source


def fetch_runtime(official, target=RUNTIME):
    base.verify_source(official)
    if target.resolve() == official.resolve():
        raise ValueError("Keep the official source and method runtime separate.")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rethink-fetch-", dir=target.parent) as tmp:
            staging = Path(tmp) / "checkout"
            base.git(Path(tmp), "clone", "--no-hardlinks", "--no-checkout", str(official), str(staging))
            base.git(staging, "config", "core.autocrlf", "false")
            base.git(staging, "remote", "set-url", "origin", base.LOCK["repository"])
            base.git(staging, "checkout", "--detach", base.LOCK["commit"])
            base.git(staging, "apply", str(base.OPSD_PATCH))
            base.git(staging, "apply", "--unidiff-zero", str(PATCH))
            verify_runtime(staging)
            staging.rename(target)
    return verify_runtime(target)


def adapt_runner(script, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown SeeNav variant")
    script = base.replace_one(script, "actor_rollout_ref.actor.output_format=json",
                              "actor_rollout_ref.actor.output_format=tags")
    if variant != "tag-control":
        script = base.replace_one(script, f"actor_rollout_ref.model.path={base.LOCK['base_model']}",
                                  'actor_rollout_ref.model.path="${SEENAV_STAGE1_CHECKPOINT}"')
    if variant == "rethink":
        option = "    ++actor_rollout_ref.actor.rethink_local_hints=True " + chr(92) + "\n"
        anchor = "    trainer.val_before_train=True $@\n"
        script = base.replace_one(script, anchor, option + anchor)
    return script


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("fetch", "check", "plan", "train"))
    p.add_argument("--source", default=base.DEFAULT_SOURCE, type=Path)
    p.add_argument("--direct", action="store_true", help="Fetch the official public checkout over direct HTTPS")
    p.add_argument("--variant", default="rethink", choices=VARIANTS)
    p.add_argument("--stage1", type=Path, help="Completed local Stage-1 Hugging Face export")
    p.add_argument("--expect-stage1-dataset", help="Require this SHA-256 of the Stage-1 dataset manifest")
    p.add_argument("--expect-stage1-step", type=int, help="Require this completed Route-SFT update")
    p.add_argument("--gpus", help="Four allocated GPUs, or use CUDA_VISIBLE_DEVICES")
    p.add_argument("--output", type=Path, help="New output directory for Stage 2")
    args = p.parse_args()
    source = args.source.resolve()
    if args.command == "fetch":
        base.fetch_source(source, args.direct)
        fetch_runtime(source)
        print(f"Verified SeeNav {base.LOCK['commit']} + pinned OPSD + isolated tag overlay.")
        return
    base.verify_source(source)
    verify_runtime(RUNTIME)
    binding = {
        "method": {"rethink": "VP + SRGPO + OPSD + RETHINK",
                   "tag-control": "VP tags + SRGPO + OPSD",
                   "sft-static": "VP tags + Route-SFT + SRGPO + static OPSD"}[args.variant],
        "variant": args.variant,
        "actor_init": "base" if args.variant == "tag-control" else "route_sft",
        "output_format": "tags",
        "rethink_rollout_patch_sha256": base.text_sha(PATCH),
        "rethink_rollout_files": EXTRA_FILES,
        "rethink_local_hints": args.variant == "rethink",
    }
    checkpoint = args.stage1.resolve() if args.stage1 else None
    expected_hash, expected_step = args.expect_stage1_dataset, args.expect_stage1_step
    if expected_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        p.error("--expect-stage1-dataset must be a SHA-256 hex digest")
    if expected_step is not None and expected_step < 1:
        p.error("--expect-stage1-step must be positive")
    if (expected_hash is not None or expected_step is not None) and checkpoint is None:
        p.error("Stage-1 expectations require --stage1")
    if checkpoint is not None:
        from seenav_rethink.assets import verify_stage1

        if args.variant == "tag-control":
            raise ValueError("Tag control starts from the baseline base model, without Stage 1.")
        binding["stage1"] = verify_stage1(checkpoint, expected_dataset=expected_hash, expected_step=expected_step)
    if args.command == "check":
        print(json.dumps({**binding, "official_commit": base.LOCK["commit"], "sdar": base.LOCK["sdar"],
                          "check": "CPU source/assets only, not GPU, simulator or API"}, indent=2))
    elif args.command == "plan":
        native = (source / base.LOCK["train_launcher"]).read_text(encoding="utf-8")
        runner = base.enable_opsd(base.isolated_paths(base.four_gpu_launcher(native)))
        print(adapt_runner(runner, args.variant), end="")
    else:
        if args.output is None or (args.variant != "tag-control" and checkpoint is None):
            raise ValueError("Provide --output and, for a Stage-1 variant, --stage1.")
        if checkpoint:
            base.check_run_path(checkpoint)
            if args.output.resolve().is_relative_to(checkpoint):
                raise ValueError("Use an output path outside the Stage-1 checkpoint.")
        if args.variant == "rethink":
            from seenav_rethink.hints import from_env

            settings = from_env()  # syntactic configuration check; no request or charge
            binding["hint_api"] = {"model": settings.model, "parallel": settings.parallel,
                                   "images_per_request": settings.images_per_request,
                                   "attempts": settings.attempts, "timeout": settings.timeout,
                                   "max_tokens": settings.max_tokens, "update_budget": settings.update_budget}
            binding["hint_prompt_sha256"] = base.text_sha(base.ROOT / "seenav_rethink/local_hint_system.txt")
        base.launch(source, args.output.resolve(), args.gpus, runtime=RUNTIME, verify_runtime=verify_runtime,
                    adapt_runner=lambda s: adapt_runner(s, args.variant),
                    extra_env={"SEENAV_STAGE1_CHECKPOINT": str(checkpoint)} if checkpoint else {},
                    extra_binding=binding)


if __name__ == "__main__":
    main()
