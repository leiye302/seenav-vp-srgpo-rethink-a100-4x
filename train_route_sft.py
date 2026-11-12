"""Stage 1: one epoch of native SeeNav Route-SFT (5 Good / 3 Bad)."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import time

from reproduce import DEFAULT_SOURCE, LOCK, sha, verify_source, write_new
from seenav_rethink import data
from seenav_rethink.assets import verify_parent

RECIPE = dict(epochs=1, global_batch_size=8, good_per_batch=5, bad_per_batch=3,
              micro_batch_size=1, lr=2e-6, betas=[0.9, 0.95], weight_decay=0.01,
              warmup_steps=60, schedule="cosine", max_grad_norm=1.0, seed=0,
              full_parameter_training=True, param_dtype="float32", autocast_dtype="bfloat16",
              attention="sdpa", gradient_checkpointing=True, reasoning_ce=False,
              route_lambda=0.1, reference_kl_coef=0.0, eval_every=100)


def emit(path, value):
    write_new(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_processor(model):
    from transformers import AutoProcessor, __version__

    data.check(__version__ == "4.51.1", "use the same Transformers 4.51.1 as native SeeNav SRGPO")
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=False, local_files_only=True, use_fast=True)
    data.check(type(processor.image_processor).__name__ == "Qwen2VLImageProcessorFast",
               "native VP requires the Qwen fast image processor; verify torchvision")
    return processor


def forward(model, root, processor, row, manifest, marker, *, full_head=False):
    import torch

    inputs, start = data.prepare_inputs(root, processor, row, manifest)
    inputs = {k: v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}
    ids = inputs["input_ids"][0]
    keep = ids.numel() - start + 1
    # Same transformer forward and gradient through H/R. Only the unused
    # rows of the *linear* vocabulary projection are skipped.
    hook = None if full_head else model.get_output_embeddings().register_forward_pre_hook(
        lambda _, args: (args[0][:, -keep:, :], *args[1:]))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs, use_cache=False).logits[0, -keep:]
            return data.route_loss(logits, ids[start-1:], 1, row["route_label"], marker)
    finally:
        if hook is not None:
            hook.remove()


def evaluate(model, fwd, rows, step, output):
    import torch

    model.eval()
    totals, counts = {}, Counter()
    with torch.random.fork_rng(devices=[0]), torch.inference_mode():
        for row in rows:
            values = {k: float(v) for k, v in fwd(row).items()}
            for group in ("all", row["route"]):
                counts[group] += 1
                totals.setdefault(group, Counter()).update(values)
    metrics = {g: {"n": n, **{k: v/n for k, v in totals[g].items()}} for g, n in counts.items()}
    emit(output / f"eval_loss_{step:04d}.json", {"step": step, "teacher_forced": True, **metrics})
    model.train()


def save(model, processor, optimizer, scheduler, output, binding, save_state):
    import torch

    step = binding["steps"]
    final = output / f"global_step_{step}"
    temp = final.with_suffix(".incomplete")
    temp.mkdir()
    model.save_pretrained(temp, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(temp)
    if save_state:
        torch.save(dict(step=step, next_global_batch=step, binding=binding, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate()),
                   temp / "training_state.pt")
    files = {f.name: {"size": f.stat().st_size, "sha256": sha(f)} for f in temp.iterdir() if f.is_file()}
    data.check(any(n.endswith(".safetensors") for n in files), "weights not saved")
    emit(temp / "checkpoint_manifest.json", {"step": step, "binding": binding, "files": files,
         "mode": "full_resume" if save_state else "weights_only"})
    temp.rename(final)
    print(f"Stage 1 complete: {final}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("check", "train"))
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--source", default=DEFAULT_SOURCE, type=Path)
    p.add_argument("--model", type=Path, help="Pinned local Qwen2.5-VL-3B-Instruct snapshot")
    p.add_argument("--model-receipt", type=Path, help="SHA-256 receipt for a relocated base model")
    p.add_argument("--verify-frames", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--save-training-state", action="store_true")
    args = p.parse_args()
    root = args.dataset.resolve()
    manifest, rows = data.verify_package(root, data.native_template(verify_source(args.source.resolve())),
                                         verify_frames=args.verify_frames)
    steps = len(rows["train"]) // 8
    binding = dict(dataset_sha256=sha(root / "manifest.json"), model=LOCK["base_model"],
                   revision=data.BASE_REVISION, steps=steps, recipe=RECIPE)
    if args.command == "check":
        if args.model:
            verify_parent(args.model, args.model_receipt)
            processor = load_processor(args.model)
            data.register_marker(processor, manifest)
            for group in rows.values():
                for kind in ("good", "bad"):
                    data.prepare_inputs(root, processor, next(r for r in group if r["route"] == kind), manifest)
        print(json.dumps({**binding, "train_rows": len(rows["train"]), "val_rows": len(rows["validation"]),
                          "training_started": False}, indent=2))
        return
    data.check(args.model and args.output, "train needs --model and a new --output directory")
    output = args.output.resolve()
    data.check(not output.exists() and not output.is_relative_to(root)
               and not output.is_relative_to(args.model.resolve()), "output must be new and outside data/model")
    binding["model_receipt_sha256"] = verify_parent(args.model, args.model_receipt)
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

    data.check(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
               and torch.cuda.get_device_properties(0).total_memory >= 70 * 1024**3,
               "expose one allocated BF16-capable 80 GB GPU for this full-parameter recipe")
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    processor = load_processor(args.model)
    marker = data.register_marker(processor, manifest)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True,
        trust_remote_code=False).to("cuda")
    data.check(marker < model.get_input_embeddings().weight.shape[0]
               and marker < model.get_output_embeddings().weight.shape[0], "marker outside original vocabulary")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], betas=tuple(RECIPE["betas"]),
                                  weight_decay=RECIPE["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, RECIPE["warmup_steps"], steps)
    output.mkdir(parents=True)
    bytes_for_weights = sum(p.numel()*p.element_size() for p in model.parameters())
    data.check(shutil.disk_usage(output).free > (3 if args.save_training_state else 1)*bytes_for_weights + 2*1024**3,
               "insufficient checkpoint space")

    def fwd(row, **kw):
        return forward(model, root, processor, row, manifest, marker, **kw)

    # Test the head-only optimization before any updates.
    model.eval()
    row = max(rows["train"], key=lambda x: x["total_tokens"])
    with torch.inference_mode():
        full, slim = fwd(row, full_head=True), fwd(row)
        for name in full:
            torch.testing.assert_close(full[name], slim[name], rtol=0.01, atol=0.02)
    emit(output / "binding.json", binding)
    evaluate(model, fwd, rows["validation"], 0, output)
    with (output / "training_metrics.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for step in range(1, steps+1):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            totals = Counter()
            for row in rows["train"][8*(step-1):8*step]:
                terms = fwd(row)
                data.check(bool(torch.isfinite(terms["loss"])), "nonfinite training loss")
                (terms["loss"] / 8).backward()
                totals.update({k: float(v.detach())/8 for k, v in terms.items()})
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["max_grad_norm"], error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            log.write(json.dumps({"step": step, "steps": steps, "seconds": time.monotonic()-started,
                                  "lr": lr, "grad_norm": float(norm), **totals}, allow_nan=False) + "\n")
            if step % RECIPE["eval_every"] == 0 or step == steps:
                evaluate(model, fwd, rows["validation"], step, output)
    save(model, processor, optimizer, scheduler, output, binding, args.save_training_state)


if __name__ == "__main__":
    main()
