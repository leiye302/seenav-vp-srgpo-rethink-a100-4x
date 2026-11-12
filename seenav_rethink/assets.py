"""Bind both stages to the native vision model and an explicit checkpoint."""

from pathlib import Path

from reproduce import LOCK, sha
from .data import BASE_REVISION, check, load_json
from .protocol import MARKER


def model_files(model):
    config = load_json(model / "config.json")
    check(config["model_type"] == "qwen2_5_vl", "expected Qwen2.5-VL")
    required = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
    index = model / "model.safetensors.index.json"
    if index.exists():
        required.add(index.name)
        weights = set(load_json(index)["weight_map"].values())
    else:
        weights = {"model.safetensors"}
    check(weights and all(Path(n).name == n and n.endswith(".safetensors") for n in weights),
          "missing or unsafe weight shards")
    check(weights == {p.name for p in model.glob("*.safetensors")}, "missing or mixed weight shards")
    check(required <= {p.name for p in model.iterdir() if p.is_file()}, "processor/model files missing")
    return weights


def verify_entries(root, files, *, skip=()):
    for name, spec in files.items():
        check(Path(name).name == name, "nonlocal model member")
        if name not in skip:
            path = root / name
            check(path.stat().st_size == spec["size"] and sha(path) == spec["sha256"], f"model SHA differs: {name}")


def verify_parent(model, receipt=None):
    weights = model_files(model)
    if receipt:
        spec = load_json(receipt)
        files = spec["files"]
        check(spec["model"] == LOCK["base_model"] and spec["revision"] == BASE_REVISION
              and weights <= files.keys(), "unexpected parent model receipt")
        verify_entries(model, files)
        return sha(receipt)
    check(model.name == BASE_REVISION and model.parent.name == "snapshots"
          and model.parent.parent.name == "models--Qwen--Qwen2.5-VL-3B-Instruct",
          "use the pinned Hub cache snapshot, or supply a verified model receipt")
    return None


def verify_stage1(model):
    from transformers import AutoTokenizer

    weights = model_files(model)
    manifest = model / "checkpoint_manifest.json"
    spec = load_json(manifest)
    binding, files = spec["binding"], spec["files"]
    recipe = binding["recipe"]
    check(binding["model"] == LOCK["base_model"] and binding["revision"] == BASE_REVISION
          and spec["step"] == binding["steps"] and spec["step"] > 0
          and recipe["route_lambda"] == 0.1 and recipe["reasoning_ce"] is False
          and recipe["reference_kl_coef"] == 0.0 and recipe["epochs"] == 1
          and recipe["good_per_batch"] == 5 and recipe["bad_per_batch"] == 3
          and weights <= files.keys()
          and {p.name for p in model.iterdir() if p.is_file()} == set(files) | {manifest.name},
          "expected a completed SeeNav Route-SFT export and its manifest")
    # Optimizer state is never used by the Stage-2 actor.
    verify_entries(model, files, skip=("training_state.pt",))
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False, local_files_only=True)
    ids = tok.encode(MARKER, add_special_tokens=False)
    config = load_json(model / "config.json")
    vocab = config.get("text_config", config)["vocab_size"]
    check(len(ids) == 1 and ids[0] in tok.all_special_ids and ids[0] < vocab
          and tok.decode(ids, skip_special_tokens=False) == MARKER, "missing native marker ID")
    return {"step": spec["step"], "manifest_sha256": sha(manifest), "marker_token_id": ids[0]}
