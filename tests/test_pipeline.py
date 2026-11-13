import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import final_eval
import rethink
import run_pipeline as pipeline
from seenav_rethink import assets
from seenav_rethink.data import BASE_REVISION


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fake_model(model):
    write(model / "config.json", {"model_type": "qwen2_5_vl", "vocab_size": 151936})
    write(model / "preprocessor_config.json", {})
    write(model / "tokenizer_config.json", {})
    (model / "model.safetensors").write_bytes(b"mock weights")


def fake_stage1(model, digest, step):
    fake_model(model)
    binding = {
        "model": pipeline.base.LOCK["base_model"],
        "revision": BASE_REVISION,
        "dataset_sha256": digest,
        "steps": step,
        "recipe": {"route_lambda": .1, "reasoning_ce": False, "reference_kl_coef": 0.,
                   "epochs": 1, "good_per_batch": 5, "bad_per_batch": 3},
    }
    files = {p.name: {"size": p.stat().st_size, "sha256": pipeline.base.sha(p)} for p in model.iterdir()}
    write(model / "checkpoint_manifest.json", {"step": step, "binding": binding, "files": files})


def fake_stage2(out, count, *, hints=True):
    write(out / "exit.json", {"returncode": 0})
    root = out / "checkpoints"
    final = root / f"global_step_{count}"
    actor = final / "actor"
    fake_model(actor / "huggingface")
    (root / "latest_checkpointed_iteration.txt").write_text(str(count), encoding="utf-8")
    (final / "data.pt").write_bytes(b"mock dataloader")
    for rank in range(pipeline.base.GPU_COUNT):
        for name in ("model", "optim", "extra_state"):
            (actor / f"{name}_world_size_{pipeline.base.GPU_COUNT}_rank_{rank}.pt").write_bytes(b"mock shard")
    for i in range(1, count + 1):
        write(out / "metrics" / f"metrics_{i}.json", {
            "training/global_step": i,
            "rethink/unique_routed": 2,
            "rethink/unique_with_local_hint": int(hints),
            "rethink/unique_static_fallback": 2 - int(hints),
            "rethink/hint_requests": 2,
        })


class PipelineTest(unittest.TestCase):
    def test_train_sequence_hands_off_exact_step_and_gpu_allocation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "run"
            digest = "a" * 64
            calls = []

            def launch(argv, log, env, **kwargs):
                calls.append((argv, log.name, env["CUDA_VISIBLE_DEVICES"]))
                if log.name == "stage1.log":
                    self.assertNotIn("SEENAV_HINT_API_KEY", env)
                    self.assertEqual(argv[2], "train")
                    self.assertFalse((output / "handoff.json").exists())
                    fake_stage1(output / "stage1/global_step_736", digest, 736)
                elif log.name == "stage2.log":
                    self.assertEqual(env["SEENAV_HINT_API_KEY"], "test-placeholder")
                    self.assertEqual(argv[argv.index("--variant") + 1], "rethink")
                    self.assertEqual(argv[argv.index("--expect-stage1-dataset") + 1], digest)
                    self.assertEqual(argv[argv.index("--expect-stage1-step") + 1], "736")
                    ckpt = Path(argv[argv.index("--stage1") + 1])
                    self.assertEqual(ckpt, output / "stage1/global_step_736")
                    self.assertTrue((ckpt / "checkpoint_manifest.json").is_file())
                    self.assertEqual(pipeline.load_json(output / "handoff.json")["stage2_variant"], "rethink")
                    fake_stage2(output / "stage2", pipeline.STAGE2_UPDATES)
                else:
                    self.assertEqual(log.name, "final_eval.log")
                    self.assertNotIn("SEENAV_HINT_API_KEY", env)
                    self.assertTrue(kwargs["resume_log"])
                    self.assertTrue((output / "training_complete.json").is_file())
                    self.assertFalse((output / "complete.json").exists())
                    self.assertEqual(argv[argv.index("--format") + 1], "tags")
                    self.assertIn("--require-marker", argv)
                    self.assertEqual(Path(argv[argv.index("--model") + 1]),
                                     output / "stage2/checkpoints/global_step_150/actor/huggingface")

            summary = {"episodes": 60, "successes": 30, "success_rate": .5,
                       "tasks_sha256": "c"*64, "contract_sha256": "d"*64}
            with patch.dict(os.environ, {"SEENAV_HINT_API_KEY": "test-placeholder"}), patch.object(
                    pipeline, "run_stage", side_effect=launch), patch.object(
                    final_eval, "verify_result", return_value=summary):
                pipeline.run(root / "data", root / "source", root / "model", None, output,
                             ["2", "3", "6", "7"], digest, 736, False)
            self.assertEqual([(log, ids) for _, log, ids in calls],
                             [("stage1.log", "2"), ("stage2.log", "2,3,6,7"), ("final_eval.log", "2")])
            result = pipeline.load_json(output / "complete.json")
            self.assertEqual(result["hint_coverage"], 0.5)
            self.assertTrue(result["local_hint_applied"])
            self.assertEqual(result["local_hint_status"], "applied")
            self.assertEqual(result["updates"], pipeline.STAGE2_UPDATES)
            self.assertEqual(result["final_eval"]["episodes"], 60)
            self.assertEqual(result["final_eval"]["success_rate"], .5)

    def test_no_second_stage_after_failed_or_wrong_sft(self):
        for failure in ("subprocess", "wrong_data", "wrong_step"):
            with self.subTest(failure=failure), TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "run"
                digest = "b" * 64

                def launch(argv, log, env):
                    self.assertEqual(log.name, "stage1.log")
                    if failure == "subprocess":
                        raise SystemExit(1)
                    fake_stage1(output / "stage1/global_step_736",
                                "a" * 64 if failure == "wrong_data" else digest,
                                735 if failure == "wrong_step" else 736)

                with patch.object(pipeline, "run_stage", side_effect=launch) as runner:
                    with self.assertRaises(SystemExit if failure == "subprocess" else ValueError):
                        pipeline.run(root / "data", root / "source", root / "model", None,
                                     output, ["0", "1", "2", "3"], digest, 736, False)
                self.assertEqual(runner.call_count, 1)
                self.assertFalse((output / "handoff.json").exists())
                self.assertFalse((output / "complete.json").exists())

    def test_stage2_result_separates_fallback_and_rejects_incomplete_artifacts(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage2"
            fake_stage2(out, 2, hints=False)
            result = pipeline.stage2_result(out, updates=2)
            self.assertEqual(result["hint_coverage"], 0.)
            self.assertFalse(result["local_hint_applied"])
            self.assertEqual(result["local_hint_status"], "all_routed_fell_back")
            self.assertEqual(result["hint_counts"]["unique_static_fallback"], 4)
            (out / "checkpoints/global_step_2/actor/optim_world_size_4_rank_1.pt").unlink()
            with self.assertRaisesRegex(ValueError, "checkpoint is incomplete"):
                pipeline.stage2_result(out, updates=2)

    def test_final_eval_can_retry_without_retraining_or_erasing_training_receipt(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            fake_stage2(out / "stage2", pipeline.STAGE2_UPDATES)
            with patch.object(pipeline, "run_stage", side_effect=SystemExit(1)):
                with self.assertRaises(SystemExit):
                    pipeline.finish(out, Path("source"), "0")
            self.assertTrue((out / "training_complete.json").is_file())
            self.assertFalse((out / "complete.json").exists())
            summary = {"episodes": 60, "successes": 1, "success_rate": 1/60,
                       "tasks_sha256": "c"*64, "contract_sha256": "d"*64}
            with patch.object(pipeline, "run_stage") as run_stage, patch.object(
                    final_eval, "verify_result", return_value=summary):
                pipeline.finish(out, Path("source"), "0")
            self.assertEqual(run_stage.call_count, 1)
            self.assertEqual(run_stage.call_args.args[1].name, "final_eval.log")
            self.assertEqual(pipeline.load_json(out / "complete.json")["final_eval"]["successes"], 1)

    def test_stage2_launcher_rejects_corrupted_sft_weights_before_worker_launch(self):
        # A fake Transformers module is enough: checksum validation must fail
        # before the tokenizer or any CUDA worker is opened.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "global_step_736"
            fake_stage1(checkpoint, "a" * 64, 736)
            (checkpoint / "model.safetensors").write_bytes(b"mock WEIGHTS")
            with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=None)}):
                with self.assertRaisesRegex(ValueError, "model SHA differs"):
                    assets.verify_stage1(checkpoint, expected_dataset="a" * 64, expected_step=736)

    def test_native_recipe_still_yields_150_updates(self):
        source = Path(os.environ.get("SEENAV_TEST_SOURCE", pipeline.base.DEFAULT_SOURCE))
        native = (source / pipeline.base.LOCK["train_launcher"]).read_text(encoding="utf-8")
        runner = rethink.adapt_runner(pipeline.base.enable_opsd(
            pipeline.base.isolated_paths(pipeline.base.four_gpu_launcher(native))), "rethink")
        self.assertIn("trainer.total_epochs=150", runner)
        self.assertIn("train_data_size=8", runner)
        self.assertIn('data.train_batch_size=$train_data_size', runner)
        self.assertIn("algorithm.adv_estimator=srgpo", runner)
        self.assertIn("++actor_rollout_ref.actor.sdar_gate_beta=5.0", runner)
        self.assertNotIn("trainer.total_training_steps=", runner)


if __name__ == "__main__":
    unittest.main()
