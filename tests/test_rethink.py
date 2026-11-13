import ast
import base64
from collections import Counter
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
from PIL import Image
import torch

import reproduce
import rethink
from seenav_opsd.teacher import attach_teacher_scores, pack_teacher_tensors
from seenav_rethink import hints, protocol
from seenav_rethink.data import digest_bytes, render_native_prompt, route_loss


class Tokenizer:
    marker = 90
    pad_token_id = 0
    all_special_ids = [0, marker, 99]

    def encode(self, text, **kw):
        if text == protocol.MARKER:
            return [self.marker]
        if "Local Navigation Hint" in text:
            return [61, 62, 63]
        if "[Privileged Skill Information]" in text:
            assert "native <action> tag" in text and "JSON format" not in text
            return [31, 32]
        raise AssertionError(text)

    def decode(self, ids, **kw):
        assert kw["skip_special_tokens"] is False
        return "".join({
            12: "<think>I should approach the marked chair.</think>",
            13: "<action>2</action>", 90: protocol.MARKER, 99: protocol.EOS,
        }[i] for i in ids)


def fixture():
    responses = torch.tensor([[12, 90, 13, 99], [12, 90, 13, 99], [12, 13, 99, 90],
                              [12, 90, 13, 99], [90, 12, 13, 99], [12, 90, 90, 99]])
    prompt = torch.tensor([[0, 21, 22]] * len(responses))
    ids = torch.cat((prompt, responses), dim=-1)
    mask = torch.ones_like(ids)
    mask[:, 0] = 0
    mask[2, -1] = 0
    pos = torch.arange(ids.shape[-1]).expand(len(ids), 3, -1).clone()
    p = {"input_ids": ids, "responses": responses, "attention_mask": mask, "position_ids": pos,
         "old_log_probs": torch.zeros_like(responses, dtype=torch.float32)}
    stage = lambda i: {"step": i, "prompt": f"H{i}", "executed_action": 2,
                        "feedback": "Native execution", "done": False, "success": False}
    a = SimpleNamespace(batch=p, non_tensor_batch={
        "traj_uid": ["epA", "epA", "epA", "epB", "epC", "epD"],
        "rethink_evidence": [stage(i) for i in (1, 1, 2, 1, 1, 1)],
        "anchor_obs": [object() for _ in responses],
        "multi_modal_inputs": [{"image_grid_thw": torch.tensor([[1, 4, 4]])} for _ in responses],
    })
    return a


class RethinkTest(unittest.TestCase):
    def test_exact_route_boundary_and_only_shy_cleanup(self):
        suffix = "</think>\u00ad\n<RETHINK>\u00ad<action>2</action>"
        good = "<think>Turn left</think><action>3</action>"
        bad = "<think>Turn\u00ad right" + suffix
        self.assertEqual(protocol.parse_decision(good), ("Turn left", False, 3))
        self.assertEqual(protocol.parse_decision(bad + protocol.EOS), ("Turn\u00ad right", True, 2))
        self.assertEqual(protocol.for_execution(bad), "<think>Turn\u00ad right</think>\n<RETHINK><action>2</action>")
        for bad in ("<action>1</action><think>Too late</think>",
                    "<think></think><RETHINK><action>0</action>",
                    "<think>There is <RETHINK> in the reason</think><action>0</action>",
                    "<think>One</think><action>2</action><RETHINK>",
                    "<think>One</think><RETHINK><RETHINK><action>2</action>",
                    "<think>One</think><action>8</action>",
                    "<think>One</think><action>0</action><action>1</action>"):
            self.assertIsNone(protocol.parse_decision(bad))

    def test_marker_loss_has_exact_causal_index_and_full_vocab_negative(self):
        for label, target in [(0, 2), (1, 5)]:
            ids = torch.tensor([1, 3, 1, target, 4])
            logits = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)
            out = route_loss(logits, ids, 3, label, 5)
            prob = logits[2].softmax(0)[5]
            torch.testing.assert_close(out["marker_bce"], -torch.log(prob if label else 1-prob))
            torch.testing.assert_close(out["suffix_ce"], torch.nn.functional.cross_entropy(logits[2:4], ids[3:]))
            out["loss"].backward()
            self.assertTrue((logits.grad[[0, 1, 4]] == 0).all())
            self.assertGreater(float(logits.grad[2].abs().sum()), 0)

    def test_legacy_prompt_digest_allows_only_native_newline(self):
        original = "<|im_start|>user\n\n<image>\nGoal<|im_end|>"
        expected = original.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        legacy = expected.replace("user\n\n", "user\n", 1)
        tok = SimpleNamespace(apply_chat_template=lambda *a, **kw: original)
        row = {"prompt_text": original, "rendered_prompt_sha256": digest_bytes(legacy.encode())}
        self.assertEqual(render_native_prompt(tok, row), expected)
        row["rendered_prompt_sha256"] = digest_bytes(legacy.replace("Goal", "Other").encode())
        with self.assertRaises(ValueError):
            render_native_prompt(tok, row)

    def test_real_teacher_hook_same_actor_token_mask_images_and_per_row_fallback(self):
        batch = fixture()
        original = {k: v.clone() for k, v in batch.batch.items()}
        response = {"choices": [{"message": {"content": '{"hints":{"1":"Next move: 3. The left path is clear."}}'}}]}
        sent = []

        def post(_, body):
            sent.append(body)
            if len(sent) == 1:
                return response
            raise URLError("simulated network failure")

        # Only the teacher sees the extra tokens. The actual native hook creates
        # exactly one data packet and no gradient-bearing teacher scores.
        def score(teacher):
            self.assertEqual(set(teacher.non_tensor_batch), {"multi_modal_inputs"})
            self.assertIs(teacher.non_tensor_batch["multi_modal_inputs"], batch.non_tensor_batch["multi_modal_inputs"])
            t = teacher.batch
            self.assertTrue(torch.equal(t["responses"], original["responses"]))
            self.assertEqual(t["input_ids"].shape[-1], original["input_ids"].shape[-1]+5)
            self.assertEqual((t["input_ids"] == 61).sum(dim=1).tolist(), [1, 1, 0, 0, 0, 0])
            baseline = pack_teacher_tensors(original, [31, 32], 64)
            torch.testing.assert_close(t["position_ids"][2:, :, -baseline["input_ids"].shape[-1]:],
                                       baseline["position_ids"][2:])
            self.assertTrue(torch.equal(t["attention_mask"][:, -4:], original["attention_mask"][:, -4:]))
            self.assertEqual(t["position_ids"][0, :, -1].tolist(), (original["position_ids"][0, :, -1]+5).tolist())
            return SimpleNamespace(batch={"old_log_probs": torch.ones_like(original["old_log_probs"], requires_grad=True)})

        trainer = SimpleNamespace(tokenizer=Tokenizer(),
            config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(output_format="tags", rethink_local_hints=True),
                rollout=SimpleNamespace(multi_turn=SimpleNamespace(enable=False), max_model_len=64))),
            actor_rollout_wg=SimpleNamespace(compute_log_prob=score))
        stub = ModuleType("verl")
        stub.DataProto = SimpleNamespace(from_dict=lambda tensors, non_tensors:
                                        SimpleNamespace(batch=tensors, non_tensor_batch=non_tensors))
        config = hints.Settings("https://test.invalid/v1/chat/completions", "fake", "unlogged", parallel=1)
        with patch.dict(sys.modules, {"verl": stub}), patch.object(hints, "from_env", return_value=config), \
                patch.object(hints, "image_url", return_value="data:image/png;base64,AA=="), \
                patch.object(hints, "post", post), patch.object(hints.time, "sleep"):
            stats = attach_teacher_scores(trainer, batch)
        self.assertEqual(stats["rethink/marker_rows"], 5)
        self.assertEqual(stats["rethink/eligible_rows"], 3)
        self.assertEqual(stats["rethink/unique_routed"], 2)
        self.assertEqual(stats["rethink/unique_with_local_hint"], 1)
        self.assertEqual(stats["rethink/unique_static_fallback"], 1)
        self.assertEqual(stats["rethink/malformed_marker_rows"], 2)
        self.assertEqual(len(sent), 3)  # one epA success + two failed epB attempts
        self.assertEqual(stats["rethink/hint_requests"], len(sent))
        self.assertEqual(Counter(x["type"] for x in sent[0]["messages"][1]["content"])["image_url"], 1)
        self.assertNotIn("rethink_evidence", batch.non_tensor_batch)
        self.assertFalse(batch.batch["teacher_log_probs"].requires_grad)
        for k, v in original.items():
            torch.testing.assert_close(batch.batch[k], v)

    def test_no_api_if_unrouted_and_no_approval_decision(self):
        from seenav_rethink.teacher import prefixes

        b = fixture()
        b.batch["responses"].masked_fill_(b.batch["responses"] == Tokenizer.marker, 0)
        t = SimpleNamespace(tokenizer=Tokenizer())
        with patch.object(hints, "from_env", side_effect=AssertionError("unexpected API")):
            _, local, stats = prefixes(t, b, enabled=True)
            self.assertFalse(local)
            self.assertEqual(stats["rethink/marker_rows"], 0)
            _, local, stats = prefixes(t, b, enabled=False)
            self.assertEqual((local, stats), ({}, {}))
        good, rejected = hints.parse_reply({"choices": [{"message": {"content": json.dumps(
            {"hints": {"1": "Next move: 4. Recover the target.", "2": "<RETHINK>", "99": "Next move: 0. Go."}})
        }}]}, [1, 2])
        self.assertEqual(set(good), {1})
        self.assertEqual(set(rejected), {2})

    def test_go_session_stays_with_episode_and_is_not_sent_to_other_providers(self):
        from dataclasses import replace

        cfg = hints.Settings("https://opencode.ai/inference/go/openai/v1/chat/completions",
                             "demo", "placeholder", parallel=1, images_per_request=1)
        slots = []

        def request(settings, timeline, selected, deadline):
            slots.append(settings.session_id)
            return {}, {}, 0

        jobs = {"ep1": {1: (1,), 2: (2,)}, "ep2": {1: (1,)}}
        with patch.object(hints, "one_request", side_effect=request):
            hints.request_hints(cfg, {uid: {} for uid in jobs}, jobs)
        self.assertIsNotNone(slots[0])
        self.assertEqual(slots[0], slots[1])
        self.assertNotEqual(slots[0], slots[2])

        for endpoint, session in ((cfg.endpoint, slots[0]),
                                  ("https://other.invalid/inference/go/v1/chat/completions", None)):
            with patch.object(hints, "build_opener") as opener:
                opener.return_value.open.return_value.__enter__.return_value.read.return_value = b'{}'
                hints.post(replace(cfg, endpoint=endpoint, session_id=slots[0]), {"model": "demo"})
                req = opener.return_value.open.call_args.args[0]
            self.assertEqual(req.get_header("User-agent"), "seenav-rethink-hints/1.0")
            self.assertEqual(req.get_header("X-opencode-session"), session)

    def test_native_anchor_is_the_actual_processed_image(self):
        source = Path(os.environ.get("SEENAV_TEST_SOURCE", reproduce.DEFAULT_SOURCE))
        py = source / "verl-agent/agent_system/multi_turn_rollout/utils.py"
        tree = ast.parse(py.read_text(encoding="utf-8"))
        native_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_image")
        ns = {"torch": torch, "np": np, "Image": Image, "math": math}
        exec(compile(ast.Module(body=[native_fn], type_ignores=[]), str(py), "exec"), ns)
        utils = ModuleType("agent_system.multi_turn_rollout.utils")
        utils.process_image = ns["process_image"]
        anchor = np.broadcast_to(np.arange(250, dtype=np.float32).repeat(4)[None, :, None]/255, (500, 1000, 3))
        with patch.dict(sys.modules, {utils.__name__: utils}):
            url = hints.image_url(anchor.astype(object))
        raw = base64.b64decode(url.partition(",")[2])
        with Image.open(BytesIO(raw)) as im:
            np.testing.assert_array_equal(np.asarray(im), np.asarray(utils.process_image(anchor)))

    def test_locked_collector_and_launch_boundaries(self):
        runtime = rethink.verify_runtime(rethink.RUNTIME)
        content = (runtime / rethink.ROLLOUT).read_text(encoding="utf-8")
        self.assertLess(content.index("execution_texts(text_actions)"), content.index("envs.step(text_actions)"))
        self.assertLess(content.index("envs.step(text_actions)"), content.index("= evidence(obs, infos"))
        self.assertIn('not gen_batch.meta_info.get("validate", False)', content)
        source = Path(os.environ.get("SEENAV_TEST_SOURCE", reproduce.DEFAULT_SOURCE))
        native = (source / reproduce.LOCK["train_launcher"]).read_text(encoding="utf-8")
        baseline = reproduce.enable_opsd(reproduce.isolated_paths(reproduce.four_gpu_launcher(native)))
        tag = rethink.adapt_runner(baseline, "tag-control")
        self.assertEqual(tag, baseline.replace("output_format=json", "output_format=tags"))
        stage1 = rethink.adapt_runner(tag.replace("output_format=tags", "output_format=json"), "sft-static")
        self.assertEqual(stage1, tag.replace("actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct",
                                            'actor_rollout_ref.model.path="${SEENAV_STAGE1_CHECKPOINT}"'))
        main = rethink.adapt_runner(baseline, "rethink")
        self.assertEqual([s for s in main.splitlines() if s not in stage1.splitlines()],
                         ["    ++actor_rollout_ref.actor.rethink_local_hints=True " + chr(92)])


if __name__ == "__main__":
    unittest.main()
