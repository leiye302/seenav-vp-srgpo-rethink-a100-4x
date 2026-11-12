"""Small driver-side bridge; neither observations nor rewards are edited."""

import numpy as np

from .protocol import for_execution


def execution_texts(text_actions):
    # The tokenizer may already have dropped the special marker. In either
    # case the original SeeNav tag projection still executes the one action.
    return [for_execution(text) for text in text_actions]


def evidence(obs, infos, dones, step):
    result = np.empty(len(infos), dtype=object)
    for i, info in enumerate(infos):
        result[i] = {
            "step": step + 1,  # unique per rollout, even on invalid attempts
            "prompt": obs["text"][i],
            "executed_action": int(info.get("action_id", -1)),
            "feedback": str(info.get("env_feedback", "")),
            "done": bool(dones[i]),
            "success": bool(info.get("won", False)),
        }
    return result
