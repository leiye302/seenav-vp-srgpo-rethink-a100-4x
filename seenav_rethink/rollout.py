"""Small driver-side bridge; neither observations nor rewards are edited."""

import numpy as np

from .protocol import decode_decision, for_execution, marker_token_id


def execution_texts(tokenizer, responses, attention_mask):
    """Give the native parser only well-formed decisions; invalid means no-op."""
    marker = marker_token_id(tokenizer)
    active = attention_mask[:, -responses.shape[-1]:].bool()
    result = []
    for ids, mask in zip(responses, active, strict=True):
        decoded = decode_decision(tokenizer, ids[mask].tolist(), marker)
        result.append(for_execution(decoded.text) if decoded.parsed is not None else "")
    return result


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
