"""Thin wrapper over a stateful LLM infer request with explicit position bookkeeping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
import openvino_tokenizers  # noqa: F401  registers tokenizer ops


def load_tokenizer(core: ov.Core, model_dir: Path):
    tok = core.compile_model(model_dir / "openvino_tokenizer.xml", "CPU")

    def encode(text: str) -> np.ndarray:
        return tok([text])["input_ids"].astype(np.int64)

    return encode


class StatefulLM:
    """Drives a stateful IR with either `input_ids` or `inputs_embeds` as the token input."""

    def __init__(self, compiled: ov.CompiledModel):
        self.compiled = compiled
        self.request = compiled.create_infer_request()
        self.input_names = {i.get_any_name() for i in compiled.inputs}
        self.kv_len = 0  # number of entries currently in the KV state
        self.next_pos = 0  # RoPE position of the next token fed

    def reset(self):
        self.request.reset_state()
        self.kv_len = 0
        self.next_pos = 0

    def forward(self, *, input_ids=None, inputs_embeds=None, position_ids=None) -> np.ndarray:
        x = input_ids if input_ids is not None else inputs_embeds
        b, t = x.shape[0], x.shape[1]
        if position_ids is None:
            position_ids = np.arange(self.next_pos, self.next_pos + t, dtype=np.int64)[None].repeat(b, 0)
        feed = {
            "attention_mask": np.ones((b, self.kv_len + t), dtype=np.int64),
            "position_ids": position_ids,
            "beam_idx": np.arange(b, dtype=np.int32),
        }
        if input_ids is not None:
            feed["input_ids"] = input_ids
        else:
            feed["inputs_embeds"] = inputs_embeds
        self.request.infer(feed)
        self.kv_len += t
        self.next_pos = int(position_ids.max()) + 1
        return self.request.get_tensor("logits").data.copy()
