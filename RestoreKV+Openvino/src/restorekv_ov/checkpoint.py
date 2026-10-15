"""RestoreKV checkpoints (higokri/RestoreKV *.pt) -> plain .npz, so the runtime needs no torch/pickle.

    python -m restorekv_ov.checkpoint artifacts/hf/restorekv/qwen3-8b_restorekv.pt

Layout of the .pt (verified 2026-10-08 for qwen3-8b): restore.state_dict.embeddings [8, hidden] bf16;
trainable_model_state["model.layers.<i>.<self_attn|mlp>.<proj>.lora_a"] [r, in] and ".lora_b" [out, r] bf16
for q/k/v/o/gate/up/down; restore.restore_config has lora_rank / lora_alpha / num_restore_tokens.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJ_TO_MODULE = {p: "self_attn" for p in ("q_proj", "k_proj", "v_proj", "o_proj")} | {
    p: "mlp" for p in ("gate_proj", "up_proj", "down_proj")}


@dataclass
class RestoreCheckpoint:
    embeddings: np.ndarray            # [n, hidden] float16
    lora: dict[str, tuple[np.ndarray, np.ndarray]]  # "layers.<i>.<proj>" -> (A [r, in], B [out, r]) float16
    rank: int
    alpha: float
    meta: dict

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def load(cls, path: Path) -> "RestoreCheckpoint":
        path = Path(path)
        if path.suffix == ".pt":
            path = convert(path)
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        lora = {}
        for key in z.files:
            if key.endswith(".A"):
                base = key[:-2]
                lora[base] = (z[key], z[base + ".B"])
        return cls(z["embeddings"], lora, meta["lora_rank"], meta["lora_alpha"], meta)


def convert(pt_path: Path) -> Path:
    import torch  # only needed for conversion

    pt_path = Path(pt_path)
    out = pt_path.with_suffix(".npz")
    if out.exists():
        return out
    ck = torch.load(pt_path, map_location="cpu", weights_only=True)
    cfg = ck["restore"]["restore_config"]
    extra = ck["restore"].get("extra", {})
    arrays = {"embeddings": ck["restore"]["state_dict"]["embeddings"].float().numpy().astype(np.float16)}
    for key, t in ck["trainable_model_state"].items():
        # model.layers.<i>.<module>.<proj>.lora_a|lora_b
        parts = key.split(".")
        i, proj, kind = parts[2], parts[4], parts[5]
        arrays[f"layers.{i}.{proj}.{'A' if kind == 'lora_a' else 'B'}"] = t.float().numpy().astype(np.float16)
    meta = {"lora_rank": int(cfg["lora_rank"]), "lora_alpha": float(cfg["lora_alpha"]),
            "num_restore_tokens": int(cfg["num_restore_tokens"]), "mode": cfg.get("mode"),
            "model": ck.get("model"), "step": ck.get("step"), "score_method": extra.get("score_method"),
            "level": extra.get("level"), "ratio_min": extra.get("ratio_min"), "ratio_max": extra.get("ratio_max"),
            "context_len_min": extra.get("context_len_min"), "context_len_max": extra.get("context_len_max"),
            "source": pt_path.name}
    np.savez(out, meta=np.array(json.dumps(meta)), **arrays)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(convert(Path(p)))
