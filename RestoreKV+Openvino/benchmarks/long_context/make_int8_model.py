"""INT8 weight-only copy of Qwen3-8B for the 32K/64K runs (8B fp16 weights + 64K KV cache do not fit in 30 GB).

NNCF INT8_ASYM per-channel weight compression of every MatMul (incl. lm_head); the token-embedding Gather is left in
fp16 because the RestoreKV engine reads the embedding table directly (inputs_embeds surgery, src/restorekv_ov/ir.py).
    env -u PYTHONPATH python benchmarks/long_context/make_int8_model.py
"""
import shutil
import sys
from pathlib import Path

import nncf
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov.safety import protect  # noqa: E402

SRC = ROOT.parent / "models/qwen3-8b-fp16-ov"
DST = ROOT / "artifacts/models/qwen3-8b-int8-ov"

protect(floor_gb=1.5)
DST.mkdir(parents=True, exist_ok=True)
model = ov.Core().read_model(SRC / "openvino_model.xml")
embed = [n.get_friendly_name() for n in model.get_ordered_ops()
         if n.get_type_name() == "Gather" and "embed_tokens" in n.get_friendly_name()]
print("keeping fp16:", embed)
model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM,
                              ignored_scope=nncf.IgnoredScope(names=embed, validate=False))
ov.save_model(model, DST / "openvino_model.xml", compress_to_fp16=False)
for f in SRC.iterdir():
    if f.is_file() and f.name not in ("openvino_model.xml", "openvino_model.bin"):
        shutil.copy(f, DST / f.name)
print("saved", DST, sum(f.stat().st_size for f in DST.iterdir()) / 1e9, "GB")
