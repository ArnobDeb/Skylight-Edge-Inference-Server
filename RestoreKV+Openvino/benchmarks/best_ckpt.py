"""Print the converted checkpoint (.npz) with the lowest mean validation distillation loss of an adapter run
(loss_history.csv from the training code), and write it to <adapter_dir>/BEST.
    python3 benchmarks/best_ckpt.py artifacts/adapters/ovblock_n16_qwen3-8b
"""
import csv
import sys
from pathlib import Path

d = Path(sys.argv[1])
rows = [r for r in csv.DictReader(open(d / "loss_history.csv")) if r.get("val_distill_loss")]
best = {}
for r in rows:
    step = int(r["step"])
    if (d / f"checkpoint_step{step}.npz").exists():
        best[step] = float(r["val_distill_loss"])
step = min(best, key=best.get)
path = d / f"checkpoint_step{step}.npz"
(d / "BEST").write_text(f"{path}\n")
print(path, f"val_distill_loss {best[step]:.2f}", " ".join(f"{s}:{v:.1f}" for s, v in sorted(best.items())))
