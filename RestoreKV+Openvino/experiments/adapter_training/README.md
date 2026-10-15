# adapter_training: RestoreKV adapters for OpenVINO's eviction

The adapter released with the RestoreKV paper is trained for KVzip eviction and does not help OpenVINO's policies.
These scripts train adapters with the official RestoreKV training code (github.com/cvsp-lab/RestoreKV, commit
2dbcfa9) after patching it to evict exactly like OpenVINO.

| File | Purpose |
|---|---|
| `apply_ov_policy_patch.py` | adds `--level ov-block`: OpenVINO H2O scoring (last 8 prompt tokens, summed over heads, max-pool 7, NORM_SUM) and 16-token blocks shared by all heads; the restore slots come out of the block budget |
| `apply_ov_arkv_patch.py` | adds `--level ov-arkv`: OpenVINO ADAPTIVE_RKV (attention mass 0.9, then most diverse blocks) |
| `apply_ov_headblock_patch.py` | adds `--level ov-headblock`: the same rule per KV head (with `--ov-block 1`, per-head tokens) |
| `test_ov_policy_torch.py`, `test_arkv_torch.py` | the PyTorch port must keep exactly the same blocks as the evaluator; writes per-case CSVs with the kept block IDs |
| `prajna/` | SLURM scripts used on the training cluster: environment, flash-attn build, training job, result collection |

Training recipe (the paper's, unchanged): 8 restore tokens, LoRA rank 8 / alpha 16 on all projections, symmetric-KL
self-distillation from the full-cache model, budget drawn from 2.5–25% at every step, 5,000 steps (~3 h on one
A40). The checkpoint with the lowest validation distillation loss is used (`benchmarks/best_ckpt.py`); convert it
for OpenVINO with `python src/restorekv_ov/checkpoint.py checkpoint_stepN.pt`.

```bash
git clone https://github.com/cvsp-lab/RestoreKV && cd RestoreKV && git checkout 2dbcfa9 && git checkout -b ov-policy
python <this folder>/apply_ov_policy_patch.py . && python <this folder>/apply_ov_arkv_patch.py .
sbatch --export=ALL,LEVEL=ov-block,STEPS=5000,NAME=ovblock_qwen3-8b prajna/train.sbatch
```

Adapters used in the results: `artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz` (H2O; picked by a
RULER sweep over all ten checkpoints, `results/adapter_training/*_ovblock_step*`) and
`artifacts/adapters/arkv_qwen3-8b/checkpoint_step2000.npz` (ADAPTIVE_RKV; lowest validation loss, 26.5).
