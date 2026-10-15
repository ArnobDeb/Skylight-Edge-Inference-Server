# ov_prototype: RestoreKV on the OpenVINO runtime

The building blocks and the RULER-4K evaluator.

| Script | What it checks / measures |
|---|---|
| `check_inputs_embeds.py` | the edited IR with an `inputs_embeds` input gives the same logits as the original `input_ids` IR (bit-exact on CPU and GPU) |
| `kv_state_checks.py` | LoRA switched off = original model; reading and writing the KV state is exact; evicting by masking = evicting by removing entries; the restore pass appends 8 entries |
| `kv_state_roundtrip.py` | the same on real text with a few decoding steps |
| `ruler_eval.py` | RULER-4K: full cache vs each eviction method, each without and with RestoreKV. Methods: `kvzip`, `kvzip_layer`, `kvzip_hs`, `kvzip_block` (KVzip scores in progressively coarser layouts), `ov` (OpenVINO H2O), `arkv` (OpenVINO ADAPTIVE_RKV), `ov_ph` / `ov_pht` (OpenVINO scores, per-head blocks / tokens) |
| `runtime_crosscheck.py` | our eviction vs the real OpenVINO GenAI `ContinuousBatchingPipeline` on the same prompts |

`ruler_eval.py` options worth knowing: `--budgets` (share of the context kept), `--ov-block` (16 on GPU, 32 on CPU),
`--ov-recent` (OpenVINO `recent_size`, 32 for 4K prompts), `--ckpt` (RestoreKV adapter), `--n-per-task`,
`--max-batch` (variants decoded together; lower it if memory is short), `--out` / `--stage` (output folder).
Every sample appends to `results.jsonl`, so an interrupted run resumes; `summary.csv` holds the per-task averages.

Runtime cross-check: build OpenVINO GenAI 2026.4.1 from source with an environment-gated log in
`src/cpp/src/continuous_batching/cache/cache_eviction.cpp` that prints, per layer, the block scores and the evicted
block IDs to the file named by `OV_GENAI_EVICTION_LOG`, then

```bash
PYTHONPATH=<genai build>/build python experiments/ov_prototype/runtime_crosscheck.py --phase ov
env -u PYTHONPATH python experiments/ov_prototype/runtime_crosscheck.py --phase ours
```

Result (13 prompts × 3 budgets × 36 layers): 99.7–99.9% of evicted blocks identical to an exact replica of
OpenVINO's first eviction (which happens at the first decode step); 97.9–99.8% for the evaluator's one-shot
eviction right after the context.
