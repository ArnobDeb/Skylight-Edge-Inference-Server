# RestoreKV on OpenVINO

KV-cache eviction lets a large language model read long documents on a laptop: after reading the prompt, most of
the cached keys and values are dropped and only a small budget (for example 5% of the document) is kept. What is
dropped is lost. **RestoreKV** (arXiv 2608.01247) adds one cheap step before eviction: 8 learned "summary" tokens
read the full cache once, with a small LoRA adapter switched on, and write 8 extra cache entries that summarise
what is about to be thrown away. Those 8 entries are kept, inside the same memory budget.

This folder brings RestoreKV to **OpenVINO** and measures it against OpenVINO's own eviction policies on an edge
device: Qwen3-8B on an Intel Core Ultra 7 258V laptop (Arc integrated GPU, 30 GB of RAM shared by CPU and GPU).

```
document ──► prefill (full KV cache) ──► restore pass: 8 learned tokens + LoRA read the full cache (~0.3 s)
                                              │ writes 8 summary entries per layer
                                              ▼
                                  eviction to the budget (minus the 8 slots)
                                              │
question ──► answer from the kept entries + the 8 summary entries (LoRA off)
```

## Main results

RULER-4K, 130 samples (10 per task × 13 tasks), Qwen3-8B fp16 on the Arc iGPU. Budget = share of the document's
KV cache kept; both columns of a pair use the same memory. Full cache: **93.1**.

| Budget kept | OpenVINO H2O (default) | + RestoreKV | OpenVINO ADAPTIVE_RKV | + RestoreKV | KVzip (paper's eviction) | + RestoreKV |
|---|---|---|---|---|---|---|
| 2.5% | 4.9 | 6.8 | 4.9 | 4.9 | 17.7 | **33.7** |
| 5% | 15.1 | 16.6 | 22.9 | 22.4 | 34.4 | **68.4** |
| 10% | 32.7 | 34.8 | 35.3 | 35.5 | 88.1 | 92.0 |
| 15% | 43.8 | **49.8** | 49.0 | 49.2 | 91.9 | 94.5 |
| 25% | 61.2 | 64.8 | 66.9 | **68.6** | 94.2 | 94.3 |

- RestoreKV improves OpenVINO's default eviction (H2O) at every budget, by +1.5 to +6.0 points; the gain is above
  sampling noise at 2.5%, 10% and 15% (paired bootstrap, 95%).
- On ADAPTIVE_RKV it adds little except at 25% (+1.7): ADAPTIVE_RKV alone is already much stronger at small budgets.
- With KVzip, the eviction used in the RestoreKV paper, the method reproduces the paper's effect on the OpenVINO
  runtime (34.4 → 68.4 at 5%; paper, Qwen3-4B: 38.2 → 73.2).
- The difference between KVzip and OpenVINO is mostly the eviction granularity: OpenVINO keeps or drops whole
  16-token blocks for all attention heads of a layer at once; KVzip decides per head and per token.

Memory saved (KV cache per sequence, Qwen3-8B f16 = 144 KiB per token; identical with and without RestoreKV):

| Budget | 4K document | 32K document |
|---|---|---|
| full | 569 MB | 4.83 GB |
| 25% | 141 MB | 1.21 GB |
| 10% | 56 MB | 0.48 GB |
| 5% | 27 MB | 0.24 GB |

Decode speed with the cache physically cut to the budget: at 32K, one sequence decodes at 3.04 tokens/s with the
full cache and 5.6–6.6 tokens/s after eviction; eight evicted sequences together reach 38–40 tokens/s (the full
32K cache fits only one). RestoreKV decodes exactly as fast as plain eviction; its only cost is the restore pass:
0.34 s per 4K document, 1.94 s per 32K document (about 1.5% of reading a 4K document and 1% of a 32K one).

Details and all tables: [docs/RESULTS.md](docs/RESULTS.md). How the code drives OpenVINO: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How it is implemented

- **On the OpenVINO runtime.** The Qwen3-8B OpenVINO IR (stateful, KV cache kept inside the compiled model) is
  edited at load time (`src/restorekv_ov/ir.py`): an `inputs_embeds` input for the learned tokens, the adapter's
  LoRA branches behind a `lora_scale` switch, and an additive eviction mask on every attention layer. One compiled
  model on the iGPU runs the prefill, the restore pass and decoding (`src/restorekv_ov/engine.py`).
- **OpenVINO's eviction, reproduced exactly.** OpenVINO GenAI's eviction (`cache_eviction.cpp`: scores from the
  attention of the last 8 prompt tokens summed over heads, max-pool, NORM_SUM, 16-token blocks shared by all heads,
  protected start/recent areas; and ADAPTIVE_RKV) is reimplemented in `src/restorekv_ov/scoring.py` and
  `eviction.py`. Checked against the real runtime: on 13 prompts × 3 budgets × 36 layers, 99.7–99.9% of evicted
  blocks are identical (`experiments/ov_prototype/runtime_crosscheck.py`).
- **Adapters trained for OpenVINO's eviction.** The adapter released with the paper was trained for KVzip and does
  not help OpenVINO's policies, so new adapters were trained with the official RestoreKV training code, patched to
  evict exactly like OpenVINO (`experiments/adapter_training/`, tested block-for-block against the evaluator).

## Setup

Requirements: Linux, an Intel GPU with OpenVINO support (results above: Arc iGPU of a Core Ultra 7 258V), ~30 GB of
RAM, Python 3.11.

```bash
conda create -n restorekv python=3.11 && conda activate restorekv
pip install openvino==2026.4.1 openvino-genai==2026.4.1 openvino-tokenizers==2026.4.1 \
            "transformers==5.0.0" "datasets==4.8.5" "huggingface_hub==1.28.0" nncf numpy pandas plotly streamlit torch
# model: the OpenVINO IR of Qwen3-8B, next to this folder (../models/qwen3-8b-fp16-ov)
python ../openvino_benchmarking/download_model.py
```

Run every command from this folder with `PYTHONPATH` unset (`env -u PYTHONPATH python ...`) if your shell sets it.

Adapters: the two trained adapters used in the main results are included (`artifacts/adapters/`). The paper's
released adapter (for KVzip) is downloaded and converted with:

```bash
huggingface-cli download higokri/RestoreKV qwen3-8b_restorekv.pt --local-dir artifacts/hf/restorekv
python src/restorekv_ov/checkpoint.py artifacts/hf/restorekv/qwen3-8b_restorekv.pt
```

## Reproducing the results

Launch scripts are in `benchmarks/` (each writes a run folder under `results/` and a log; runs resume where they
stopped). Approximate times on the laptop above.

| What | Command | Time | Expected |
|---|---|---|---|
| Sanity checks of the IR edits | `python experiments/ov_prototype/check_inputs_embeds.py --model ../models/qwen3-8b-fp16-ov --device GPU`, then `kv_state_checks.py` | minutes | identical logits with / without the edits |
| Eviction port vs PyTorch port (CPU) | `python experiments/adapter_training/test_ov_policy_torch.py` and `test_arkv_torch.py` | ~5 min | `PASS` (64/64, 64/64, 72/72 identical) |
| Port vs the real OpenVINO runtime | see `experiments/ov_prototype/README.md` (needs OpenVINO GenAI built with the eviction log) | ~30 min | 99.7–99.9% identical evicted blocks |
| Main comparison, RULER-4K 130 samples | `bash benchmarks/run_ruler4k_main.sh` | ~20 h | the table above |
| Layout comparison (KVzip → OpenVINO layout) | `bash benchmarks/run_ruler4k_layouts.sh` | ~24 h | KVzip 34.4 → 68.4 at 5%, heads sharing tokens 25.1 at 10% |
| Memory and decode speed | `bash benchmarks/run_throughput.sh` | ~40 min | `results/eviction_study/*_throughput_memory/` |
| Long context, LOFT-32K and RULER-32K | `bash benchmarks/run_long_context.sh` (data: `benchmarks/long_context/`) | ~1 day | `results/eviction_study/*_32k_*` |
| Train an adapter | `experiments/adapter_training/README.md` (one A40, ~3 h) | | validation distillation loss ~26–32 |

Watch the results while they run:

```bash
streamlit run dashboard/app.py
```

## Layout

```
src/restorekv_ov/        library: IR edits, inference engine, eviction scoring and selection, adapter loading
experiments/
  ov_prototype/          IR checks, RULER evaluator, runtime cross-check against OpenVINO GenAI
  adapter_training/      patches for the RestoreKV training code, port tests, cluster job scripts
  eviction_study/        long-context evaluator, memory and speed benchmark
benchmarks/              launch scripts, long-context data generation (RULER-32K, LOFT-32K), INT8 model export
profiling/               speed / memory measurements of the OpenVINO pipeline on the iGPU
results/                 one folder per run (config.json, results.jsonl, summary.csv)
dashboard/               Streamlit dashboard
docs/                    ARCHITECTURE.md, RESULTS.md
artifacts/adapters/      trained adapters used in the results
```
