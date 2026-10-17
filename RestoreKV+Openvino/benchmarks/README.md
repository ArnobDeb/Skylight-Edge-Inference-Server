# Benchmarks

| Script | Output |
|---|---|
| `run_ruler4k_layouts.sh` | `results/ov_prototype/*_ruler4k_ladder_n10` |
| `run_ruler4k_main.sh` | `results/eviction_study/*_ruler130_*` |
| `run_throughput.sh` | `results/eviction_study/*_throughput_memory` |
| `run_genai_ruler4k.sh` | `results/genai_integration/*_genai_ruler4k_n10*` (inside OpenVINO GenAI) |
| `run_genai_throughput.sh` | `results/genai_integration/*_genai_throughput_*` |
| `run_long_context.sh` | `results/eviction_study/*_loft32k_*`, `*_ruler32k_*` |
| `long_context/generate_ruler.sh` | RULER-32K data in `artifacts/data/ruler/` |
| `long_context/build_loft.py` | LOFT-32K prompts in `artifacts/data/loft_qwen/` (download the LOFT datasets first, see the script) |
| `long_context/make_int8_model.py` | INT8-weight copy of Qwen3-8B in `artifacts/models/` |
| `seed_run.py` | copies finished rows of one run into a new run folder (the evaluator then only computes the missing variants) |
| `best_ckpt.py` | picks the adapter checkpoint with the lowest validation loss |

Protocol: budgets are the share of each prompt's KV cache kept (OpenVINO's `max_cache_size` = budget × prompt
length, rounded down to whole 16-token blocks); RestoreKV's 8 slots come out of the same budget. Greedy decoding;
RULER is scored by string match, LOFT with LOFT's own metrics.
