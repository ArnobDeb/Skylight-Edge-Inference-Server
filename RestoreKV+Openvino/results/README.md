# Results

One folder per run, `<date>_<name>/`, with `config.json` (all options), `results.jsonl` (one line per sample and
variant, including the generated answer) and `summary.csv` (per-task averages).

| Folder | Runs |
|---|---|
| `ov_prototype/` | IR checks; RULER-4K layout comparison (`*_ruler4k_ladder_n10`, 130 samples); runtime cross-check against OpenVINO GenAI |
| `adapter_training/` | every checkpoint of the H2O adapter, the ADAPTIVE_RKV and window-32 adapters and the released adapter, RULER-4K 39 samples; port tests with kept block IDs (`*_policy_port_tests`) |
| `eviction_study/` | main comparison (`*_ruler130_*`, 130 samples); 39-sample studies of block size, recent area, restore tokens and per-head layouts (`*_prestudy_*`, `*_ovblock_n16_*`, `*_rec128_*`, `*_ovheadblock_*`); long context (`*_32k_*`); speed and memory (`*_throughput_memory`) |
| `profiling/` | OpenVINO GenAI pipeline on the iGPU: long-context scaling, eviction memory, CSV summaries in `csv/` |

Scores are task-averaged (each of the 13 RULER tasks or LOFT datasets counts equally).
