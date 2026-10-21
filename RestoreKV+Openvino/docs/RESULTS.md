# Results

All numbers: Qwen3-8B on an Intel Core Ultra 7 258V laptop (Arc iGPU, 30 GB shared RAM), OpenVINO 2026.4.1, greedy
decoding. Budget = share of the prompt's KV cache that is kept. RestoreKV's 8 summary entries come out of the same
budget, so "without" and "with" always use the same memory.

## 1. Main comparison: RULER-4K, 130 samples

10 samples for each of the 13 RULER tasks (needle-in-a-haystack variants, variable tracking, word counting, QA);
16-token blocks as on the iGPU; full cache 93.1.

| Budget | OpenVINO H2O | + RestoreKV | ADAPTIVE_RKV | + RestoreKV | KVzip | + RestoreKV | Per-head tokens* | + RestoreKV |
|---|---|---|---|---|---|---|---|---|
| 2.5% | 4.9 | 6.8 | 4.9 | 4.9 | 17.7 | 33.7 | 6.1 | 8.6 |
| 5% | 15.1 | 16.6 | 22.9 | 22.4 | 34.4 | 68.4 | 13.8 | 21.2 |
| 10% | 32.7 | 34.8 | 35.3 | 35.5 | 88.1 | 92.0 | 24.6 | 33.7 |
| 15% | 43.8 | 49.8 | 49.0 | 49.2 | 91.9 | 94.5 | 44.5 | 47.3 |
| 25% | 61.2 | 64.8 | 66.9 | 68.6 | 94.2 | 94.3 | 52.9 | 65.0 |

\* OpenVINO's scores with each KV head keeping its own tokens; not supported by OpenVINO's cache layout, shown to
measure what the layout costs.

Paired per-sample gain of RestoreKV on OpenVINO H2O (95% bootstrap interval; samples better / worse):

| Budget | Gain | Interval | Better / worse |
|---|---|---|---|
| 2.5% | +2.0 | [+0.7, +3.7] | 10 / 1 |
| 5% | +1.5 | [−2.4, +5.3] | 14 / 9 |
| 10% | +2.1 | [+0.2, +4.1] | 12 / 5 |
| 15% | +6.0 | [+2.7, +9.6] | 19 / 3 |
| 25% | +3.6 | [−0.8, +8.4] | 19 / 7 |

Adapters: H2O and per-head tokens use the adapter trained for OpenVINO H2O; ADAPTIVE_RKV its own adapter; KVzip the
adapter released with the paper (trained for KVzip). KVzip at 5/10/25% comes from the layout run below (same 130
samples).

## 2. Why OpenVINO gains less than KVzip: eviction layout

Same 130 samples, the paper's released adapter, 32-token blocks. KVzip's per-head, per-token eviction is changed
one step at a time towards OpenVINO's layout:

| Layout | 5% | 10% | 25% |
|---|---|---|---|
| KVzip: every (layer, head, token) ranked globally | 34.4 → 68.4 | 88.1 → 92.0 | 94.2 → 94.3 |
| same number of entries per layer, heads still differ | 29.5 → 42.6 | 66.6 → 78.6 | 91.3 → 92.4 |
| one set of tokens shared by all heads of a layer | 19.3 → 20.8 | 25.1 → 26.3 | 38.1 → 40.9 |
| KVzip scores in OpenVINO's block layout | 18.3 → 19.1 | 27.4 → 26.2 | 35.7 → 37.3 |
| OpenVINO H2O | 17.5 → 14.9 | 32.3 → 32.7 | 61.8 → 63.4 |

(without → with RestoreKV)

- Forcing all heads of a layer to keep the same tokens is the largest loss (66.6 → 25.1 at 10%); blocks and
  OpenVINO's cheaper scoring cost little on top.
- The KVzip-trained adapter does not help OpenVINO's eviction (17.5 → 14.9 at 5%): an adapter only helps the
  eviction it was trained for.

## 3. Adapters trained for OpenVINO's eviction

Recipe of the paper, training code patched to evict exactly like OpenVINO (16-token blocks shared by all heads);
5,000 steps on one A40 each. RULER-4K, 39 samples (3 per task; one sample changes a score by ~2.6 points), full
cache 89.5:

| Setup | 5% | 10% | 25% |
|---|---|---|---|
| OpenVINO H2O | 11.2 | 26.6 | 64.3 |
| + released KVzip adapter | 15.2 | 27.6 | 66.7 |
| + adapter trained on H2O | 17.5 | 27.7 | 65.6 |
| ADAPTIVE_RKV | 16.7 | 26.1 | 66.8 |
| + adapter trained on ADAPTIVE_RKV | 17.9 | 28.4 | 69.0 |
| H2O, SnapKV window 32 | 13.1 | 26.8 | 50.9 |
| + adapter trained on window 32 | 17.5 | 28.1 | 51.3 |

All ten checkpoints of the H2O adapter were evaluated (`results/adapter_training/*_ovblock_step*`): every checkpoint
from step 1000 on improves on H2O at every budget; steps 4000–5000 are best at 5%.

Variants tested on the same 39 samples (without → with RestoreKV):

| Setup | 2.5% | 5% | 10% | 15% | 25% |
|---|---|---|---|---|---|
| 8 restore tokens (default) | 4.3 → 5.7 | 11.2 → 17.5 | 26.6 → 27.7 | 41.3 → 45.8 | 64.3 → 65.6 |
| 16 restore tokens | 4.3 → 7.4 | 11.2 → 16.5 | 26.6 → 27.9 | 41.3 → 45.8 | 64.3 → 65.0 |
| recent area 128 instead of 32 | 7.3 → 14.2 | 11.1 → 13.9 | 27.1 → 27.2 | 38.2 → 39.1 | 64.3 → 65.1 |
| per-head blocks, adapter trained for them | 4.3 → 3.4 | 8.4 → 9.0 | 22.8 → 31.6 | 31.2 → 41.3 | 57.9 → 56.5 |

16 restore tokens cost the same memory as 8 under 16-token blocks but do not help. A recent area of 128 only helps
at 2.5%, where the protected tokens exceed the budget (more memory is actually kept). Per-head blocks with
OpenVINO's scoring are worse than shared blocks: a score from 8 query rows is noisy when split per head.

## 4. Memory and decode speed

KV cache per sequence (Qwen3-8B, f16: 144 KiB per token):

| Budget | 4K document | 32K document |
|---|---|---|
| full | 569 MB | 4.83 GB |
| 25% | 141 MB | 1.21 GB |
| 15% | 84 MB | 0.72 GB |
| 10% | 56 MB | 0.48 GB |
| 5% | 27 MB | 0.24 GB |
| 2.5% | 13 MB | 0.12 GB |

Decode tokens/s with the cache physically shortened (plain eviction / + RestoreKV):

| 4K document (fp16) | batch 1 | batch 4 | batch 8 |
|---|---|---|---|
| full cache | 3.30 | 9.54 | does not fit |
| 2.5% | 3.70 / 3.71 | 13.42 / 12.83 | 26.1 / 25.6 |
| 5% | 3.69 / 3.64 | 13.03 / 12.74 | 23.8 / 24.2 |
| 10% | 3.60 / 3.57 | 12.89 / 12.63 | 23.1 / 23.7 |
| 25% | 3.46 / 3.44 | 11.72 / 11.03 | 21.0 / 20.8 |

| 32K document (INT8 weights) | batch 1 | batch 4 | batch 8 |
|---|---|---|---|
| full cache | 3.04 | does not fit | does not fit |
| 2.5% | 6.24 / 6.16 | 21.3 / 22.0 | 38.4 / 39.5 |
| 5% | 6.58 / 6.31 | 20.7 / 20.0 | 31.8 / 33.2 |
| 10% | 6.27 / 6.16 | 17.1 | 23.7 |

One-time costs per document:

| | 4K (fp16) | 32K (INT8) |
|---|---|---|
| Prefill, warm | 11.3 s unmodified IR, 22.9 s with the RestoreKV graph edits | 180 s unmodified IR, 201 s with the edits |
| OpenVINO H2O scoring | 6.3 s | 14.7 s |
| KVzip scoring | 85.9 s | – |
| **RestoreKV restore pass** | **0.34 s (~1.5% of the prefill)** | **1.94 s (~1% of the prefill)** |

Warm prefill times from `profiling/07_prefill_ablation.py` (compiled model already used once). The throughput run's
`one_time_costs_*.json` record 48.8 s / 817 s: the 4K value includes GPU kernel compilation on the first call, the
32K value was slowed by memory pressure in a long-running process. The restore pass is timed once per document
(`time.perf_counter()` around `restore_pass()`), runs before eviction and is the same for every budget. The paper
reports 0.03–0.04 s (<0.5%) for 32K on an RTX PRO 6000; the iGPU is a much smaller device.

- At 4K the weights (14 GB) dominate every decode step, so eviction alone gives 5–12% at batch 1; the gain comes
  from fitting more sequences (2.2–2.7× throughput at batch 8).
- At 32K the cache matters: eviction doubles single-sequence speed (3.04 → 5.6–6.6 tokens/s) and lets 8 sequences
  run (38–40 tokens/s).
- RestoreKV does not change decode speed (same cache size); differences are run-to-run noise.

## 5. Agreement with the OpenVINO runtime

OpenVINO GenAI 2026.4.1 rebuilt with a log of every eviction (per layer: block scores and evicted block IDs);
13 RULER prompts × 3 budgets × 36 layers on the iGPU:

| | Evicted blocks identical | Block-score rank correlation |
|---|---|---|
| exact replica of OpenVINO's first eviction | 99.7–99.9% | 0.998 |
| the evaluator (one-shot eviction right after the document) | 97.9–99.8% | 0.986 |

OpenVINO evicts at the first decode step, so its scores also include the first generated token's attention; the
evaluator evicts before the question is known, as in the RestoreKV paper. Two runtime issues observed with
OpenVINO GenAI 2026.4.1 on the Arc iGPU: eviction scores are NaN for a 5,127-token prompt prefilled in one chunk,
and ADAPTIVE_RKV stops with "Token diversity must be registered before each eviction". Enabling eviction on the iGPU
also disables the fused attention kernels during prefill (2.9–9.4× slower, `profiling/05_gpu_eviction_root_cause.md`).

## 6. Long context

RULER-32K (generated with NVIDIA RULER and the Qwen3 tokenizer) and LOFT-32K (13 text datasets, 10 queries each)
run with Qwen3-8B INT8 weights; results are written to `results/eviction_study/*_32k_*` and shown in the dashboard.
64K contexts do not fit in the laptop's 30 GB of shared memory.

## 7. Inside OpenVINO GenAI (C++ pipeline)

RestoreKV in `ContinuousBatchingPipeline` (patched GenAI 2026.4.1, `experiments/genai_integration/`): real paged
cache, real block eviction, OpenVINO's own scoring, the 8 restore tokens protected. Before the runs, the exported
model was checked against the prototype on a 5,127-token sample: with plain token ids it gives the same output as the
original model, and with the restore tokens in the prompt the same restore entries (relative difference <= 1.3%) and
the same answer as the prototype's restore pass.

### 7.1 Two protocols

| | Question-blind (RestoreKV paper, prototype) | Question-aware (OpenVINO GenAI default) |
|---|---|---|
| Compression happens | once, right after the document | after the whole prompt (document + question) |
| OpenVINO's scoring window (last 8 prompt tokens) | the end of the document | the end of the question / chat template |
| The question is read | on the compressed cache | before compression, on the full cache |
| Later evictions while generating | none (as in the paper) | whenever the cache exceeds the budget |
| Use case | compress a document once, reuse it for many questions | one question per document |
| Comparable with | the paper, the prototype (§1) | only itself |

Stock GenAI only does question-aware eviction. The question-blind runs use an opt-in mode added to the patched
pipeline (`RESTOREKV_QUERY_BLIND=1`): the prompt `[document][SEP][question]` is prefilled up to SEP, evicted once
with the window on the document's end, and the question is then fed token by token on the evicted cache. In this
"inside order" variant the 8 restore tokens sit just before the last 8 document tokens, so that both variants are
scored from the same window. A second variant that follows the prototype's order exactly (restore tokens fed after
the document prefill and excluded from the scores, `RESTOREKV_QUERY_BLIND=2`) is in the branch as well.

### 7.2 Accuracy, Qwen3-8B, RULER-4K 130 samples

OpenVINO H2O (NORM_SUM, SnapKV window 8, start/recent 32, 16-token blocks), H2O adapter. Full cache **93.1**
(identical to the prototype). Paired gain of RestoreKV with 95% bootstrap interval and samples better / worse:

| Budget | Prototype, question-blind | **GenAI, question-blind** | gain | **GenAI, question-aware** | gain |
|---|---|---|---|---|---|
| 2.5% | 4.9 → 6.8 | 4.9 → 9.1 | +4.2 [+2.1, +6.7], 16 / 0 | 10.5 → 10.2 | −0.3 [−3.3, +2.3], 9 / 5 |
| 5% | 15.1 → 16.6 | 17.1 → 22.2 | +5.1 [+1.0, +9.5], 17 / 5 | 49.6 → 53.5 | +4.0 [+0.9, +7.5], 18 / 6 |
| 10% | 32.7 → 34.8 | 33.9 → 38.2 | +4.2 [+0.7, +8.0], 19 / 5 | 67.6 → 72.3 | +4.7 [+1.5, +8.3], 21 / 8 |
| 15% | 43.8 → 49.8 | 42.3 → 52.7 | +10.4 [+5.4, +15.9], 29 / 7 | 75.3 → 81.3 | +6.0 [+2.1, +10.2], 18 / 6 |
| 25% | 61.2 → 64.8 | 60.9 → 67.4 | +6.5 [+0.6, +12.3], 27 / 11 | 80.7 → 89.7 | +9.1 [+4.6, +13.9], 19 / 3 |

- Question-blind: plain eviction inside GenAI reproduces the prototype within 1–2 points at every budget, and
  RestoreKV's gain is above sampling noise at all five budgets.
- Question-aware: absolute scores are much higher (the scorer knows the question); RestoreKV still adds +4 to +9
  points from 5% to 25%, although its adapter was trained for the question-blind setting.

Runs: `results/genai_integration/2026-10-18_genai_ruler4k_qblind_n10` (question-blind),
`..._2026-10-17_genai_ruler4k_n10` and `..._n10_full` (question-aware, full cache).

### 7.3 Speed and memory, Qwen3-8B (4K input, question-aware)

Total decode tokens/s over all sequences:

| Batch | Output tokens | Full cache | Eviction 10% | + RestoreKV |
|---|---|---|---|---|
| 1 | 2048 | 4.55 | 4.61 | 4.59 |
| 4 | 2048 | 13.21 | 15.70 | 15.69 |
| 8 | 1024 | 19.38 | 23.03 | 23.50 |
| 8 | 2048 | 20.61 | 27.05 | 26.83 |

- RestoreKV runs at the speed of plain eviction (same tokens/s, TTFT and memory): its 8 tokens are part of the prompt.
- Eviction adds ~3 s to the first token at 4K (TTFT 7.9 s full, 10.9–11.1 s with eviction, batch 1): on the iGPU,
  GenAI 2026.4.1 turns off the fused attention kernels during prefill when eviction is on. At 8K: TTFT 16.5 s full,
  31 s with eviction. At 16K, eviction ran out of memory during the single-chunk prefill (a limitation of the GenAI
  GPU path, not of RestoreKV); long-context numbers therefore come from the prototype (§4, §6).
- Process GPU memory is about the same in all configurations, because GenAI reserves its cache pool up front and
  each prompt must fit whole before the first eviction; after eviction a 4K sequence uses 56 MB of cache instead of
  0.6–0.9 GB, which is where the speed-up at batch 4–8 comes from.

## 8. Experiment 3: decode scaling in GenAI, three models

The design of `openvino_benchmarking`'s experiment 2 (fixed input, growing output), now with eviction and RestoreKV:
4K-token input from RULER-4K documents with a report request, 128 → 16K generated tokens (`ignore_eos`), GenAI
default eviction (question-aware, continuous) at 10% of the input. Models: Qwen3-8B, Qwen3-4B, Llama-3.1-8B-Instruct,
each with an adapter trained for OpenVINO H2O (`artifacts/adapters/ovblock_*`). Script
`experiments/genai_integration/decode_scaling.py`; results `results/genai_integration/2026-10-20_decode_scaling_*`.

### 8.1 Latency, throughput, memory (batch 1)

Time per output token (ms) / tokens/s / KV cache held per sequence:

| Output | Qwen3-8B full | eviction | + RestoreKV | Qwen3-4B full | eviction | + RestoreKV | Llama full | eviction | + RestoreKV |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 187 / 4.11 / 594 MB | 212 / 3.31 / 58 MB | 213 / 3.37 / 58 MB | 120 / 6.01 / 594 MB | 121 / 5.21 / 58 MB | 114 / 5.65 / 58 MB | 228 / 3.32 / 528 MB | 224 / 3.09 / 52 MB | 211 / 3.40 / 52 MB |
| 1K | 216 / 4.48 / 720 MB | 210 / 4.53 / 58 MB | 208 / 4.57 / 58 MB | 116 / 8.32 / 720 MB | 109 / 8.57 / 58 MB | 110 / 8.50 / 58 MB | 213 / 4.54 / 640 MB | 210 / 4.53 / 52 MB | 209 / 4.56 / 52 MB |
| 4K | 217 / 4.56 / 1.12 GB | 218 / 4.54 / 58 MB | 208 / 4.74 / 58 MB | 118 / 8.41 / 1.12 GB | 109 / 9.03 / 58 MB | 110 / 8.92 / 58 MB | 215 / 4.62 / 1.00 GB | 212 / 4.66 / 52 MB | 208 / 4.75 / 52 MB |
| 8K | 221 / 4.50 / 1.69 GB | 210 / 4.74 / 58 MB | 208 / 4.77 / 58 MB | 121 / 8.20 / 1.69 GB | 109 / 9.11 / 58 MB | 114 / 8.68 / 58 MB | 218 / 4.57 / 1.50 GB | 210 / 4.72 / 52 MB | 208 / 4.78 / 52 MB |
| 16K | 228 / 4.37 / 2.81 GB | 210 / 4.76 / 58 MB | 210 / 4.75 / 58 MB | 129 / 7.73 / 2.81 GB | 111 / 8.95 / 58 MB | 110 / 9.01 / 58 MB | 224 / 4.45 / 2.50 GB | 210 / 4.74 / 52 MB | 208 / 4.80 / 52 MB |

All lengths (128, 256, 512, 1K, 2K, 4K, 8K, 16K) are in the dashboard and each run's `results.csv`.

- The full cache grows with every token (to 2.6–2.9 GB per sequence at 16K output) and its time per token rises with
  it; with eviction the cache stays at 52–58.5 MB (−98%) and the time per token stays flat, so from ~1K output on
  eviction is faster (−6% to −14% per token at 16K).
- Below ~1K output eviction is slightly slower (scoring work, little cache to save), and its time to first token is
  ~3 s longer (fused prefill kernels off with eviction on the iGPU).
- RestoreKV runs at the speed and memory of plain eviction for every model and length.

### 8.2 Batch 8 (total tokens/s)

| Model | Output | Full cache | Eviction | + RestoreKV |
|---|---|---|---|---|
| Qwen3-8B | 2K | 19.7 | 26.3 | 26.6 |
| Qwen3-8B | 8K | does not fit | 30.5 | 30.1 |
| Qwen3-4B | 2K | 28.4 | 47.8 | 48.1 |
| Qwen3-4B | 8K | 25.1 | 56.0 | 55.9 |

Eight sequences share every read of the weights, and the evicted caches are small, so more of them fit and they decode
faster together: up to 2.2× the full cache's throughput (Qwen3-4B, 8K output). Llama-3.1-8B batch-8 points: see
`results/genai_integration/2026-10-20_decode_scaling_llama3.1-8b`.

### 8.3 How close the evicted outputs stay to the full-cache output

Same prompt, greedy decoding: "identical prefix" = tokens generated before the first difference from the full-cache
output; "4-gram overlap" = F1 of shared 4-token phrases (position-independent). One prompt per model, so this is an
illustration rather than a statistical result; accuracy is measured by RULER (§7.2).

| Model | Identical prefix, eviction / + RestoreKV | 4-gram overlap at 128 tokens | at 16K tokens |
|---|---|---|---|
| Qwen3-8B | 7 / **40** | 0.19 / **0.44** | 0.005 / 0.014 |
| Qwen3-4B | 11 / **23** | 0.26 / 0.27 | 0.004 / 0.006 |
| Llama-3.1-8B | 30 / 0* | 0.46 / 0.47 | 0.035 / 0.020 |

\* The Llama output with RestoreKV is the same text without a leading `**` markdown marker, so its first token
differs. Per-length tables: `agreement.csv` in each run folder.
