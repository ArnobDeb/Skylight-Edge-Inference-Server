# 06: Can the KV cache spill to the SSD instead of being evicted?

Date: 2026-10-12. Question from the student: OpenVINO keeps the whole KV cache in RAM and evicts (deletes) blocks
when it is full. Could evicted blocks go to the SSD instead, and be brought back when needed, at some latency cost?

## Numbers on this laptop (Core Ultra 7 258V, 30 GB LPDDR5X shared by CPU / iGPU / NPU, NVMe SSD)

| Quantity | Value | Source |
|---|---|---|
| KV per token, Qwen3-8B, f16 | 36 layers × 2 (K, V) × 8 heads × 128 × 2 B = **144 KiB** | model config |
| KV at 32K / 64K / 128K tokens | 4.8 / 9.7 / 19.3 GB | |
| One 16-token block, one layer | 64 KiB; all 36 layers: 2.3 MiB | |
| Weights fp16 / int8 | 16.4 / 8.8 GB | model files |
| RAM free for the model | ~20–22 GB | measured (`free`) |
| Effective RAM bandwidth during decode | ~79 GB/s ((16.4 + 4.8) GB / 269 ms TPOT at 32K) | profiling/02 |
| NVMe sequential read | **3.7 GB/s** (`dd iflag=direct bs=16M`, 4 GiB) | measured 2026-10-12 |

## Three ways to use the SSD

**A. Keep the whole KV cache on the SSD and stream it every decode step.** Decoding reads every cached entry once
per generated token. At 64K that is 9.7 GB per token: 2.6 s from the SSD vs ~0.12 s from RAM, so TPOT would be about
20× worse. Not viable.

**B. Let Linux swap handle it (swap file / mmap).** The iGPU's buffers are driver allocations that the kernel does
not page out. The two global OOMs on 2026-10-08 (profiling/05) show the kernel killing processes rather than
swapping GPU memory. Even if it did swap, every decode step touches all pages, so it would thrash like option A.

**C. Tiered cache: "evict to SSD, recall on demand" (recommended to study).** Keep the eviction budget in RAM as now,
but write evicted blocks to the SSD instead of deleting them, and bring a few back when they are needed:

- *When to recall:* once per question (query-aware): when the question arrives, re-score the offloaded blocks with
  the question's attention and fetch the best ones. Optionally per decode step for long answers.
- *Cost:* writing 90–95% of a 64K cache once is ~9 GB (3–4 s, overlappable with prefill). Recalling 5% of a 64K
  context (3.3K tokens, ~475 MB) takes ~0.13 s at 3.7 GB/s, small next to a multi-minute 64K prefill.
- *Ranking without reading the SSD:* keep a small per-block summary in RAM (e.g. the mean key per block and head:
  72 KiB per block for all layers, under 1% of the cache) and score blocks against the question's queries.
- *What changes in OpenVINO:* the ContinuousBatching block manager has no host or disk tier today (under memory
  pressure it preempts and recomputes). A third tier with copy-out / copy-in of physical blocks is needed.
  PagedAttention reads blocks through per-layer block tables, so a recalled block can be put back in its logical
  place, as long as `apply_rotation` is off (keys keep their original RoPE positions).
- *SSD wear:* ~9 GB written per 64K prompt; a consumer NVMe rated ~600 TBW allows tens of thousands of such prompts.

## How it changes the role of the eviction scores

Today a wrong eviction is permanent, and our results show OpenVINO's query-agnostic, head-shared block eviction
loses most of the quality at 5–10%. With a recall tier the eviction score only decides what stays *hot*; the question
can still pull back what it needs. RestoreKV stays useful in between: its summary slots cover what is neither resident
nor recalled.

Related published systems: FlexGen (offloading), InfiniGen and ShadowKV (keep keys or summaries resident, fetch
values), ArkVale (recallable KV eviction with page summaries), InfLLM (block memory with retrieval).

## Next step if pursued

The quality side can be estimated in the Python prototype before touching C++: a method that, at question time,
re-selects blocks from the full cache with the question's attention (query-aware recall of k% on top of the
resident budget), compared with permanent eviction at the same resident budget. The latency side is the
0.13 s-per-question recall plus the write-out, both measured above.
