# Skylight Edge Server

Running large language models on edge hardware (laptops and small servers with integrated GPUs and NPUs, no
discrete GPU) with OpenVINO, and making long-context inference fit and run fast there.

The target device for everything in this repository is an Intel Core Ultra 7 258V laptop: CPU, Arc 140V integrated
GPU and NPU sharing 30 GB of RAM. Models are served with Intel's [OpenVINO](https://github.com/openvinotoolkit/openvino)
runtime and [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai).

## The problem

On such a device two things limit an LLM:

1. **Decoding is memory-bound.** Every generated token reads all model weights and the whole KV cache from the
   shared DDR memory. Qwen3-8B (fp16) decodes at 4–5 tokens/s on any of the three devices.
2. **The KV cache grows with the context.** For Qwen3-8B it is 144 KiB per token: 4.8 GB for a 32K-token document,
   on top of 16 GB of weights. Around 64K tokens the iGPU runs out of memory, and long caches slow down every
   decode step.

Shrinking the KV cache (KV-cache eviction: after reading the prompt, keep only the most useful 5–25% of the cached
entries) addresses both, but costs accuracy.

## What is in this repository

| Folder | What |
|---|---|
| [`openvino_benchmarking/`](openvino_benchmarking) | Profiling of OpenVINO GenAI on the laptop: TTFT, TPOT, throughput and memory for Phi-3.5, Qwen3 4B/8B/14B and Gemma-3 on CPU, iGPU and NPU, for short and very long sequences; where the time and memory go. |
| [`RestoreKV+Openvino/`](RestoreKV+Openvino) | RestoreKV on the OpenVINO runtime: a learned "restore" step before KV-cache eviction, evaluated against OpenVINO's own eviction policies on RULER and LOFT, with memory and decode-speed measurements. |

## Main insights

From `openvino_benchmarking`:

- The iGPU gives the fastest prompt processing (Qwen3-8B, 1K-token prompt: 1.3 s to first token vs 29 s on the
  CPU); for decoding the devices are within 2× of each other because all of them wait on memory.
- Decode time per token grows with the KV cache (Qwen3-8B on the iGPU: 200 ms at 1K tokens, 269 ms at 33K), and
  at ~64K tokens the iGPU runs out of memory.

From `RestoreKV+Openvino`:

- Eviction makes long contexts practical on the laptop: at 32K tokens, keeping 5% of the cache cuts it from 4.8 GB
  to 0.24 GB, doubles single-sequence decode speed (3.0 → 6.6 tokens/s) and lets 8 sequences run together
  (38–40 tokens/s; the full cache fits only one).
- RestoreKV recovers part of the accuracy lost to eviction in the same memory budget, for a 0.3–2 s one-time
  restore step: +1.5 to +6.0 points on RULER-4K with OpenVINO's default eviction, and 34.4 → 68.4 at a 5% budget
  with the per-head eviction used in the RestoreKV paper.
- Integrated into OpenVINO GenAI's C++ serving pipeline (`ContinuousBatchingPipeline`,
  [fork + diff](https://github.com/ArnobDeb/openvino.genai/pull/1)). RULER-4K, Qwen3-8B, eviction → + RestoreKV:
  - **question-blind** (cache compressed after the document, before the question; the RestoreKV paper's protocol):
    gains of +4.2 to +10.4 points at all budgets (e.g. 42.3 → 52.7 at 15%), plain eviction matching the Python
    prototype within 1–2 points; with the prototype's exact order of operations the gains match the prototype's
    (+1.7 to +6.1);
  - **question-aware** (GenAI's default: compressed after reading the question): +3.9 to +9.1 points from 5% to 25%
    (80.7 → 89.7 at 25%).
- Experiment 3 (decode scaling, Qwen3-8B / Qwen3-4B / Llama-3.1-8B, 4K input, 128 → 16K output): eviction caps the
  KV cache at 52–58.5 MB per sequence (−98% vs 2.6–2.9 GB), keeps the time per token flat (−6% to −14% at 16K) and
  raises batch-8 throughput by about a third for the 8B models and more than doubles it for Qwen3-4B (25 → 56
  tokens/s); RestoreKV costs no speed or memory.
- How much it helps depends on how finely the cache is evicted: OpenVINO drops 16-token blocks for all heads of a
  layer at once, which loses far more than per-head, per-token eviction.

Each folder has its own README with setup, how to reproduce the results, and what to expect.

## Requirements

Linux, Python 3.11, an Intel CPU/GPU/NPU supported by OpenVINO, ~30 GB of RAM for 8B models. Models are downloaded
into `models/` at the repository root (not tracked by git) with `openvino_benchmarking/download_model.py`.
