# OpenVINO benchmarking on an edge laptop

How fast do open LLMs run with OpenVINO GenAI on a laptop, on which device, and what stops them from going
further? This folder profiles `ov_genai.LLMPipeline` on an Intel Core Ultra 7 258V (CPU, Arc 140V iGPU, NPU,
30 GB of RAM shared by all three) and records time-to-first-token (TTFT), time per output token (TPOT), throughput,
memory and failures for several models, devices and sequence lengths.

Models: Phi-3.5-mini (INT4), Qwen3-4B, Qwen3-8B, Qwen3-14B (fp16), Gemma-3 4B / 12B (fp16), all from the
[OpenVINO Hugging Face hub](https://huggingface.co/OpenVINO).

## Setup

```bash
conda create -n ovbench python=3.11 && conda activate ovbench
pip install openvino openvino-genai openvino-tokenizers huggingface_hub psutil
pip install -r openvino_dashboard/requirements.txt

# models go to <repo>/models (git-ignored, ~90 GB for all of them)
python openvino_benchmarking/download_model.py
# the profilers look for models inside this folder
ln -s ../models openvino_benchmarking/models
```

To download only a subset, comment out entries in `TEXT_GENERATION_MODELS` in `download_model.py`.

## What is in here

| Path | What it does |
|---|---|
| `profiling/first_inference.py`, `load_model.py` | first steps: load a pipeline, generate once |
| `profiling/01_manual_profiler.py` | wall-clock split of pipeline load vs generation |
| `profiling/02_streaming_profiler.py` | streamer callback timing (TTFT and inter-token latency as seen by the user) |
| `profiling/03_tokenizer_explorer.py` | tokenizer / detokenizer behaviour |
| `profiling/04_timeline_profiler.py` | per-step timeline of one generation |
| `profiling/05_perfmetrics_profiler.py` | one run with OpenVINO's `PerfMetrics` (TTFT, TPOT, throughput, tokenize/detokenize time) → CSV |
| `profiling/05_perfmetrics_profiler_bottleneck.py` | same, plus process / system memory, swap and disk I/O sampled during the run |
| `profiling/05_perfmetrics_profiler_experiment2.py` | same, plus the KV-cache size estimate and NPU memory, for long generations |
| `profiling/06_runtime_introspection.py`, `inspect_profiling_info.py`, `explore_*.py` | what the runtime exposes: devices, properties, per-op profiling info |
| `profiling/run_full_benchmark.py` | sweep: models × devices × input lengths, one subprocess per run |
| `profiling/run_bottleneck_benchmark.py` | sweep for the memory bottleneck study (long inputs) |
| `profiling/run_experiment2_kv.py` | sweep: fixed ~1K-token prompt, 128 → 65,536 generated tokens (the KV cache grows with the output) |
| `openvino_dashboard/` | Streamlit dashboards over the collected CSVs |
| `notes/` | notes written while learning the GenAI API (loading, manual profiling, streaming) |
| `docs/analysis/openvino_inference_flow.md` | how a `compile_model` / `infer` call travels through the OpenVINO source |
| `results/` | logs of the sweep runs (one folder per launch) |

## Running

A single measurement:

```bash
python openvino_benchmarking/profiling/05_perfmetrics_profiler.py \
    --model qwen3-8b-fp16-ov --device GPU --input-tokens 1024 --max-new-tokens 256 --ignore-eos \
    --output-csv openvino_benchmarking/results/my_runs.csv
```

The sweeps (each run is a separate process, so a crash or out-of-memory kill only loses that run; failures are
written to `runner_failures.csv` in the run's log folder):

```bash
python openvino_benchmarking/profiling/run_full_benchmark.py        # several hours
python openvino_benchmarking/profiling/run_experiment2_kv.py        # ~1 day for all models and devices
python openvino_benchmarking/profiling/run_bottleneck_benchmark.py
```

Models, devices and lengths of a sweep are set in the CONFIGURATION block at the top of each runner. Close other
applications first: the iGPU and NPU allocate from the same system RAM, and a run that runs out of memory can take
the desktop session down with it.

Dashboards:

```bash
streamlit run openvino_benchmarking/openvino_dashboard/app.py               # device / model comparison
streamlit run openvino_benchmarking/openvino_dashboard/app_experiment2.py   # long generations, KV cache, memory
```

Both read the CSVs in `openvino_dashboard/data/` (the measurements below); copy new result CSVs there to view them.

## Findings

Device comparison (`openvino_dashboard/data/old_perf.csv`, medians over input lengths 128–8K, fp16 unless noted):

| Model | Device | TTFT | TPOT | Decode tokens/s |
|---|---|---|---|---|
| Phi-3.5-mini INT4 | CPU / GPU / NPU | 10.5 s / 0.27 s / 1.0 s | 53 / 30 / 38 ms | 19.0 / 33.3 / 26.4 |
| Qwen3-4B | CPU / GPU / NPU | 14.1 s / 0.47 s / 1.6 s | 158 / 110 / 115 ms | 6.3 / 9.1 / 8.7 |
| Qwen3-8B | CPU / GPU / NPU | 29.5 s / 1.3 s / 3.3 s | 283 / 226 / 205 ms | 3.5 / 4.4 / 4.9 |

Long generations (`openvino_dashboard/data/experiment2_kv_cache.csv`, ~1K-token prompt, 128 to 65,536 new tokens):

| Qwen3-8B | 1K total tokens | 9K | 33K | 66K |
|---|---|---|---|---|
| KV cache | 0.16 GB | 1.27 GB | 4.64 GB | 9.14 GB |
| GPU TPOT | 200 ms | 228 ms | 269 ms | out of memory |
| NPU TPOT | 207 ms | 215 ms | 265 ms | 335 ms |
| CPU TPOT | 279 ms | 307 ms | 396 ms | 530 ms |

What this says about LLM inference on this class of device:

- **The iGPU is the device for prompts.** Prefill is compute-bound; the GPU is 20–50× faster than the CPU to the
  first token and 2.5–4× faster than the NPU.
- **Decoding is memory-bound on every device.** Each new token reads all weights and the whole KV cache from the
  same DDR memory, so decode speed is close to (memory bandwidth) / (model + cache size): the three devices are
  within 2× of each other, and Qwen3-8B fp16 (16 GB of weights) decodes at only 4–5 tokens/s anywhere.
  `HETERO:CPU,GPU,NPU` is never faster than the best single device.
- **The KV cache is what limits long contexts.** For Qwen3-8B it costs 144 KiB per token; at 33K tokens it is 4.6 GB,
  TPOT has grown by a third, and at 66K tokens the GPU runs out of memory (weights + cache + driver buffers exceed
  what the iGPU may allocate from the 30 GB). The NPU needs a static maximum prompt length and only runs short
  prompts by default.
- **Memory is shared**, so "GPU memory" and "system RAM" are the same pool; swap activity during long GPU runs shows
  the pressure before a run fails.

This is why the second part of this repository ([RestoreKV+Openvino](../RestoreKV+Openvino)) works on shrinking
the KV cache: keeping 5–10% of it removes the long-context memory limit and speeds up decoding, and RestoreKV is a
way to keep more of the accuracy when doing so.
