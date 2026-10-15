# 00: device baseline (stateful IR, plain OpenVINO runtime)

Date: 2026-10-08. openvino 2026.4.1. Batch 1, dummy tokens, one run each (no warm-up). These numbers are for sizing only; they are not benchmark results.

| Model | Device | Precision | Prefill 512 | Prefill 4096 | Decode | Host RSS |
|---|---|---|---|---|---|---|
| Qwen3-4B fp16 | CPU | f32 exec (default), KV f16 | 16.6 s (31 tok/s) | — | 129 ms/tok | **16.0 GB** |
| Qwen3-4B fp16 | CPU | f16 exec hint, KV f16 | 67.2 s | — | 171 ms/tok | 16.2 GB |
| Qwen3-4B fp16 | GPU (Arc iGPU) | f16 (default), KV default | 0.69 s (744 tok/s) | 7.5 s (545 tok/s) | 110–120 ms/tok | 2.8 GB |
| Qwen3-8B fp16 | GPU (Arc iGPU) | f16 (default), KV default | 0.99 s (515 tok/s) | 13.5 s (304 tok/s) | 205–239 ms/tok | 2.6 GB |

## Takeaways

- **The CPU runs fp16 IRs in f32 and unpacks the weights**: about 2× the weight size in RAM (4B → 16 GB), so **Qwen3-8B fp16 does not fit on CPU** (30 GB RAM). The f16 hint is slower (no native fp16 on Lunar Lake cores) and saves no memory.
- **The GPU is ~20× faster at prefill** and keeps fp16. GPU-visible memory is 29.3 GB (shared). → the prototype runs on the GPU.
- Default KV-cache precision is quantized (CPU `u8`, GPU `dynamic`). The prototype pins `KV_CACHE_PRECISION=f16` so `get_state`/`set_state` round-trips are exact. The quantized-KV path is a separate later experiment.
- KV state is readable on GPU via `query_state()`: 72 states, `[1, 8, T, 128]`, exposed as f32.
