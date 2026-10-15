# Environment

Hardware used for all results: Intel Core Ultra 7 258V (Lunar Lake), 8 cores, 30 GB LPDDR5X shared by CPU, Arc iGPU
and NPU. OpenVINO devices: CPU, GPU, NPU.

Software: Python 3.11, openvino / openvino-genai / openvino-tokenizers 2026.4.1, transformers 5.0.0, datasets 4.8.5,
nncf 3.2.0, torch (CPU) for the port tests. `requirements-ov.lock.txt` is a full package list of a working env.

Notes:
- Qwen3-8B fp16 does not fit on the CPU plugin (weights are unpacked to f32, ~32 GB): use `GPU`.
- Use `KV_CACHE_PRECISION=f16` for exact KV-state access (the defaults are quantized).
- If your shell exports a `PYTHONPATH` (e.g. ROS), run with `env -u PYTHONPATH python ...`.
- The iGPU allocates from system RAM; long runs call `restorekv_ov.safety.protect()` so that, if memory runs out,
  the experiment is the process the kernel stops.
