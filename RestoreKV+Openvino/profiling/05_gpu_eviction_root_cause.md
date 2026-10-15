# 05: why OpenVINO cache eviction fails on the Arc iGPU (Qwen3)

Date: 2026-10-08. openvino / openvino-genai 2026.4.1 (conda env `arnob`). GPU source read from `../openvino` master @ f77c1e9b (2026-10-07).
Scripts: [03_gpu_eviction_diag.py](03_gpu_eviction_diag.py), [04_gpu_eviction_memory.py](04_gpu_eviction_memory.py), [02_cb_long_context.py](02_cb_long_context.py).
Raw data: `results/profiling/2026-10-08_*`.

## Symptom

`ContinuousBatchingPipeline` (via `LLMPipeline(..., scheduler_config=...)`) on `GPU` with `use_cache_eviction=True`:

- Qwen3-4B / 8B with prompts longer than one prefill chunk: `clWaitForEvents -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` already at 4K.
- One chunk for the whole prompt: works at 4K/8K. Qwen3-8B at 16K fails with `CL_OUT_OF_RESOURCES`.
- Without eviction, everything up to 32K works (8B: TTFT 120 s at 32K).
- TinyLlama works in all cases tried (it is small enough that the same overhead fits).

## Root cause: host RAM exhaustion, from eviction-only GPU buffers × 36 layers

The iGPU allocates from system RAM. The kernel log shows a **global OOM** during the failing runs: the OOM killer killed pipewire, wireplumber and gcr-ssh-agent, and gnome-shell was restarted. Then `xe: VM worker error: -12` (ENOMEM). The `-14` is the GPU job failing because its memory could not be backed. There were no GPU hangs or job timeouts (`xe` job timeout is 5000 ms; none logged).

Why eviction needs so much more memory (source: `intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp`):

1. Eviction needs per-token attention scores (`has_scores_output`). With scores requested, **`supports_micro_sdpa()` returns false**, so the oneDNN micro-SDPA (XMX flash-attention) kernels are disabled for every stage.
2. The second and later prompt chunks run in the **MIXED** stage. Without micro-SDPA, MIXED uses the partitioned "multi-token" kernel (`paged_attention_opt.cl`), which keeps per-partition intermediates for *every query token in the chunk*:
   `tmp_out = chunk_tokens × heads × head_size × ceil(context / 256) × 4 B`, plus `exp_sums` and `max_logits` (`get_internal_buffer_descs`, buffers 5–7).
3. Internal buffers are retained per primitive (`primitive_inst::_intermediates_memory`), so each of the **36 PagedAttention layers keeps its own copy**.

For Qwen3 (36 layers, 32 heads, head 128), the extra memory is about **2304 × chunk × context bytes**:

| Case (Qwen3-4B, cache 2 GB) | Predicted extra | Measured peak GPU mem (eviction off → on) |
|---|---|---|
| 3072 prompt, chunk 2048 | 36 × 201 MB = 7.25 GB | 9.97 → **17.51 GB** (+7.5 GB) |
| 2100 prompt, chunk 2048 (second chunk 52 tok) | 0.3 GB | → 10.94 GB |
| 8192 prompt, chunk 64 | ~1.2 GB | → 13.06 GB |
| Qwen3-8B, 4096, chunk 2048 | 19.3 GB (+16 GB weights) | OOM |
| Qwen3-8B, 32K, chunk 2048 | ~155 GB | impossible |

GPU memory is measured as `drm-total-system + drm-total-gtt` from `/proc/<pid>/fdinfo`; process RSS does not include it.

Single-chunk prefill (PREFILL stage) avoids the MIXED buffers, but it also runs without micro-SDPA:

| Qwen3-4B, 8K prompt | TTFT | Peak GPU mem |
|---|---|---|
| one chunk, eviction off | 7.4 s | 10.4 GB |
| one chunk, eviction on | 21.6 s (2.9× slower) | 12.9 GB |
| 64-token chunks, eviction on | 69.4 s (9.4× slower) | 13.1 GB |

Its extra memory grows with prompt length, so Qwen3-8B at 16K (16 GB weights + 6 GB cache) runs out.

**Side effect to note in the thesis:** on the iGPU, enabling eviction makes *prefill* 3–9× slower, because it switches off the flash-attention kernels. The decode-phase benefit of eviction has to be weighed against that.

## Why it matters for this project

- The baseline we must beat (OpenVINO eviction) cannot run at 32K/64K on this device in its current form.
- RestoreKV needs the same scores (it extends eviction), so it inherits the problem.
- **The CPU plugin does not have this problem.** It computes scores inline in its PA kernel (`attn_reduce` over heads). But fp16 weights are unpacked to f32 there (Qwen3-8B ≈ 32 GB, so it swaps; MTP experiment 2: TTFT 67 s for a 1K prompt).

## Options

| Option | What | Cost | Result |
|---|---|---|---|
| A. Small chunks (no code change) | `max_num_batched_tokens` small enough that 2304·C·L fits | none | Works, but TTFT ~9× worse (32K: C≈32 → ~1000 MIXED passes). OK as a stopgap baseline, not for a fair TTFT comparison |
| **B. Fix the GPU plugin (proposed)** | Keep micro-SDPA for the attention output. Compute eviction scores with a separate pass for only the last `snapkv_window_size` query rows (8), e.g. run the existing multi-token kernel on those 8 rows only. Buffers become 8 × heads × context × 4 B per layer (~33 MB at 32K) | Build OpenVINO core from source (GPU plugin), ~2–3 h first build; kernel-side change + tests | Eviction prefill as fast as no-eviction, and memory flat. A contribution in its own right (upstreamable) |
| C. Share the buffers across layers | Mark the MIXED intermediates reusable across PA primitives | small change | Removes the ×36, but a single 32K/2048 buffer is still 4.3 GB. Partial fix |

## NPU

- `LLMPipeline` on NPU runs Qwen3-8B (MTP experiment 2: 1K prompt, TTFT 6 s, TPOT ≈ 200 ms) once `MAX_PROMPT_LEN` is raised.
- Continuous batching / cache eviction **does not exist on NPU**. The NPUW LLM infer request (`intel_npu/src/plugin/npuw/llm_infer_request.cpp`) exposes **only LoRA and stored-token states** in `query_state()`, **not the KV cache**. So neither Python-side eviction nor RestoreKV can be done without C++ changes in NPUW.
- NPUW has what a port would need: native LoRA as runtime states (`NPUW_LLM_MAX_LORA_RANK`), a block-based KV strategy (`llm_block_kvcache_strategy`), and chunked prefill (`NPUW_LLM_PREFILL_CHUNK_SIZE`).
- **Opportunity:** NPU generate graphs have a *static* KV length (`MAX_PROMPT_LEN + MIN_RESPONSE_LEN`), and TPOT grows with it (MTP data: 206 → 335 ms from 1K to 65K total). Evicting once after prefill to a budget B lets the generate graph run with KV length B + response instead of the full context. That is a direct decode-speed and memory win specific to the NPU. Worth evaluating after the GPU path works.
