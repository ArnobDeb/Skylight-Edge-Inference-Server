# How the RestoreKV prototype runs on OpenVINO

This document explains how the experiments drive OpenVINO: which objects exist, which calls are made, and in what
order. Code: `src/restorekv_ov/` (library) and `experiments/` (evaluation scripts).

## 1. The layers

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ experiments/*/  evaluation scripts (RULER, LOFT, throughput)                           │
│     one loop per sample: prefill → score → restore pass → evict → decode → score answer │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ src/restorekv_ov/  our driver library                                                  │
│   ir.py        edits the model graph once at load time                                  │
│   engine.py    OVRestoreEngine: owns the compiled model, the KV cache state, positions  │
│   scoring.py   eviction scores (OpenVINO H2O window scores, KVzip reconstruction)       │
│   eviction.py  turns scores into "which (layer, head, token) entries are dropped"       │
│   checkpoint.py loads the RestoreKV adapter (8 embeddings + LoRA weights)               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ OpenVINO runtime (Python API of the `openvino` package)                                │
│   ov.Core → read_model / compile_model("GPU") → CompiledModel → InferRequest            │
│   InferRequest.infer(), InferRequest.query_state() → VariableState (= the KV cache)     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ OpenVINO GPU plugin → Intel Arc iGPU (memory shared with the CPU)                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

The model is Qwen3-8B exported by `optimum-intel` as a **stateful** OpenVINO IR: the KV cache lives inside the
compiled model as 72 variables (36 layers × key/value), shape `[batch, 8 KV heads, tokens, 128]`. Every `infer()`
appends the new tokens' keys and values to these variables.

This is the OpenVINO runtime, not the OpenVINO GenAI `LLMPipeline` / `ContinuousBatchingPipeline`. GenAI's pipeline
has no way to run an extra pass with an adapter between reading the prompt and evicting, or to protect extra cache
slots, so the method is prototyped on the runtime underneath it.

## 2. Load time (once per run): editing the graph

`OVRestoreEngine.__init__` (engine.py):

| Step | Call | What it adds to the model graph |
|---|---|---|
| 1 | `core.read_model(openvino_model.xml)` | loads the exported graph |
| 2 | `ir.make_inputs_embeds_model` | replaces the token-id input with an `inputs_embeds` input, so learned vectors (the 8 restore tokens) can be fed; ordinary tokens are looked up in the embedding table on the host |
| 3 | `ir.add_gated_lora` | adds the adapter's LoRA branch next to each of the 252 projection matmuls, multiplied by a scalar input `lora_scale` (1 for the restore pass, 0 otherwise) |
| 4 | `ir.slice_logits_to_last_token` | computes the vocabulary logits only for the last position (saves ~1 GB per chunk) |
| 5 | `ir.add_query_outputs` | exposes the attention queries of the last few positions of each layer (needed for OpenVINO-style scoring) |
| 6 | `ir.add_kv_head_mask` | adds an input `kv_head_mask` that is added to every layer's attention mask: this is how eviction is applied |
| 7 | `core.compile_model(model, "GPU", {KV_CACHE_PRECISION: f16, ACTIVATIONS_SCALE_FACTOR: 8})` | compiles for the iGPU |
| 8 | `compiled.create_infer_request()` | one request = one KV cache |

One compiled model serves every phase (prefill, restore pass, decoding), so the KV cache never has to be copied
between models.

## 3. Per document: the call sequence

```
evaluation script                     OVRestoreEngine                         OpenVINO runtime
─────────────────                     ───────────────                         ────────────────
tokenize(context)
eng.prefill(ids, chunk=2048) ───────► forward(embed(chunk)) per chunk ──────► request.infer(...)   KV cache grows to T
                                      last chunk: q_rows=8 ─────────────────► returns the 8 window queries
scoring.ov_policy_scores ───────────► layer_keys(l) per layer ──────────────► query_state()[l].state (read keys)
   softmax(window queries · keys), summed over heads, max-pool 7, NORM_SUM → one score per (layer, token)
eng.restore_pass() ─────────────────► forward(8 restore embeddings, lora=1) ► request.infer(...)   cache = T + 8
for each budget and variant:
  eviction.ov_select(scores, budget) → bool mask: which (layer, head, token) are dropped
                                      (whole 16-token blocks, same for all heads, first 32 / last recent kept,
                                       budget minus the 8 restore slots)
  eng.mask_entries(evict) ──────────► stores the additive mask (−large for dropped entries)
  eng.forward(question) ────────────► request.infer(..., kv_head_mask) ──────► attention ignores dropped entries
  eng._greedy_batch(...) ───────────► one infer() per generated token
score the answer (exact match / LOFT metric)
```

Notes:

- **Positions:** the document sits at positions 0..T−1, the restore tokens at T..T+7, the question after them.
  Keys are stored after RoPE, so dropping entries never changes the positions of the ones that remain.
- **Eviction is a mask, not a deletion.** OpenVINO's stateful model needs every layer and head to hold the same
  number of entries, which per-head layouts (KVzip) cannot satisfy. The mask gives exactly the answer a deleted
  entry would give, so the quality numbers are exact for every layout. Memory and speed are measured separately with
  a physically shortened cache (`experiments/eviction_study/throughput_memory.py`, which copies the kept entries
  into a dense state with `set_state`).
- **Several variants per document:** the expensive part (prefill, scoring, restore pass) is done once; the eviction
  variants are decoded either as batch rows (`decode_variants`, 4K) or one after another on the same cache with
  their own masks (32K).
- **KVzip** (`scoring.kvzip_scores`) asks the model to repeat the document after reading it and records how much
  each cached entry is attended to while repeating (a small custom OpenVINO kernel computes the per-head maxima).
- **The adapter**: `checkpoint.RestoreCheckpoint` holds 8 embedding vectors and the LoRA matrices; they are trained
  in PyTorch (`experiments/adapter_training/`) with OpenVINO's eviction rule reimplemented in PyTorch
  (`src/restorekv_ov/ov_policy_torch.py`), tested to keep exactly the same blocks.

## 4. How we know the eviction matches OpenVINO's

OpenVINO GenAI 2026.4.1 was rebuilt with a small log in `cache_eviction.cpp` that prints, per layer, its block
scores and the evicted block IDs. Running the real `ContinuousBatchingPipeline` and our code on the same prompts
(13 prompts × 3 budgets × 36 layers): 99.7–99.9% of evicted blocks are identical
(`experiments/ov_prototype/runtime_crosscheck.py`).

## 5. What the OpenVINO GenAI pipeline does differently

`ContinuousBatchingPipeline` stores the cache in pages ("blocks") and runs PagedAttention kernels, with a scheduler
and the eviction algorithm in C++. Its prefill is faster (fused attention kernels, pre-allocated pages), and
eviction there frees real memory. Bringing RestoreKV into it means adding, in C++: the restore pass after the prompt
(adapter on), 8 protected cache slots, and the adapter loading.
