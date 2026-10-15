# eviction_study: long context, memory and speed

| Script | What it measures |
|---|---|
| `long_context_eval.py` | RULER-32K and LOFT-32K: full cache vs eviction vs eviction + RestoreKV. One prefill per document; all variants decode on the same device cache with their own masks; LOFT's 10 questions per document share the prefill. Qwen3-8B with INT8 weights (`benchmarks/long_context/make_int8_model.py`). |
| `throughput_memory.py` | decode tokens/s, time per token and device memory with the KV cache physically shortened to each budget (kept entries copied into a dense cache), batch 1 / 4 / 8, at 4K and 32K; plus one-time costs (prefill, scoring, restore pass). `--only-full` measures the full 32K cache from the prefill state. |

Long-context data (needs `git clone https://github.com/hsiehjackson/RULER artifacts/ref/RULER` with its data
downloads, a small venv in `env/venv-data` with RULER's requirements, `git clone https://github.com/google-deepmind/loft
artifacts/ref/LOFT`, and the LOFT zips from `https://storage.googleapis.com/loft-bench/<task>/<dataset>.zip` unpacked into
`artifacts/data/loft/data/<task>/`): `bash benchmarks/long_context/generate_ruler.sh` (NVIDIA RULER, Qwen3 tokenizer, 13 tasks) and
`python benchmarks/long_context/build_loft.py --lengths 32k` (LOFT RAG nq / hotpotqa / musique / qampari / quest and
retrieval nq / hotpotqa / scifact / arguana / fever / musique / qampari / quest, 10 queries each). 64K contexts do
not fit in the 30 GB of shared memory of the test laptop (Qwen3-8B fp16 works up to ~34K total tokens on its iGPU).
