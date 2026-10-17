# genai_integration: RestoreKV inside OpenVINO GenAI

RestoreKV running in OpenVINO GenAI's own serving pipeline (`ContinuousBatchingPipeline`: PagedAttention,
continuous batching, real block eviction). The GenAI change is on branch `restorekv-2026.4.1` of
[ArnobDeb/openvino.genai](https://github.com/ArnobDeb/openvino.genai/tree/restorekv-2026.4.1)
([diff against 2026.4.1](https://github.com/ArnobDeb/openvino.genai/pull/1)).

How it works:

```
prompt ids:  [chat prefix + document] [V .. V+7] [question]       V = 151936 (rows of Qwen3's embedding table)
model (export_genai_model.py, once):   ids >= V -> learned restore embeddings; LoRA x (id >= V) on every projection
GenAI prefill (unchanged):             the 8 restore tokens attend to the full document = the restore pass
GenAI eviction (patched, ~90 lines):   blocks holding the 8 restore tokens are never evicted, count to max_cache_size
decoding (unchanged)
```

| Script | What |
|---|---|
| `export_genai_model.py` | writes a GenAI model folder: restore tokens as token ids, token-gated LoRA, rt_info naming the restore tokens |
| `check_export.py` | exported model vs the Python prototype on one sample (plain ids identical; restore tokens in the prompt = restore pass) |
| `genai_ruler_eval.py` | RULER-4K through `ContinuousBatchingPipeline`: OpenVINO eviction vs + RestoreKV, five budgets, full-cache reference |
| `genai_throughput.py` | TTFT, TPOT, tokens/s and memory: fixed input, 128-2048 output tokens, batch 1/4/8, full vs eviction vs + RestoreKV |

Build the patched GenAI against the installed OpenVINO 2026.4.1:

```bash
git clone -b restorekv-2026.4.1 https://github.com/ArnobDeb/openvino.genai && cd openvino.genai
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_TESTS=ON -DENABLE_SAMPLES=OFF \
      -DENABLE_TOOLS=OFF -DENABLE_XGRAMMAR=OFF -DBUILD_TOKENIZERS=OFF \
      -DOpenVINO_DIR=$(python -c "import openvino,os;print(os.path.dirname(openvino.__file__))")/cmake
cmake --build build -j 6
./build/tests/cpp/tests_continuous_batching --gtest_filter='CacheEviction*'
export PYTHONPATH=$PWD/build            # use this openvino_genai for the scripts below
```

Protocol difference to the prototype: GenAI evicts at the first decode step, after the question has been read
(its SnapKV window is the last 8 prompt tokens, i.e. the end of the question), so absolute scores are much higher
than in the question-blind prototype; the comparison that matters is without vs with RestoreKV under the same eviction.
Results: `results/genai_integration/`, summary in `docs/RESULTS.md` §7.
