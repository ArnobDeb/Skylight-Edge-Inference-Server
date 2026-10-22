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
| `genai_ruler_eval.py --query-blind` | the same RULER evaluation under the RestoreKV paper's **question-blind** protocol (evict after the document, before the question; `--restore-order inside|exact`) |
| `genai_throughput.py` | TTFT, TPOT, tokens/s and memory: fixed input, 128-2048 output tokens, batch 1/4/8, full vs eviction vs + RestoreKV |
| `decode_scaling.py` | decode scaling: 4K input, 128-16K output, batch 1 (+ batch 8), full / eviction / + RestoreKV, with the openvino_benchmarking metrics (TTFT, TPOT, throughput, durations, RSS, swap, disk reads, page faults, KV held, GPU memory) and the generated tokens |
| `output_agreement.py` | how long / how closely the evicted outputs follow the full-cache output |

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

### Two protocols

- **Question-aware** (GenAI as shipped, the default): the prompt `[document][restore ids][question]` is prefilled
  in one go, OpenVINO evicts at the first decode step with its SnapKV window on the end of the question, and keeps
  evicting while generating. Absolute scores are high because the scorer has seen the question.
- **Question-blind** (RestoreKV paper / prototype, opt-in): set `RESTOREKV_QUERY_BLIND=1` (or use
  `genai_ruler_eval.py --query-blind`). The prompt `[document][SEP][question]` (SEP = first restore id + 8) is cut at
  SEP; the document is prefilled and evicted once with the window on the document's end; the question is then fed
  token by token on the evicted cache, and nothing more is evicted. `RESTOREKV_QUERY_BLIND=2`
  (`--restore-order exact`) additionally feeds the restore tokens after the document prefill and keeps their
  attention out of the scores, exactly the prototype's order.

Compare without vs with RestoreKV within one protocol. Results: `results/genai_integration/`, summary in
`docs/RESULTS.md` §7 (accuracy, both protocols) and §8 (decode scaling). Dashboard: `streamlit run dashboard/app_genai.py`.
