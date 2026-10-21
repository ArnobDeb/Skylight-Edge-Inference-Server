"""RULER with RestoreKV inside OpenVINO GenAI's ContinuousBatchingPipeline (PagedAttention, real eviction).

Needs the patched GenAI (build/genai-restorekv, protects the restore tokens from eviction) and a model folder from
export_genai_model.py. Per sample and budget, two requests go through the same pipeline:
    ov       [chat prefix + document][question]                 OpenVINO's eviction as shipped
    ov+R     [chat prefix + document][8 restore ids][question]  restore pass inside prefill, 8 protected slots
Eviction is OpenVINO's own (NORM_SUM, SnapKV window 8, start/recent 32, 16-token blocks on the GPU); the budget is
max_cache_size = budget x prompt length, rounded down to whole blocks, as in runtime_crosscheck.py. The
restore slots come out of the same max_cache_size.

    PY=<openvino.genai fork build dir>/openvino_genai
    env PYTHONPATH=$PY python experiments/genai_integration/genai_ruler_eval.py \
        --model artifacts/genai_models/qwen3-8b-fp16-restorekv-ovblock5000 --n-per-task 10

Output: results/genai_integration/<date>_genai_ruler4k_<name>/results.jsonl (+ summary.csv), resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/ov_prototype"))

from restorekv_ov.safety import protect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="folder written by export_genai_model.py")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--n-per-task", type=int, default=10)
    ap.add_argument("--budgets", default="0.025,0.05,0.10,0.15,0.25")
    ap.add_argument("--variants", default="ov,ov+R")
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--recent", type=int, default=32)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--block", type=int, default=16, help="KV block size of the device (16 on the GPU, 32 on the CPU)")
    ap.add_argument("--cache-gb", type=int, default=2)
    ap.add_argument("--max-batched-tokens", type=int, default=8192,
                    help="whole prompt in one chunk: multi-chunk prefill with eviction fails on the iGPU (2026.4.1)")
    ap.add_argument("--no-eviction", action="store_true", help="full-cache reference (ignores --budgets)")
    ap.add_argument("--query-blind", action="store_true",
                    help="evict once right after the document, before the question (RestoreKV paper / prototype "
                         "protocol; patched GenAI, RESTOREKV_QUERY_BLIND=1). Not what stock GenAI does.")
    ap.add_argument("--restore-order", choices=["inside", "exact"], default="inside",
                    help="question-blind only. inside: restore tokens in the prompt, 8 tokens before the document end "
                         "(one prefill). exact: prototype order - prefill + score the document, restore tokens fed "
                         "afterwards on the full cache (not scored), evict, then the question (RESTOREKV_QUERY_BLIND=2)")
    ap.add_argument("--then-continuous", action="store_true",
                    help="question-blind, inside order, then GenAI's continuous eviction once the answer starts "
                         "(RESTOREKV_QUERY_BLIND=3): the cache stays capped at max_cache_size during the answer")
    ap.add_argument("--out", default="genai_ruler4k")
    ap.add_argument("--max-requests", type=int, default=0, help="stop cleanly after this many requests (0 = all)")
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--mem-floor-gb", type=float, default=3.0)
    args = ap.parse_args()
    protect(floor_gb=args.mem_floor_gb)

    import os
    if args.then_continuous:
        assert args.query_blind and args.restore_order == "inside", "--then-continuous needs --query-blind, inside order"
    if args.query_blind:   # read at construction
        os.environ["RESTOREKV_QUERY_BLIND"] = ("2" if args.restore_order == "exact" else
                                               "3" if args.then_continuous else "1")
    import openvino as ov
    import openvino_genai as g
    from transformers import AutoTokenizer
    from ruler_eval import load_samples, string_match

    rk = json.loads((args.model / "restorekv.json").read_text())
    restore_ids = np.arange(rk["first_token_id"], rk["first_token_id"] + rk["num_tokens"], dtype=np.int64)
    tok = AutoTokenizer.from_pretrained(args.model)
    encode = lambda s: np.array(tok.encode(s, add_special_tokens=False), dtype=np.int64)  # noqa: E731
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    prefix_text, question_suffix = templ.split(sep)[0].split("CTX")[0], templ.split(sep)[1]

    samples = load_samples(args.seq_len, None, args.n_per_task)
    prompts = {}
    for idx, row in samples:
        ctx = encode(prefix_text + row["context"])
        q = encode(row["question"] + question_suffix + row["answer_prefix"])
        if args.query_blind:
            # [document][SEP][question]; the pipeline prefills up to SEP, evicts once, then feeds the question.
            # The SnapKV window is the last 8 prompt tokens, so the restore tokens go just before the last 8
            # document tokens: the scoring window is the document's end for both variants.
            sep = np.array([rk["first_token_id"] + rk["num_tokens"]], np.int64)
            w = args.window
            r_part = (np.concatenate([ctx, sep, restore_ids, q]) if args.restore_order == "exact"
                      else np.concatenate([ctx[:-w], restore_ids, ctx[-w:], sep, q]))
            prompts[idx] = {"ov": np.concatenate([ctx, sep, q]), "ov+R": r_part, "T": len(ctx), "Q": len(q)}
        else:
            prompts[idx] = {"ov": np.concatenate([ctx, q]), "ov+R": np.concatenate([ctx, restore_ids, q]), "T": len(ctx)}

    out_dir = ROOT / "results/genai_integration" / f"{args.date}_{args.out}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "results.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["idx"], r["variant"], r["budget"]))
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                      "restorekv": rk, "openvino_genai": g.__file__,
                                                      "openvino": ov.get_version()}, indent=2))

    def max_cache(n_tokens: int, budget: float) -> int:
        return max(int(budget * n_tokens) // args.block * args.block, args.start + args.recent + args.block)

    variants = args.variants.split(",")
    budgets = [None] if args.no_eviction else [float(b) for b in args.budgets.split(",")]
    # max_cache_size is a pipeline setting: one pipeline per distinct value, prompts grouped by it
    jobs: dict[int | None, list] = {}
    for b in budgets:
        for idx, row in samples:
            for v in (["full"] if b is None else variants):
                if (idx, v, b) in done:
                    continue
                ids = prompts[idx]["ov" if v == "full" else v]
                n_budget = prompts[idx]["T"] if args.query_blind else len(ids)   # restore slots inside the budget
                mc = None if b is None else max_cache(n_budget, b)
                if mc is not None and args.query_blind and args.restore_order == "exact" and v == "ov+R":
                    # restore entries sit past the scored range: take their slots out of the evictable area, as in
                    # the prototype (eviction.ov_select: keep (max_cache - start - recent - n_restore) // block blocks)
                    fixed = args.start + args.recent
                    mc = fixed + max(1, (mc - fixed - len(restore_ids)) // args.block) * args.block   # GenAI needs >= 1 block
                jobs.setdefault(mc, []).append((idx, row, v, b, ids))

    # end-of-turn tokens of Qwen3 and Llama-3.1 (only those the tokenizer knows)
    stop_ids = {i for i in (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>",
                                                                   "<|end_of_text|>")) if i is not None and i != tok.unk_token_id}
    props = {"KV_CACHE_PRECISION": "f16", "CACHE_DIR": str(ROOT / "build/ov_cache")}
    if args.device == "GPU":
        props["ACTIVATIONS_SCALE_FACTOR"] = "8"     # some adapters overflow fp16 in the restore tokens' LoRA
    n_requests = 0
    with rows_path.open("a") as fout:
        for mc, items in sorted(jobs.items(), key=lambda kv: kv[0] or 0):
            if args.max_requests and n_requests >= args.max_requests:
                break
            sc = g.SchedulerConfig()
            sc.cache_size = args.cache_gb
            sc.max_num_batched_tokens = args.max_batched_tokens
            sc.dynamic_split_fuse = False
            if mc is not None:
                cec = g.CacheEvictionConfig(args.start, args.recent, mc, g.AggregationMode.NORM_SUM)
                cec.apply_rotation = False
                cec.snapkv_window_size = args.window
                sc.use_cache_eviction = True
                sc.cache_eviction_config = cec
            t0 = time.perf_counter()
            pipe = g.ContinuousBatchingPipeline(str(args.model), sc, args.device, props)
            print(f"max_cache_size {mc}: pipeline ready in {time.perf_counter() - t0:.0f}s, {len(items)} requests",
                  flush=True)
            for idx, row, v, b, ids in items:
                if args.max_requests and n_requests >= args.max_requests:
                    break
                n_requests += 1
                gc = g.GenerationConfig()
                gc.max_new_tokens = int(row["max_new_tokens"])
                gc.stop_token_ids = stop_ids
                n_forced = prompts[idx]["Q"] if args.query_blind else 0
                if args.query_blind and args.restore_order == "exact" and v == "ov+R":
                    n_forced += len(restore_ids)     # restore tokens are fed as forced tokens before the question
                if n_forced:          # the question is emitted as the first generated tokens (the patched sampler
                    gc.max_new_tokens += n_forced     # ignores stop tokens inside it)
                t0 = time.perf_counter()
                res = pipe.generate([ov.Tensor(ids[None])], [gc])[0]
                wall = time.perf_counter() - t0
                gen = list(res.m_generation_ids[0])
                if n_forced:
                    gen = gen[n_forced:]
                    cut = next((i for i, t in enumerate(gen) if t in stop_ids), len(gen))
                    gen = gen[:cut]
                pred = tok.decode([t for t in gen if t not in stop_ids], skip_special_tokens=True)
                score = string_match(row["task"], pred, list(row["answer"]))
                rec = {"idx": idx, "task": row["task"], "variant": v, "budget": b, "max_cache_size": mc,
                       "prompt_tokens": len(ids), "context_tokens": prompts[idx]["T"], "score": score,
                       "pred": pred, "answer": list(row["answer"]), "wall_s": round(wall, 2),
                       "new_tokens": len(gen)}
                try:
                    pm = res.perf_metrics
                    rec.update(ttft_ms=round(pm.get_ttft().mean, 1), tpot_ms=round(pm.get_tpot().mean, 1))
                except Exception:
                    pass
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                print(f"[{idx} {row['task']} {v}@{b}] {score:.0f} {wall:.1f}s", flush=True)
            del pipe

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    agg: dict[tuple, list] = {}
    for r in rows:
        agg.setdefault((r["variant"], r["budget"]), []).append(r["score"])
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "budget", "n", "score"])
        for (v, b), s in sorted(agg.items(), key=lambda kv: (kv[0][1] or 1.0, kv[0][0])):
            w.writerow([v, b, len(s), round(float(np.mean(s)), 2)])
    print((out_dir / "summary.csv").read_text())


if __name__ == "__main__":
    main()
