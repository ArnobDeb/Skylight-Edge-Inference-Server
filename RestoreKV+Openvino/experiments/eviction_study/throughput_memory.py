"""what eviction and RestoreKV buy in speed and memory, measured with a PHYSICALLY smaller KV cache.

The quality runs evict by masking (the full cache stays in memory), so their decode times say nothing about eviction.
Here the cache is compacted: for OpenVINO's layout (per layer, blocks shared by all heads, the same number of entries
in every layer) the kept entries are copied into a dense, shorter KV state, exactly what OpenVINO's paged cache holds
after eviction (compaction == masking was verified by kv_state_roundtrip.py). Then the question is decoded and timed.

Per context (RULER sample at 4K: fp16 model; at 32K: INT8 model) it records:
  * one-time costs: prefill, scoring (OpenVINO H2O, ADAPTIVE_RKV at 4K, KVzip at 4K), RestoreKV restore pass;
  * per configuration (full cache; OpenVINO eviction without / with RestoreKV at each budget) and batch size:
    KV entries kept, KV memory (MB, analytic), device memory of the process (drm fdinfo), time per output token,
    decode throughput (tokens/s = batch / TPOT).
With RestoreKV the restore slots come out of the budget, so the cache size equals plain eviction's; it is measured
anyway. Per-head / KVzip layouts keep the same number of entries at the same budget but cannot be stored densely in
OpenVINO's cache, so they are not timed (same memory, speed would need new kernels).

    env -u PYTHONPATH python experiments/eviction_study/throughput_memory.py --context 4k
    env -u PYTHONPATH python experiments/eviction_study/throughput_memory.py --context 32k
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/ov_prototype"))
from restorekv_ov import eviction as E  # noqa: E402
from restorekv_ov.engine import OVRestoreEngine  # noqa: E402
from restorekv_ov.safety import mem_available_gb, protect  # noqa: E402
from restorekv_ov.scoring import _AttnMaxKernel, kvzip_scores, ov_policy_scores  # noqa: E402

KV_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2          # Qwen3-8B, f16: layers x (K, V) x kv-heads x head_dim x 2 B


def gpu_mem_gb(pid: int) -> float:
    total = 0
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            if "renderD" not in str(fd.readlink()):
                continue
            for line in open(f"/proc/{pid}/fdinfo/{fd.name}"):
                if line.startswith(("drm-total-system:", "drm-total-gtt:")):
                    total += int(line.split()[1])
        except OSError:
            continue
    return total / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=["4k", "32k"], required=True)
    ap.add_argument("--budgets", default="0.025,0.05,0.10,0.15,0.25")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--decode-tokens", type=int, default=48)
    ap.add_argument("--only-full", action="store_true",
                    help="full cache, batch 1, decoded straight from the prefill state (no host copy of the cache): "
                         "the 32K full-cache baseline, which does not fit with the host-copy path")
    args = ap.parse_args()
    protect(floor_gb=1.5)
    long = args.context == "32k"
    model = ROOT / ("artifacts/models/qwen3-8b-int8-ov" if long else "../models/qwen3-8b-fp16-ov")
    model = model.resolve()
    recent = 128 if long else 32
    out_dir = ROOT / "results/eviction_study" / f"{date.today()}_throughput_memory"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    encode = lambda s: np.array(tok.encode(s, add_special_tokens=False), dtype=np.int64)  # noqa: E731
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    prefix_text, question_suffix = templ.split(sep)[0].split("CTX")[0], templ.split(sep)[1]
    if long:
        sys.path.insert(0, str(ROOT / "experiments/eviction_study"))
        from long_context_eval import load_ruler
        row = load_ruler(32768, 1, ["niah_multikey_2"])[0]
    else:
        from ruler_eval import load_samples
        row = dict(load_samples(4096, ["niah_multikey_2"], 1)[0][1])
    ctx_ids = encode(prefix_text + row["context"])
    q_ids = encode(row["question"] + question_suffix + row["answer_prefix"])
    T = len(ctx_ids)

    core = ov.Core()
    eng = OVRestoreEngine(model, "GPU", checkpoint=ROOT / "artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz",
                          core=core, compile_config={"ACTIVATIONS_SCALE_FACTOR": "8"})
    n_r = eng.n_restore
    pid = os.getpid()
    costs = {"context": args.context, "context_tokens": T, "model": model.name, "weights_gpu_mem_gb": round(gpu_mem_gb(pid), 2)}
    t = time.perf_counter()
    pre = eng.prefill(ctx_ids, chunk=2048, q_rows_last=8)
    costs["prefill_s"] = round(time.perf_counter() - t, 2)
    if args.only_full:
        t = time.perf_counter()
        logits = eng.forward(eng.embed(q_ids)).logits
        q_ms = 1000 * (time.perf_counter() - t)
        steps = []
        for _ in range(args.decode_tokens):
            feed = logits.argmax(-1).astype(np.int64)[:, None]
            t = time.perf_counter()
            logits = eng.forward(eng.embed(feed)).logits
            steps.append(time.perf_counter() - t)
        tpot = float(np.median(steps[2:]))
        row = {"context": args.context, "context_tokens": T, "config": "full", "budget": None, "batch": 1,
               "status": "ok", "kv_entries_per_layer": T, "kv_mb_per_sequence": round(T * KV_BYTES_PER_TOKEN / 1e6, 1),
               "kv_mb_total": round(T * KV_BYTES_PER_TOKEN / 1e6, 1), "kv_saved_vs_full_pct": 0.0,
               "process_gpu_mem_gb": round(gpu_mem_gb(pid), 2), "question_prefill_ms": round(q_ms, 1),
               "tpot_ms": round(1000 * tpot, 1), "decode_tokens_per_s": round(1 / tpot, 2)}
        print(row, flush=True)
        path = out_dir / "throughput.csv"
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(row)).writerow(row)
        return
    t = time.perf_counter()
    scores = ov_policy_scores(eng, pre.queries)
    costs["score_openvino_h2o_s"] = round(time.perf_counter() - t, 2)
    ctx_kv = eng.get_kv(dtype=np.float16)
    if not long:
        t = time.perf_counter()
        raw = ov_policy_scores(eng, pre.queries, max_pool=1, norm_sum=False)
        E.arkv_select(raw, [k.astype(np.float32) for k, _ in ctx_kv], 0.1, 0, n_kv_heads=8, block=16, recent_size=32)
        costs["score_and_select_adaptive_rkv_s"] = round(time.perf_counter() - t, 2)
        suffix_ids = encode(question_suffix)
        prefix_len = len(encode(prefix_text))
        t = time.perf_counter()
        kvzip_scores(eng, ctx_ids, prefix_len, encode, suffix_ids, kernel=_AttnMaxKernel(core, "GPU", eng.head_dim))
        costs["score_kvzip_s"] = round(time.perf_counter() - t, 2)
        eng.set_kv(ctx_kv)                                   # drop the KVzip replay tokens
    t = time.perf_counter()
    eng.restore_pass()
    costs["restore_pass_s"] = round(time.perf_counter() - t, 3)
    restore_kv = [(k[:, T:], v[:, T:]) for k, v in eng.get_kv(dtype=np.float16)]
    (out_dir / f"one_time_costs_{args.context}.json").write_text(json.dumps(costs, indent=2))
    print(costs, flush=True)

    configs = [("full", None, False)] + [(m, float(b), m.endswith("+restore"))
                                         for b in args.budgets.split(",") for m in ("ov", "ov+restore")]
    rows = []
    for name, b, restore in configs:
        if b is None:
            kv, n_entries = ctx_kv, T
        else:
            ev = E.ov_select(scores, b, n_r if restore else 0, n_kv_heads=8, block=16, start_size=32,
                             recent_size=recent)[:, 0]                                       # [L, T], head-shared
            keep = [np.flatnonzero(~ev[l]) for l in range(eng.n_layers)]
            assert len({len(k) for k in keep}) == 1, "OpenVINO layout keeps the same count in every layer"
            kv = [(k[:, idx], v[:, idx]) for (k, v), idx in zip(ctx_kv, keep)]
            if restore:
                kv = [(np.concatenate([k, rk], 1), np.concatenate([v, rv], 1)) for (k, v), (rk, rv) in zip(kv, restore_kv)]
            n_entries = kv[0][0].shape[1]
        for B in [int(x) for x in args.batches.split(",")]:
            need_gb = B * n_entries * KV_BYTES_PER_TOKEN / 2**30 * 1.5        # broadcast host copy + device state
            if need_gb > mem_available_gb() - 3:
                rows.append({"context": args.context, "context_tokens": T, "config": name, "budget": b, "batch": B,
                             "status": "skipped (does not fit in memory)"})
                continue
            eng.set_kv(kv, batch=B, positions_next=T + (n_r if restore else 0))
            t = time.perf_counter()
            logits = eng.forward(eng.embed(np.repeat(q_ids[None], B, 0))).logits
            q_ms = 1000 * (time.perf_counter() - t)
            steps = []
            for _ in range(args.decode_tokens):
                feed = logits.argmax(-1).astype(np.int64)[:, None]
                t = time.perf_counter()
                logits = eng.forward(eng.embed(feed)).logits
                steps.append(time.perf_counter() - t)
            tpot = float(np.median(steps[2:]))
            rows.append({"context": args.context, "context_tokens": T, "config": name, "budget": b, "batch": B,
                         "status": "ok", "kv_entries_per_layer": n_entries,
                         "kv_mb_per_sequence": round(n_entries * KV_BYTES_PER_TOKEN / 1e6, 1),
                         "kv_mb_total": round(B * n_entries * KV_BYTES_PER_TOKEN / 1e6, 1),
                         "kv_saved_vs_full_pct": round(100 * (1 - n_entries / T), 1),
                         "process_gpu_mem_gb": round(gpu_mem_gb(pid), 2),
                         "question_prefill_ms": round(q_ms, 1), "tpot_ms": round(1000 * tpot, 1),
                         "decode_tokens_per_s": round(B / tpot, 2)})
            print(rows[-1], flush=True)
    path = out_dir / "throughput.csv"
    new = not path.exists()
    fields = ["context", "context_tokens", "config", "budget", "batch", "status", "kv_entries_per_layer",
              "kv_mb_per_sequence", "kv_mb_total", "kv_saved_vs_full_pct", "process_gpu_mem_gb",
              "question_prefill_ms", "tpot_ms", "decode_tokens_per_s"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
