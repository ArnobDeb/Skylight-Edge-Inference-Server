"""Collect results/profiling/*/metrics.json into readable CSVs under results/profiling/csv/.

    python profiling/make_csv.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/profiling"
OUT = RES / "csv"

# Qwen3-4B/8B: 36 layers x 32 heads x head_dim 128 x 4 B / 256-token partitions = 2304 B per (chunk token x context token)
EVICT_MIXED_BYTES_PER_TOKEN2 = 36 * 32 * 128 * 4 / 256


def load(run_glob: str):
    for d in sorted(RES.glob(run_glob)):
        m = d / "metrics.json"
        if m.exists():
            yield d.name, json.loads(m.read_text()), json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}


def write(name: str, rows: list[dict]):
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(OUT / name, len(rows), "rows")


def short_error(e: str) -> str:
    if not e:
        return ""
    if "stopped manually" in e:
        return e
    if "CL_OUT_OF_RESOURCES" in e:
        return "GPU out of memory (CL_OUT_OF_RESOURCES)"
    if "-14" in e or "CL_EXEC_STATUS_ERROR" in e:
        return "GPU job failed, error -14 (host RAM exhausted)"
    return " ".join(e.split())[:100]


def r2(x, nd=2):
    return None if x is None else round(x, nd)


def long_context():
    rows = []
    for run, m, cfg in load("*_cb_long_context_*"):
        for r in m:
            ok = "error" not in r
            rows.append({
                "path": "ContinuousBatching (PagedAttention)", "model": Path(cfg.get("model", "")).name or run.split("_")[-3],
                "device": cfg.get("device", "GPU"), "eviction": "on" if r.get("evict") else "off",
                "budget_tokens": r.get("budget") if r.get("evict") else "", "prefill_chunk": cfg.get("chunk"),
                "context_tokens": r["context"], "status": "ok" if ok else "FAILED",
                "ttft_s": r2(r.get("ttft_ms", 0) / 1000) if ok else "", "tpot_ms": r2(r.get("tpot_ms"), 1) if ok else "",
                "decode_tok_s": r2(r.get("decode_throughput_tok_s")) if ok else "",
                "prefill_tok_s": r2(r.get("prefill_tok_s"), 0) if ok else "",
                "error": "" if ok else short_error(r["error"]), "run": run})
    for run, m, cfg in load("*_long_context_scaling_*"):
        for r in m["checkpoints"]:
            ok = "error" not in r
            rows.append({
                "path": "stateful IR (raw runtime)", "model": Path(cfg.get("model", "")).name, "device": cfg.get("device"),
                "eviction": "off", "budget_tokens": "", "prefill_chunk": cfg.get("chunk"),
                "context_tokens": r.get("context", r.get("stopped_at", r.get("stopped_after"))),
                "status": "ok" if ok else "FAILED",
                "ttft_s": r2(r.get("prefill_cum_s")) if ok else "", "tpot_ms": r2(r.get("tpot_ms"), 1) if ok else "",
                "decode_tok_s": r2(1000 / r["tpot_ms"]) if ok else "", "prefill_tok_s": r2(r.get("prefill_avg_tok_s"), 0) if ok else "",
                "error": "" if ok else short_error(r["error"]), "run": run})
    rows.sort(key=lambda r: (r["path"], r["eviction"], r["context_tokens"]))
    write("long_context_qwen3-8b_gpu.csv", rows)


def eviction_memory():
    rows = []
    for run, m, cfg in load("*_gpu_eviction_memory_*"):
        base = {r["prompt"]: r["gpu_mem_peak_gb"] for r in m if not r["evict"] and r.get("ok")}
        for r in m:
            mixed_tokens = 0 if r["prompt"] <= r["chunk"] else min(r["chunk"], r["prompt"] - r["chunk"])
            predicted = EVICT_MIXED_BYTES_PER_TOKEN2 * mixed_tokens * r["prompt"] / 2**30 if r["evict"] else 0
            rows.append({
                "model": run.split("_")[-1], "prompt_tokens": r["prompt"], "prefill_chunk": r["chunk"],
                "eviction": "on" if r["evict"] else "off", "status": "ok" if r.get("ok") else "FAILED",
                "ttft_s": r2(r["ttft_ms"] / 1000) if r.get("ttft_ms") else "",
                "gpu_mem_after_load_gb": r.get("gpu_mem_after_load_gb"), "gpu_mem_peak_gb": r.get("gpu_mem_peak_gb"),
                "peak_minus_noevict_same_prompt_gb": r2(r["gpu_mem_peak_gb"] - base[r["prompt"]]) if r["evict"] and r["prompt"] in base else "",
                "predicted_mixed_stage_extra_gb (2304*chunk*ctx)": r2(predicted) if r["evict"] else "",
                "kv_cache_gb": r.get("cache_gb"), "run": run})
    write("gpu_eviction_memory_qwen3-4b.csv", rows)


def eviction_diag():
    rows = []
    for run, m, cfg in load("*_gpu_eviction_diag_*"):
        for r in m:
            rows.append({"model": run.split("_")[-1], "prompt_tokens": r["prompt"], "prefill_chunk": r["chunk"],
                         "n_chunks": -(-r["prompt"] // r["chunk"]), "eviction": "on" if r["evict"] else "off",
                         "snapkv_window": r["snapkv"], "status": "ok" if r.get("ok") else "FAILED",
                         "ttft_s": r2(r["ttft_ms"] / 1000) if r.get("ttft_ms") else "",
                         "tpot_ms": r2(r.get("tpot_ms"), 1) if r.get("tpot_ms") else "",
                         "error": short_error(r.get("error") or ""), "run": run})
    write("gpu_eviction_failure_matrix_qwen3-4b.csv", rows)


if __name__ == "__main__":
    long_context()
    eviction_memory()
    eviction_diag()
