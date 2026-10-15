"""04: system-memory footprint of GPU ContinuousBatching with/without cache eviction.

The iGPU allocates from system RAM and its buffers do not show up in process RSS. The kernel reports
them per process as `drm-total-system` in /proc/<pid>/fdinfo/<render fd>; this samples that while each
case runs in its own subprocess. A watchdog kills the case if its GPU memory exceeds --limit-gb or
MemAvailable drops below --floor-gb, so the kernel OOM killer never fires (it killed desktop
processes in the first diagnostic run). MemAvailable alone is unreliable: freed GPU pages linger in
the kernel TTM pool and are not counted as available.

    env -u PYTHONPATH python profiling/04_gpu_eviction_memory.py --cases 3072:2048:0,3072:2048:1
    case = prompt_len:chunk:evict
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASE = r'''
import sys, json, numpy as np, openvino as ov, openvino_genai as g
model, n, chunk, evict = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
sc = g.SchedulerConfig(); sc.cache_size = int(sys.argv[5]); sc.max_num_batched_tokens = chunk; sc.dynamic_split_fuse = True
sc.enable_prefix_caching = False; sc.use_cache_eviction = evict
if evict:
    sc.cache_eviction_config = g.CacheEvictionConfig(start_size=32, recent_size=128, max_cache_size=1024,
        aggregation_mode=g.AggregationMode.NORM_SUM, apply_rotation=False)
p = g.LLMPipeline(model, "GPU", scheduler_config=sc, KV_CACHE_PRECISION=ov.Type.f16)
print("LOADED", flush=True)
ids = np.random.default_rng(0).integers(1000, 30000, size=(1, n), dtype=np.int64)
try:
    r = p.generate(g.TokenizedInputs(ov.Tensor(ids), ov.Tensor(np.ones_like(ids))), g.GenerationConfig(max_new_tokens=4, min_new_tokens=4))
    print("RESULT", json.dumps({"ok": True, "ttft_ms": r.perf_metrics.get_ttft().mean}), flush=True)
except Exception as e:
    print("RESULT", json.dumps({"ok": False, "error": " ".join(str(e).split())[-160:]}), flush=True)
'''


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError


def gpu_mem_gb(pid: int) -> float:
    """Sum of drm-total-system + drm-total-gtt over the process's DRM render fds (xe driver)."""
    total_kib = 0
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        for fd in fd_dir.iterdir():
            try:
                if "renderD" not in str(fd.readlink()):
                    continue
                for line in open(f"/proc/{pid}/fdinfo/{fd.name}"):
                    if line.startswith(("drm-total-system:", "drm-total-gtt:")):
                        total_kib += int(line.split()[1])
            except OSError:
                continue
    except OSError:
        return 0.0
    return total_kib / 2**20


def run_case(model: Path, n: int, chunk: int, evict: int, cache_gb: int, floor_gb: float, limit_gb: float) -> dict:
    base = mem_available_gb()
    proc = subprocess.Popen([sys.executable, "-c", CASE, str(model), str(n), str(chunk), str(evict), str(cache_gb)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    import threading
    lines: list[str] = []
    threading.Thread(target=lambda: lines.extend(iter(proc.stdout.readline, "")), daemon=True).start()
    low, killed = base, None
    gpu_peak, gpu_after_load = 0.0, None
    while proc.poll() is None:
        m, g = mem_available_gb(), gpu_mem_gb(proc.pid)
        low, gpu_peak = min(low, m), max(gpu_peak, g)
        if gpu_after_load is None and any(l.startswith("LOADED") for l in lines):
            gpu_after_load = g
        if m < floor_gb or g > limit_gb:
            proc.kill(); killed = f"avail {m:.1f} GB" if m < floor_gb else f"gpu {g:.1f} GB"
            break
        time.sleep(0.05)
    proc.wait()
    time.sleep(0.5)
    res_line = next((l for l in lines if l.startswith("RESULT ")), None)
    res = json.loads(res_line[7:]) if res_line else {"ok": False, "error": "no result"}
    return {"prompt": n, "chunk": chunk, "evict": bool(evict), "cache_gb": cache_gb, **res,
            "killed_by_watchdog": killed, "gpu_mem_after_load_gb": round(gpu_after_load or 0, 2),
            "gpu_mem_peak_gb": round(gpu_peak, 2), "baseline_avail_gb": round(base, 2),
            "min_avail_gb": round(low, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-4b-fp16-ov")
    ap.add_argument("--cases", default="3072:2048:0,3072:2048:1")
    ap.add_argument("--cache-gb", type=int, default=2)
    ap.add_argument("--floor-gb", type=float, default=3.0)
    ap.add_argument("--limit-gb", type=float, default=20.0)
    args = ap.parse_args()
    rows = []
    for c in args.cases.split(","):
        n, chunk, evict = (int(x) for x in c.split(":"))
        rows.append(run_case(args.model, n, chunk, evict, args.cache_gb, args.floor_gb, args.limit_gb))
        print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "results/profiling" / f"{date.today()}_gpu_eviction_memory_{args.model.name}"
    prev = out / "metrics.json"
    rows = (json.loads(prev.read_text()) if prev.exists() else []) + rows  # append runs of the same day
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
