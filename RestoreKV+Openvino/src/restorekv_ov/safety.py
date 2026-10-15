"""Protect the desktop session from host-RAM exhaustion (the iGPU allocates from system RAM).

On 2026-10-08 two experiment runs triggered the kernel OOM killer, which killed the user's systemd/dbus/tmux
(and the remote session) instead of the experiment. Call `protect()` at the start of every heavy script.
"""
from __future__ import annotations

import os
import threading
import time


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("inf")


def protect(floor_gb: float = 3.0, interval_s: float = 0.5) -> None:
    """Make this process the OOM killer's first choice and exit it if MemAvailable drops below floor_gb."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass

    def watch():
        while True:
            m = mem_available_gb()
            if m < floor_gb:
                print(f"\n[safety] MemAvailable {m:.1f} GB < {floor_gb} GB: aborting to protect the system", flush=True)
                os._exit(75)
            time.sleep(interval_s)

    threading.Thread(target=watch, daemon=True).start()
