#!/usr/bin/env python3

import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
PROFILER = Path(__file__).resolve().parent / "05_perfmetrics_profiler_bottleneck.py"

MODELS = [
    "phi-3.5-mini-int4",
    "qwen3-4b-fp16-ov",
    "qwen3-8b-fp16-ov",
]

# For the GUIDE'S current question ("where does inference bottleneck
# as INPUT context grows?"), keep output small and fixed so decode
# does not dominate the measurement.
INPUT_SWEEPS = {
    # Phi-3.5-mini supports a much longer native context, so continue
    # beyond 32K until the machine/runtime fails or reaches 128K.
    "phi-3.5-mini-int4": [
        1024, 2048, 4096, 8192,
        12288, 16384, 24576, 32768,
        49152, 65536, 98304, 131072,
    ],

    # Qwen3-4B/8B are natively 32K context models. 32K is therefore
    # the clean native-context endpoint for this experiment.
    "qwen3-4b-fp16-ov": [
        1024, 2048, 4096, 8192,
        12288, 16384, 24576, 32768,
    ],
    "qwen3-8b-fp16-ov": [
        1024, 2048, 4096, 8192,
        12288, 16384, 24576, 32768,
    ],
}

# Fixed output for input/prefill bottleneck localization.
INPUT_SWEEP_OUTPUT_TOKENS = 128

# Run CPU/GPU/NPU first. HETERO can be added later as a separate
# scheduling experiment; it obscures device-local bottlenecks.
DEVICES = [
    "CPU",
    "GPU",
    "NPU",
    # "HETERO:CPU,GPU,NPU",
]

# Separate output/KV-growth sweep. This is NOT needed to answer where
# the input/prefill bottleneck occurs, but it is useful for studying
# KV-cache growth during decode and possible memory pressure.
RUN_OUTPUT_SWEEP = True
OUTPUT_SWEEP_INPUT_TOKENS = 1024

OUTPUT_SWEEPS = {
    "phi-3.5-mini-int4": [
        128, 256, 512, 1024, 2048, 4096,
        8192, 16384,
    ],
    "qwen3-4b-fp16-ov": [
        128, 256, 512, 1024, 2048, 4096,
        8192, 16384,
    ],
    "qwen3-8b-fp16-ov": [
        128, 256, 512, 1024, 2048, 4096,
        8192, 16384,
    ],
}

RESULTS_CSV = BASE_DIR / "results" / "bottleneck_benchmark.csv"

# Each experiment gets its own stdout/stderr log.
LOG_ROOT = BASE_DIR / "results" / "bottleneck_logs"

# A single run that exceeds this wall time is treated as hung.
# Increase if you deliberately want >4h per run.
TIMEOUT_SECONDS = 4 * 60 * 60

CONTINUE_ON_ERROR = True
DELAY_BETWEEN_RUNS = 5

# False = failed/partial runs are treated as completed observations
# (useful because the failure point itself is the bottleneck result).
# Set True only if you want to retry them automatically.
RETRY_FAILED = False


# ============================================================
# Experiment generation
# ============================================================

def experiment_key(model, device, input_tokens, output_tokens):
    return (model, device, int(input_tokens), int(output_tokens))


def build_experiments():
    experiments = []

    # A. Input/context scaling.
    for model in MODELS:
        for input_tokens in INPUT_SWEEPS[model]:
            for device in DEVICES:
                experiments.append({
                    "kind": "input_scaling",
                    "model": model,
                    "device": device,
                    "input_tokens": input_tokens,
                    "output_tokens": INPUT_SWEEP_OUTPUT_TOKENS,
                })

    # B. Output/decode scaling.
    if RUN_OUTPUT_SWEEP:
        for model in MODELS:
            for output_tokens in OUTPUT_SWEEPS[model]:
                for device in DEVICES:
                    experiments.append({
                        "kind": "output_scaling",
                        "model": model,
                        "device": device,
                        "input_tokens": OUTPUT_SWEEP_INPUT_TOKENS,
                        "output_tokens": output_tokens,
                    })

    return experiments


# ============================================================
# Resume / status
# ============================================================

def load_completed():
    completed = {}

    if not RESULTS_CSV.exists():
        return completed

    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = experiment_key(
                    row["model"],
                    row["device"],
                    row["requested_input_tokens"],
                    row["max_new_tokens"],
                )
            except Exception:
                continue

            status = row.get("status", "")
            if status == "SUCCESS":
                completed[key] = status
            elif status in {"FAILED", "PARTIAL"} and not RETRY_FAILED:
                completed[key] = status

    return completed


def model_exists(model):
    return (BASE_DIR / "models" / model).exists()


# ============================================================
# Logging
# ============================================================

def safe_name(text):
    return (
        text.replace(":", "_")
        .replace(",", "-")
        .replace("/", "_")
        .replace(" ", "_")
    )


def make_log_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_ROOT / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_runner_log(path, message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(message, flush=True)


def append_runner_failure(path, experiment, reason, returncode="", log_file=""):
    exists = path.exists()
    fields = [
        "timestamp", "kind", "model", "device",
        "input_tokens", "output_tokens", "reason",
        "returncode", "log_file",
    ]
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **experiment,
        "reason": reason,
        "returncode": returncode,
        "log_file": str(log_file),
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# Run one experiment
# ============================================================

def run_one(exp, run_id, index, total, log_dir, runner_log, failures_csv):
    model = exp["model"]
    device = exp["device"]
    inp = exp["input_tokens"]
    out = exp["output_tokens"]

    log_file = (
        log_dir
        / f"{index:04d}_{exp['kind']}_{safe_name(model)}_"
          f"{safe_name(device)}_in{inp}_out{out}.log"
    )

    command = [
        sys.executable,
        "-u",
        str(PROFILER),
        "--model", model,
        "--device", device,
        "--input-tokens", str(inp),
        "--max-new-tokens", str(out),
        "--ignore-eos",
        "--output-csv", str(RESULTS_CSV),
        "--run-id", run_id,
    ]

    append_runner_log(
        runner_log,
        f"RUN {index}/{total} | {exp['kind']} | {model} | "
        f"{device} | in={inp} | out={out} | log={log_file.name}",
    )

    start = time.time()

    with log_file.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(command) + "\n\n")
        log.flush()

        try:
            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            reason = f"TIMEOUT after {elapsed/3600:.2f} h"
            append_runner_log(runner_log, f"  -> {reason}")
            append_runner_failure(
                failures_csv, exp, reason, "TIMEOUT", log_file
            )
            return False
        except KeyboardInterrupt:
            append_runner_log(
                runner_log,
                "Interrupted by user. Completed CSV rows are preserved."
            )
            raise

    elapsed = time.time() - start

    if proc.returncode == 0:
        append_runner_log(
            runner_log,
            f"  -> completed in {elapsed/60:.2f} min"
        )
        return True

    reason = f"FAILED with return code {proc.returncode}"
    append_runner_log(runner_log, f"  -> {reason}; inspect {log_file}")
    append_runner_failure(
        failures_csv, exp, reason, proc.returncode, log_file
    )
    return False


# ============================================================
# Main
# ============================================================

def main():
    for model in MODELS:
        if not model_exists(model):
            raise SystemExit(f"Missing model directory: {BASE_DIR / 'models' / model}")

    experiments = build_experiments()
    completed = load_completed()

    pending = [
        exp for exp in experiments
        if experiment_key(
            exp["model"], exp["device"],
            exp["input_tokens"], exp["output_tokens"]
        ) not in completed
    ]

    log_dir = make_log_dir()
    runner_log = log_dir / "runner.log"
    failures_csv = log_dir / "runner_failures.csv"
    run_id = log_dir.name

    append_runner_log(runner_log, "=" * 100)
    append_runner_log(runner_log, "OPENVINO LONG-CONTEXT BOTTLENECK BENCHMARK")
    append_runner_log(runner_log, f"Run id      : {run_id}")
    append_runner_log(runner_log, f"Results CSV : {RESULTS_CSV}")
    append_runner_log(runner_log, f"Log dir     : {log_dir}")
    append_runner_log(runner_log, f"Total plan  : {len(experiments)} experiments")
    append_runner_log(runner_log, f"Already done: {len(experiments)-len(pending)}")
    append_runner_log(runner_log, f"Pending     : {len(pending)}")
    append_runner_log(runner_log, "=" * 100)

    if not pending:
        print("Nothing to run.")
        return

    print("\nExperiment A: INPUT SCALING")
    print(f"  Output fixed at {INPUT_SWEEP_OUTPUT_TOKENS} tokens.")
    print("  This is the primary bottleneck-localization experiment.")
    print("\nExperiment B: OUTPUT SCALING")
    print(f"  Input fixed at {OUTPUT_SWEEP_INPUT_TOKENS} tokens.")
    print("  This studies decode/KV-cache growth and memory pressure.")
    print(f"\nPer-run logs: {log_dir}")
    print(f"CSV:          {RESULTS_CSV}")

    answer = input("\nStart? [yes/no]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return

    success = 0
    failed = 0

    try:
        for i, exp in enumerate(pending, start=1):
            ok = run_one(
                exp, run_id, i, len(pending),
                log_dir, runner_log, failures_csv
            )
            if ok:
                success += 1
            else:
                failed += 1
                if not CONTINUE_ON_ERROR:
                    break

            if DELAY_BETWEEN_RUNS:
                time.sleep(DELAY_BETWEEN_RUNS)

    except KeyboardInterrupt:
        print("\nStopped. Re-run the script later; completed rows will be skipped.")
        return

    append_runner_log(runner_log, "=" * 100)
    append_runner_log(
        runner_log,
        f"DONE | successful={success} | failed={failed}"
    )
    append_runner_log(runner_log, "=" * 100)


if __name__ == "__main__":
    main()
