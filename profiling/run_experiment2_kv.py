#!/usr/bin/env python3

import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROFILER = (
    Path(__file__).resolve().parent
    / "05_perfmetrics_profiler_experiment2.py"
)

MODELS = [
    "phi-3.5-mini-int4",
    "qwen3-4b-fp16-ov",
    "qwen3-8b-fp16-ov",
]

# ============================================================
# EXPERIMENT 2
#
# Fixed input context.
# Increasing generated sequence length.
#
# Purpose:
#   1. Observe decode scaling.
#   2. Observe memory growth.
#   3. Observe KV-cache-related memory growth.
#   4. Detect swap / disk activity / paging.
# ============================================================

FIXED_INPUT_TOKENS = 1024

OUTPUT_SWEEP = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]

DEVICES = [
    # "CPU",
    # "GPU",
    "NPU",
]

# ============================================================
# SEPARATE OUTPUTS
# ============================================================

RESULTS_CSV = (
    BASE_DIR
    / "results"
    / "experiment2_kv_cache.csv"
)

LOG_ROOT = (
    BASE_DIR
    / "results"
    / "experiment2_bottleneck_logs"
)

# No artificial timeout.
#
# We want the actual system/runtime behavior.
# The Linux OOM killer or OpenVINO itself can still terminate
# an impossible workload.
TIMEOUT_SECONDS = None

CONTINUE_ON_ERROR = True

DELAY_BETWEEN_RUNS = 5

# Don't automatically repeat a failed experiment.
RETRY_FAILED = True


# ============================================================
# HELPERS
# ============================================================

def experiment_key(model, device, input_tokens, output_tokens):
    return (
        model,
        device,
        int(input_tokens),
        int(output_tokens),
    )


def build_experiments():

    experiments = []

    for model in MODELS:

        for output_tokens in OUTPUT_SWEEP:

            for device in DEVICES:

                experiments.append({
                    "kind": "output_scaling",
                    "model": model,
                    "device": device,
                    "input_tokens": FIXED_INPUT_TOKENS,
                    "output_tokens": output_tokens,
                })

    return experiments


def load_completed():

    completed = set()

    if not RESULTS_CSV.exists():
        return completed

    with RESULTS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

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
                completed.add(key)

            elif (
                status in {"FAILED", "PARTIAL"}
                and not RETRY_FAILED
            ):
                completed.add(key)

    return completed


def model_exists(model):

    return (
        BASE_DIR
        / "models"
        / model
    ).exists()


def safe_name(text):

    return (
        text.replace(":", "_")
        .replace(",", "-")
        .replace("/", "_")
        .replace(" ", "_")
    )


def make_log_dir():

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = LOG_ROOT / stamp

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def append_runner_log(path, message):

    stamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{stamp}] {message}\n"

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(line)

    print(message, flush=True)


def append_runner_failure(
    path,
    experiment,
    reason,
    returncode="",
    log_file="",
):

    exists = path.exists()

    fields = [
        "timestamp",
        "kind",
        "model",
        "device",
        "input_tokens",
        "output_tokens",
        "reason",
        "returncode",
        "log_file",
    ]

    row = {
        "timestamp":
            time.strftime("%Y-%m-%d %H:%M:%S"),

        **experiment,

        "reason": reason,

        "returncode": returncode,

        "log_file": str(log_file),
    }

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_one(
    exp,
    run_id,
    index,
    total,
    log_dir,
    runner_log,
    failures_csv,
):

    model = exp["model"]
    device = exp["device"]
    inp = exp["input_tokens"]
    out = exp["output_tokens"]

    log_file = (
        log_dir
        / (
            f"{index:04d}_"
            f"{exp['kind']}_"
            f"{safe_name(model)}_"
            f"{safe_name(device)}_"
            f"in{inp}_"
            f"out{out}.log"
        )
    )

    command = [
        sys.executable,
        "-u",
        str(PROFILER),

        "--model",
        model,

        "--device",
        device,

        "--input-tokens",
        str(inp),

        "--max-new-tokens",
        str(out),

        "--ignore-eos",

        "--output-csv",
        str(RESULTS_CSV),

        "--run-id",
        run_id,
    ]

    append_runner_log(
        runner_log,
        (
            f"RUN {index}/{total} | "
            f"{model} | {device} | "
            f"input={inp} | output={out} | "
            f"log={log_file.name}"
        ),
    )

    start = time.time()

    with log_file.open(
        "w",
        encoding="utf-8",
    ) as log:

        log.write("COMMAND:\n")
        log.write(
            " ".join(command)
            + "\n\n"
        )
        log.flush()

        try:

            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        except KeyboardInterrupt:

            append_runner_log(
                runner_log,
                "Interrupted by user."
            )

            raise

    elapsed = time.time() - start

    if proc.returncode == 0:

        append_runner_log(
            runner_log,
            (
                f"  -> completed in "
                f"{elapsed / 60:.2f} min"
            ),
        )

        return True

    reason = (
        f"FAILED with return code "
        f"{proc.returncode}"
    )

    append_runner_log(
        runner_log,
        (
            f"  -> {reason}; "
            f"inspect {log_file}"
        ),
    )

    append_runner_failure(
        failures_csv,
        exp,
        reason,
        proc.returncode,
        log_file,
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("EXPERIMENT 2 — KV CACHE / OUTPUT SCALING")
    print("=" * 100)

    print(
        f"\nFixed input tokens : "
        f"{FIXED_INPUT_TOKENS}"
    )

    print(
        f"Output sweep      : "
        f"{OUTPUT_SWEEP}"
    )

    print("\nModels:")

    for model in MODELS:
        print(f"  - {model}")

    print("\nDevices:")

    for device in DEVICES:
        print(f"  - {device}")

    experiments = build_experiments()

    total = len(experiments)

    print(
        f"\nTotal planned runs : "
        f"{total}"
    )

    print(
        f"Results CSV        : "
        f"{RESULTS_CSV}"
    )

    print(
        f"Logs               : "
        f"{LOG_ROOT}"
    )

    print(
        "\nTimeout            : "
        "DISABLED"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Model validation
    # --------------------------------------------------------

    for model in MODELS:

        if not model_exists(model):

            raise SystemExit(
                f"\nMissing model:\n"
                f"{BASE_DIR / 'models' / model}"
            )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    completed = load_completed()

    pending = [

        exp

        for exp in experiments

        if experiment_key(
            exp["model"],
            exp["device"],
            exp["input_tokens"],
            exp["output_tokens"],
        )
        not in completed

    ]

    print(
        f"\nAlready completed : "
        f"{len(experiments) - len(pending)}"
    )

    print(
        f"Remaining         : "
        f"{len(pending)}"
    )

    if not pending:

        print(
            "\nEverything is already completed."
        )

        return

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_dir = make_log_dir()

    runner_log = (
        log_dir
        / "runner.log"
    )

    failures_csv = (
        log_dir
        / "runner_failures.csv"
    )

    run_id = log_dir.name

    append_runner_log(
        runner_log,
        "EXPERIMENT 2 START",
    )

    append_runner_log(
        runner_log,
        f"Run ID: {run_id}",
    )

    append_runner_log(
        runner_log,
        f"CSV: {RESULTS_CSV}",
    )

    append_runner_log(
        runner_log,
        f"Log directory: {log_dir}",
    )

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print(
        "\nThis experiment studies:"
    )

    print(
        "  • decode scaling"
    )

    print(
        "  • process memory growth"
    )

    print(
        "  • swap activity"
    )

    print(
        "  • disk I/O during generation"
    )

    print(
        "  • page faults"
    )

    print(
        "  • KV-cache growth indicators"
    )

    answer = input(
        "\nStart Experiment 2? [yes/no]: "
    ).strip().lower()

    if answer not in {"yes", "y"}:

        print("Cancelled.")

        return

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    success = 0
    failed = 0

    try:

        for i, exp in enumerate(
            pending,
            start=1,
        ):

            ok = run_one(
                exp,
                run_id,
                i,
                len(pending),
                log_dir,
                runner_log,
                failures_csv,
            )

            if ok:
                success += 1
            else:
                failed += 1

                if not CONTINUE_ON_ERROR:
                    break

            if DELAY_BETWEEN_RUNS:

                time.sleep(
                    DELAY_BETWEEN_RUNS
                )

    except KeyboardInterrupt:

        append_runner_log(
            runner_log,
            (
                "Benchmark interrupted. "
                "Completed CSV rows are preserved."
            ),
        )

        return

    append_runner_log(
        runner_log,
        "=" * 100,
    )

    append_runner_log(
        runner_log,
        (
            f"DONE | successful={success} | "
            f"failed={failed}"
        ),
    )

    append_runner_log(
        runner_log,
        "=" * 100,
    )


if __name__ == "__main__":
    main()