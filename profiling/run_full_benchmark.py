#!/usr/bin/env python3

import csv
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROFILER = Path(__file__).resolve().parent / "05_perfmetrics_profiler.py"

# Five models you want to benchmark
    # "phi-3.5-mini-int4",
    # "qwen3-4b-fp16-ov",
    # "qwen3-8b-fp16-ov",
    # "qwen3-14b-fp16-ov",
    # "gemma-3-4b-it-fp16-ov",

MODELS = [
    "phi-3.5-mini-int4",
    "qwen3-4b-fp16-ov",
    "qwen3-8b-fp16-ov",
]

# Input context lengths
INPUT_TOKENS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]

# Output lengths
OUTPUT_TOKENS = [
    512,
]

# Devices
DEVICES = [
    "CPU",
    "GPU",
    "NPU",
    "HETERO:CPU,GPU,NPU",
]

# CSV produced by the profiler
RESULTS_CSV = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "perfmetrics_comparison_new.csv"
)

# ============================================================
# SAFETY / EXECUTION OPTIONS
# ============================================================

# Set to True if you want the script to continue automatically
# after an individual failed experiment.
CONTINUE_ON_ERROR = True

# Seconds to wait between experiments.
# Helps avoid immediately hammering the machine.
DELAY_BETWEEN_RUNS = 2

# ============================================================
# HELPERS
# ============================================================


def csv_key(model, device, input_tokens, output_tokens):
    return (
        model,
        device,
        int(input_tokens),
        int(output_tokens),
    )


def load_completed_runs():
    """
    Read existing CSV and return completed experiment keys.

    This makes the benchmark resumable. If the script is stopped,
    already completed runs will not be executed again.
    """

    completed = set()

    if not RESULTS_CSV.exists():
        return completed

    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "model",
            "device",
            "requested_input_tokens",
            "max_new_tokens",
        }

        if not required.issubset(reader.fieldnames or []):
            print(
                "WARNING: Existing CSV does not contain the expected "
                "benchmark columns."
            )
            return completed

        for row in reader:
            try:
                key = csv_key(
                    row["model"],
                    row["device"],
                    row["requested_input_tokens"],
                    row["max_new_tokens"],
                )
                completed.add(key)
            except (KeyError, ValueError):
                continue

    return completed


def model_exists(model):
    model_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / model
    )
    return model_path.exists()


def run_experiment(model, device, input_tokens, output_tokens, run_number, total_runs):
    print("\n")
    print("=" * 90)
    print(f"RUN {run_number}/{total_runs}")
    print("=" * 90)
    print(f"Model        : {model}")
    print(f"Device       : {device}")
    print(f"Input tokens : {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print("=" * 90)

    command = [
        sys.executable,
        str(PROFILER),
        "--model",
        model,
        "--device",
        device,
        "--input-tokens",
        str(input_tokens),
        "--max-new-tokens",
        str(output_tokens),
        "--ignore-eos",
    ]

    start = time.time()

    try:
        result = subprocess.run(command)

        elapsed = time.time() - start

        print("\n" + "-" * 90)
        print(f"Experiment completed in {elapsed / 60:.2f} minutes")
        print(f"Return code: {result.returncode}")
        print("-" * 90)

        if result.returncode != 0:
            print(
                f"WARNING: Experiment FAILED: "
                f"{model} / {device} / "
                f"{input_tokens} / {output_tokens}"
            )

            if not CONTINUE_ON_ERROR:
                sys.exit(result.returncode)

            return False

        return True

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        print("Already completed experiments remain in the CSV.")
        print("You can safely restart the script later.")
        sys.exit(130)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("OPENVINO FULL WORKLOAD BENCHMARK")
    print("=" * 90)

    print(f"Profiler       : {PROFILER}")
    print(f"Results CSV    : {RESULTS_CSV}")
    print()

    print("Models:")
    for model in MODELS:
        print(f"  - {model}")

    print("\nInput tokens:")
    print("  ", INPUT_TOKENS)

    print("\nOutput tokens:")
    print("  ", OUTPUT_TOKENS)

    print("\nDevices:")
    for device in DEVICES:
        print(f"  - {device}")

    total_runs = (
        len(MODELS)
        * len(INPUT_TOKENS)
        * len(OUTPUT_TOKENS)
        * len(DEVICES)
    )

    print("\n" + "=" * 90)
    print(f"TOTAL POSSIBLE RUNS: {total_runs}")
    print("=" * 90)

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    print("\nChecking model directories...")

    missing_models = []

    for model in MODELS:
        if model_exists(model):
            print(f"[OK]   {model}")
        else:
            print(f"[MISS] {model}")
            missing_models.append(model)

    if missing_models:
        print("\nERROR: The following models are missing:")

        for model in missing_models:
            print(f"  - {model}")

        print("\nDownload/convert them before starting.")
        sys.exit(1)

    # --------------------------------------------------------
    # Load existing progress
    # --------------------------------------------------------

    completed = load_completed_runs()

    print(
        f"\nExisting completed experiments found: "
        f"{len(completed)}"
    )

    remaining = total_runs - len(completed)

    print(f"Remaining experiments: {remaining}")

    if remaining == 0:
        print("\nEverything is already completed.")
        return

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print("\nWARNING:")
    print(
        "This benchmark can take a VERY long time, especially "
        "for large models and 4K/8K contexts."
    )

    print(
        "\nThe script will run experiments sequentially and "
        "append results through 05_perfmetrics_profiler.py."
    )

    answer = input("\nStart benchmark? [yes/no]: ").strip().lower()

    if answer not in {"yes", "y"}:
        print("Benchmark cancelled.")
        return

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    run_number = 0
    successful = 0
    failed = 0
    skipped = 0

    benchmark_start = time.time()

    for model in MODELS:

        for input_tokens in INPUT_TOKENS:

            for output_tokens in OUTPUT_TOKENS:

                for device in DEVICES:

                    key = csv_key(
                        model,
                        device,
                        input_tokens,
                        output_tokens,
                    )

                    if key in completed:
                        skipped += 1
                        print(
                            f"[SKIP] {model} | {device} | "
                            f"{input_tokens} -> {output_tokens}"
                        )
                        continue

                    run_number += 1

                    success = run_experiment(
                        model=model,
                        device=device,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        run_number=run_number,
                        total_runs=remaining,
                    )

                    if success:
                        successful += 1
                    else:
                        failed += 1

                    # Mark as completed only after profiler succeeds.
                    if success:
                        completed.add(key)

                    if DELAY_BETWEEN_RUNS > 0:
                        time.sleep(DELAY_BETWEEN_RUNS)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    elapsed = time.time() - benchmark_start

    print("\n")
    print("=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)

    print(f"Successful runs : {successful}")
    print(f"Failed runs     : {failed}")
    print(f"Skipped runs    : {skipped}")
    print(f"Elapsed time    : {elapsed / 3600:.2f} hours")

    print(f"\nResults CSV:")
    print(RESULTS_CSV)

    print("=" * 90)


if __name__ == "__main__":
    main()