#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import os
import resource
import threading
import time
import traceback

import numpy as np
import openvino_genai as ov
from openvino import Core


# ============================================================
# Paths / models
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

MODEL_DIRS = {
    "phi-3.5-mini-int4": MODELS_DIR / "phi-3.5-mini-int4",
    "qwen3-4b-fp16-ov": MODELS_DIR / "qwen3-4b-fp16-ov",
    "qwen3-8b-fp16-ov": MODELS_DIR / "qwen3-8b-fp16-ov",
}

DEFAULT_MODEL = "phi-3.5-mini-int4"
DEFAULT_DEVICE = "CPU"
DEFAULT_INPUT_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 128

core = Core()


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenVINO GenAI long-context bottleneck profiler"
    )
    parser.add_argument("--model", choices=MODEL_DIRS.keys(), default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--input-tokens", type=int, default=DEFAULT_INPUT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BASE_DIR / "results" / "bottleneck_benchmark.csv",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier written to CSV.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="Memory sampling period in seconds.",
    )
    return parser.parse_args()


# ============================================================
# Basic helpers
# ============================================================

def validate_device(device):
    if ":" not in device:
        requested_devices = [device]
    else:
        plugin, device_list = device.split(":", 1)
        if plugin.upper() not in {"MULTI", "HETERO"}:
            raise ValueError(f"Unsupported plugin prefix in {device!r}")
        requested_devices = [
            item.strip() for item in device_list.split(",") if item.strip()
        ]

    for requested in requested_devices:
        base = requested.split("(", 1)[0].split(".", 1)[0].upper()
        if base not in core.available_devices:
            raise RuntimeError(
                f"{requested} is unavailable in {device}. "
                f"Available devices: {core.available_devices}"
            )


def read_proc_status():
    data = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if not parts:
                    continue
                if key in {"VmRSS", "VmHWM", "VmSwap"}:
                    # Linux reports these in kB.
                    data[key] = float(parts[0]) / 1024.0
    except Exception:
        pass
    return data


def read_meminfo():
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if parts:
                    values[key] = float(parts[0]) / 1024.0  # MB
    except Exception:
        pass

    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", 0.0)
    swap_total = values.get("SwapTotal", 0.0)
    swap_free = values.get("SwapFree", 0.0)

    return {
        "system_ram_total_mb": total,
        "system_ram_used_mb": max(0.0, total - available),
        "system_ram_available_mb": available,
        "system_swap_total_mb": swap_total,
        "system_swap_used_mb": max(0.0, swap_total - swap_free),
    }


def read_proc_io():
    values = {"read_bytes": 0, "write_bytes": 0}
    try:
        with open("/proc/self/io", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                if key in values:
                    values[key] = int(value.strip())
    except Exception:
        pass
    return values


def read_vmstat():
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    values = {k: 0 for k in wanted}
    try:
        with open("/proc/vmstat", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split()
                if key in wanted:
                    values[key] = int(value)
    except Exception:
        pass
    return values


def get_npu_allocated_memory_mb():
    """
    Best-effort query. Returns None when the property is not supported
    by the installed OpenVINO/NPU driver.
    """
    try:
        value = core.get_property("NPU", "NPU_DEVICE_ALLOC_MEM_SIZE")
        return float(value) / (1024.0 * 1024.0)
    except Exception:
        return None


class MemorySampler:
    def __init__(self, interval=0.25):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_process_rss_mb = 0.0
        self.peak_process_swap_mb = 0.0
        self.peak_system_ram_used_mb = 0.0
        self.peak_system_swap_used_mb = 0.0

    def _sample(self):
        while not self.stop_event.is_set():
            proc = read_proc_status()
            mem = read_meminfo()

            self.peak_process_rss_mb = max(
                self.peak_process_rss_mb, proc.get("VmRSS", 0.0)
            )
            self.peak_process_swap_mb = max(
                self.peak_process_swap_mb, proc.get("VmSwap", 0.0)
            )
            self.peak_system_ram_used_mb = max(
                self.peak_system_ram_used_mb,
                mem.get("system_ram_used_mb", 0.0),
            )
            self.peak_system_swap_used_mb = max(
                self.peak_system_swap_used_mb,
                mem.get("system_swap_used_mb", 0.0),
            )
            self.stop_event.wait(self.interval)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval * 4))


# ============================================================
# Prompt construction
# ============================================================

PARAGRAPH = (
    "Modern computer systems use heterogeneous processors to efficiently "
    "execute machine learning workloads. Large language models perform "
    "autoregressive generation where each generated token depends on "
    "previously generated tokens. During inference, the key-value cache "
    "stores intermediate attention information from earlier tokens and "
    "therefore grows with sequence length. Efficient memory management, "
    "scheduling, and cache organization are important for deploying "
    "long-context language models on resource-constrained edge devices. "
)


def _ids_from_encoded(encoded):
    data = encoded.input_ids.data
    return np.asarray(data[0], dtype=np.int64)


def build_prompt(tokenizer, target_tokens):
    """
    Efficiently construct a synthetic prompt close to exactly target_tokens.

    Unlike the older implementation, this does not repeatedly retokenize
    an ever-growing string. It estimates the required repeats, tokenizes once,
    trims the token IDs to target_tokens, then detokenizes the trimmed IDs.
    """
    one = _ids_from_encoded(tokenizer.encode(PARAGRAPH))
    tokens_per_paragraph = max(1, len(one))

    repeats = max(1, math.ceil(target_tokens / tokens_per_paragraph) + 2)
    text = PARAGRAPH * repeats

    ids = _ids_from_encoded(tokenizer.encode(text))
    if len(ids) < target_tokens:
        # Extremely unlikely, but extend safely if tokenizer boundary effects
        # made the estimate too small.
        while len(ids) < target_tokens:
            text += PARAGRAPH
            ids = _ids_from_encoded(tokenizer.encode(text))

    trimmed = ids[:target_tokens]
    prompt = tokenizer.decode(trimmed)

    # Re-tokenize once to report the actual synthetic prompt size.
    actual = len(_ids_from_encoded(tokenizer.encode(prompt)))
    return prompt, actual


# ============================================================
# CSV
# ============================================================

FIELDNAMES = [
    "run_id",
    "timestamp",
    "status",
    "failure_stage",
    "error_type",
    "error_message",
    "model",
    "device",
    "requested_input_tokens",
    "synthetic_prompt_tokens",
    "input_tokens",
    "max_new_tokens",
    "generated_tokens",
    "ignore_eos",
    "load_time_ms",
    "ttft_ms",
    "tpot_ms_per_token",
    "generate_duration_ms",
    "inference_duration_ms",
    "tokenization_duration_ms",
    "detokenization_duration_ms",
    "throughput_tokens_per_second",
    "wall_pipeline_load_s",
    "wall_generation_s",
    "process_peak_rss_mb",
    "process_peak_swap_mb",
    "system_peak_ram_used_mb",
    "system_peak_swap_used_mb",
    "process_disk_read_mb_generation",
    "process_disk_write_mb_generation",
    "system_pswpin_delta",
    "system_pswpout_delta",
    "system_pgmajfault_delta",
    "npu_allocated_mem_before_mb",
    "npu_allocated_mem_after_mb",
    "npu_max_prompt_len_config",
    "npu_min_response_len_config",
]


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def blank_row(args):
    return {
        "run_id": args.run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "FAILED",
        "failure_stage": "",
        "error_type": "",
        "error_message": "",
        "model": args.model,
        "device": args.device,
        "requested_input_tokens": args.input_tokens,
        "synthetic_prompt_tokens": "",
        "input_tokens": "",
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": "",
        "ignore_eos": args.ignore_eos,
        "npu_max_prompt_len_config": "",
        "npu_min_response_len_config": "",
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    validate_device(args.device)

    model_path = MODEL_DIRS[args.model]
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    row = blank_row(args)

    print("=" * 90)
    print("OpenVINO GenAI Long-Context Bottleneck Profiler")
    print("=" * 90)
    print(f"Model           : {args.model}")
    print(f"Device          : {args.device}")
    print(f"Input target    : {args.input_tokens}")
    print(f"Output target   : {args.max_new_tokens}")
    print(f"Ignore EOS      : {args.ignore_eos}")
    print(f"CSV             : {args.output_csv}")
    print("=" * 90)

    # NPU is different: OpenVINO's LLM NPU path uses a statically configured
    # context allocation. Configure the requested prompt/response capacity so
    # long-context failures represent actual memory/compilation limits rather
    # than the 1024+128 defaults.
    pipeline_config = {}
    if args.device.upper() == "NPU":
        pipeline_config = {
            "MAX_PROMPT_LEN": int(args.input_tokens),
            "MIN_RESPONSE_LEN": max(128, int(args.max_new_tokens)),
        }
        row["npu_max_prompt_len_config"] = pipeline_config["MAX_PROMPT_LEN"]
        row["npu_min_response_len_config"] = pipeline_config["MIN_RESPONSE_LEN"]

    stage = "pipeline_load"
    try:
        npu_before = get_npu_allocated_memory_mb()
        row["npu_allocated_mem_before_mb"] = (
            "" if npu_before is None else npu_before
        )

        load_start = time.perf_counter()
        pipe = ov.LLMPipeline(str(model_path), args.device, pipeline_config)
        row["wall_pipeline_load_s"] = time.perf_counter() - load_start

        generation_config = pipe.get_generation_config()
        generation_config.max_new_tokens = args.max_new_tokens
        if args.ignore_eos:
            generation_config.ignore_eos = True
        pipe.set_generation_config(generation_config)

        tokenizer = pipe.get_tokenizer()

        stage = "build_prompt"
        prompt, synthetic_prompt_tokens = build_prompt(
            tokenizer, args.input_tokens
        )
        row["synthetic_prompt_tokens"] = synthetic_prompt_tokens
        print(f"Synthetic prompt tokens: {synthetic_prompt_tokens}")

        generation_kwargs = {"max_new_tokens": args.max_new_tokens}
        if args.ignore_eos:
            generation_kwargs["ignore_eos"] = True

        stage = "generation"

        io_before = read_proc_io()
        vm_before = read_vmstat()
        mem_sampler = MemorySampler(args.sample_interval)
        mem_sampler.start()

        gen_start = time.perf_counter()
        try:
            result = pipe.generate([prompt], **generation_kwargs)
        finally:
            row["wall_generation_s"] = time.perf_counter() - gen_start
            mem_sampler.stop()

        io_after = read_proc_io()
        vm_after = read_vmstat()

        row["process_peak_rss_mb"] = mem_sampler.peak_process_rss_mb
        row["process_peak_swap_mb"] = mem_sampler.peak_process_swap_mb
        row["system_peak_ram_used_mb"] = mem_sampler.peak_system_ram_used_mb
        row["system_peak_swap_used_mb"] = mem_sampler.peak_system_swap_used_mb

        row["process_disk_read_mb_generation"] = max(
            0, io_after["read_bytes"] - io_before["read_bytes"]
        ) / (1024.0 * 1024.0)
        row["process_disk_write_mb_generation"] = max(
            0, io_after["write_bytes"] - io_before["write_bytes"]
        ) / (1024.0 * 1024.0)

        row["system_pswpin_delta"] = max(
            0, vm_after["pswpin"] - vm_before["pswpin"]
        )
        row["system_pswpout_delta"] = max(
            0, vm_after["pswpout"] - vm_before["pswpout"]
        )
        row["system_pgmajfault_delta"] = max(
            0, vm_after["pgmajfault"] - vm_before["pgmajfault"]
        )

        metrics = result.perf_metrics

        row.update({
            "status": "SUCCESS",
            "input_tokens": metrics.get_num_input_tokens(),
            "generated_tokens": metrics.get_num_generated_tokens(),
            "load_time_ms": metrics.get_load_time(),
            "ttft_ms": metrics.get_ttft().mean,
            "tpot_ms_per_token": metrics.get_tpot().mean,
            "generate_duration_ms": metrics.get_generate_duration().mean,
            "inference_duration_ms": metrics.get_inference_duration().mean,
            "tokenization_duration_ms": metrics.get_tokenization_duration().mean,
            "detokenization_duration_ms": metrics.get_detokenization_duration().mean,
            "throughput_tokens_per_second": metrics.get_throughput().mean,
        })

        npu_after = get_npu_allocated_memory_mb()
        row["npu_allocated_mem_after_mb"] = (
            "" if npu_after is None else npu_after
        )

        if metrics.get_num_generated_tokens() < args.max_new_tokens:
            row["status"] = "PARTIAL"
            row["error_message"] = (
                f"Generated {metrics.get_num_generated_tokens()} of "
                f"{args.max_new_tokens} requested output tokens"
            )

        print("\n" + "-" * 90)
        print(f"Status       : {row['status']}")
        print(f"Input tokens : {row['input_tokens']}")
        print(f"Output tokens: {row['generated_tokens']}")
        print(f"TTFT         : {row['ttft_ms']:.2f} ms")
        print(f"TPOT         : {row['tpot_ms_per_token']:.2f} ms/token")
        print(f"Throughput   : {row['throughput_tokens_per_second']:.2f} tok/s")
        print(f"Peak RSS     : {row['process_peak_rss_mb']:.1f} MB")
        print(f"Peak swap    : {row['process_peak_swap_mb']:.1f} MB")
        print(
            f"Disk read during generation: "
            f"{row['process_disk_read_mb_generation']:.1f} MB"
        )
        print(
            f"System swap-in/out delta: "
            f"{row['system_pswpin_delta']} / {row['system_pswpout_delta']}"
        )

    except Exception as exc:
        row["status"] = "FAILED"
        row["failure_stage"] = stage
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc).replace("\n", " ")[:4000]

        print("\n" + "!" * 90)
        print(f"FAILED during {stage}")
        print(f"{type(exc).__name__}: {exc}")
        print("!" * 90)
        traceback.print_exc()

    finally:
        append_row(args.output_csv, row)
        print(f"\nResult written to: {args.output_csv}")

    # Return non-zero for a real failure so the outer runner can log it.
    if row["status"] == "FAILED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
