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
# PATHS / MODELS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

MODEL_DIRS = {
    "phi-3.5-mini-int4":
        MODELS_DIR / "phi-3.5-mini-int4",

    "qwen3-4b-fp16-ov":
        MODELS_DIR / "qwen3-4b-fp16-ov",

    "qwen3-8b-fp16-ov":
        MODELS_DIR / "qwen3-8b-fp16-ov",
}

DEFAULT_MODEL = "phi-3.5-mini-int4"
DEFAULT_DEVICE = "CPU"

core = Core()


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "OpenVINO GenAI Experiment 2 profiler: "
            "fixed input, increasing output / KV growth"
        )
    )

    parser.add_argument(
        "--model",
        choices=MODEL_DIRS.keys(),
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--input-tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--ignore-eos",
        action="store_true",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=(
            BASE_DIR
            / "results"
            / "experiment2_kv_cache.csv"
        ),
    )

    parser.add_argument(
        "--run-id",
        default="",
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
    )

    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

def validate_device(device):

    if ":" not in device:

        requested_devices = [device]

    else:

        plugin, device_list = device.split(":", 1)

        if plugin.upper() not in {
            "MULTI",
            "HETERO",
        }:
            raise ValueError(
                f"Unsupported plugin prefix: {device}"
            )

        requested_devices = [
            x.strip()
            for x in device_list.split(",")
            if x.strip()
        ]

    for requested in requested_devices:

        base = (
            requested
            .split("(", 1)[0]
            .split(".", 1)[0]
            .upper()
        )

        if base not in core.available_devices:

            raise RuntimeError(
                f"{requested} unavailable. "
                f"Available devices: "
                f"{core.available_devices}"
            )


# ============================================================
# MEMORY / SYSTEM METRICS
# ============================================================

def read_proc_status():

    result = {}

    try:

        with open(
            "/proc/self/status",
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if ":" not in line:
                    continue

                key, value = line.split(
                    ":",
                    1,
                )

                parts = value.strip().split()

                if not parts:
                    continue

                if key in {
                    "VmRSS",
                    "VmHWM",
                    "VmSwap",
                }:

                    result[key] = (
                        float(parts[0])
                        / 1024.0
                    )

    except Exception:
        pass

    return result


def read_meminfo():

    values = {}

    try:

        with open(
            "/proc/meminfo",
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                key, value = line.split(
                    ":",
                    1,
                )

                parts = value.strip().split()

                if parts:

                    values[key] = (
                        float(parts[0])
                        / 1024.0
                    )

    except Exception:
        pass

    total = values.get(
        "MemTotal",
        0.0,
    )

    available = values.get(
        "MemAvailable",
        0.0,
    )

    swap_total = values.get(
        "SwapTotal",
        0.0,
    )

    swap_free = values.get(
        "SwapFree",
        0.0,
    )

    return {

        "ram_total_mb":
            total,

        "ram_used_mb":
            max(
                0.0,
                total - available,
            ),

        "ram_available_mb":
            available,

        "swap_total_mb":
            swap_total,

        "swap_used_mb":
            max(
                0.0,
                swap_total - swap_free,
            ),
    }


def read_proc_io():

    result = {
        "read_bytes": 0,
        "write_bytes": 0,
    }

    try:

        with open(
            "/proc/self/io",
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                key, value = line.split(
                    ":",
                    1,
                )

                if key in result:

                    result[key] = int(
                        value.strip()
                    )

    except Exception:
        pass

    return result


def read_vmstat():

    wanted = {
        "pswpin",
        "pswpout",
        "pgmajfault",
    }

    result = {
        key: 0
        for key in wanted
    }

    try:

        with open(
            "/proc/vmstat",
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                key, value = line.split()

                if key in wanted:

                    result[key] = int(value)

    except Exception:
        pass

    return result


# ============================================================
# NPU MEMORY
# ============================================================

def get_npu_allocated_memory_mb():

    try:

        value = core.get_property(
            "NPU",
            "NPU_DEVICE_ALLOC_MEM_SIZE",
        )

        return (
            float(value)
            / (1024.0 * 1024.0)
        )

    except Exception:

        return None


# ============================================================
# MEMORY SAMPLER
# ============================================================

class MemorySampler:

    def __init__(
        self,
        interval=0.25,
    ):

        self.interval = interval

        self.stop_event = (
            threading.Event()
        )

        self.thread = None

        self.peak_rss_mb = 0.0
        self.peak_swap_mb = 0.0
        self.peak_ram_used_mb = 0.0
        self.peak_swap_used_mb = 0.0

        self.initial_rss_mb = 0.0
        self.final_rss_mb = 0.0

        self.samples = []

    def _sample(self):

        while not self.stop_event.is_set():

            proc = read_proc_status()
            mem = read_meminfo()

            rss = proc.get(
                "VmRSS",
                0.0,
            )

            swap = proc.get(
                "VmSwap",
                0.0,
            )

            ram = mem.get(
                "ram_used_mb",
                0.0,
            )

            swap_used = mem.get(
                "swap_used_mb",
                0.0,
            )

            if self.initial_rss_mb == 0.0:

                self.initial_rss_mb = rss

            self.final_rss_mb = rss

            self.peak_rss_mb = max(
                self.peak_rss_mb,
                rss,
            )

            self.peak_swap_mb = max(
                self.peak_swap_mb,
                swap,
            )

            self.peak_ram_used_mb = max(
                self.peak_ram_used_mb,
                ram,
            )

            self.peak_swap_used_mb = max(
                self.peak_swap_used_mb,
                swap_used,
            )

            self.samples.append(
                (
                    time.time(),
                    rss,
                    swap,
                )
            )

            self.stop_event.wait(
                self.interval
            )

    def start(self):

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.stop_event.set()

        if self.thread:

            self.thread.join(
                timeout=max(
                    1.0,
                    self.interval * 4,
                )
            )


# ============================================================
# PROMPT
# ============================================================

PARAGRAPH = (
    "Modern computer systems use heterogeneous processors "
    "to efficiently execute machine learning workloads. "
    "Large language models perform autoregressive generation "
    "where each generated token depends on previously generated "
    "tokens. During inference, the key-value cache stores "
    "intermediate attention information from earlier tokens "
    "and therefore grows with sequence length. Efficient "
    "memory management, scheduling, and cache organization "
    "are important for deploying long-context language models "
    "on resource-constrained edge devices. "
)


def ids_from_encoded(encoded):

    return np.asarray(
        encoded.input_ids.data[0],
        dtype=np.int64,
    )


def build_prompt(
    tokenizer,
    target_tokens,
):

    one = ids_from_encoded(
        tokenizer.encode(
            PARAGRAPH
        )
    )

    tokens_per_paragraph = max(
        1,
        len(one),
    )

    repeats = max(
        1,
        math.ceil(
            target_tokens
            / tokens_per_paragraph
        ) + 2,
    )

    text = (
        PARAGRAPH
        * repeats
    )

    ids = ids_from_encoded(
        tokenizer.encode(text)
    )

    while len(ids) < target_tokens:

        text += PARAGRAPH

        ids = ids_from_encoded(
            tokenizer.encode(text)
        )

    trimmed = ids[
        :target_tokens
    ]

    prompt = tokenizer.decode(
        trimmed
    )

    actual = len(
        ids_from_encoded(
            tokenizer.encode(prompt)
        )
    )

    return prompt, actual


# ============================================================
# ESTIMATED KV CACHE
# ============================================================

def try_get_model_config(
    model_path,
):

    config_path = (
        model_path
        / "config.json"
    )

    if not config_path.exists():

        return {}

    try:

        import json

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def estimate_kv_cache(
    model_path,
    sequence_tokens,
):

    """
    Estimate KV cache footprint.

    Formula for standard MHA/GQA KV cache:

        2 × layers × KV_heads × head_dim
          × sequence_length × bytes_per_element

    This is an analytical estimate, NOT a measurement of the
    actual OpenVINO allocation.

    We try several common HuggingFace config names.
    """

    cfg = try_get_model_config(
        model_path
    )

    def first(*names):

        for name in names:

            value = cfg.get(name)

            if value is not None:
                return value

        return None

    layers = first(
        "num_hidden_layers",
        "n_layer",
        "num_layers",
    )

    kv_heads = first(
        "num_key_value_heads",
        "num_kv_heads",
    )

    attention_heads = first(
        "num_attention_heads",
        "n_head",
    )

    head_dim = first(
        "head_dim",
    )

    hidden_size = first(
        "hidden_size",
        "n_embd",
    )

    if head_dim is None:

        if (
            hidden_size is not None
            and attention_heads is not None
        ):

            head_dim = (
                hidden_size
                // attention_heads
            )

    if kv_heads is None:

        kv_heads = attention_heads

    if (
        layers is None
        or kv_heads is None
        or head_dim is None
    ):

        return {
            "kv_layers": "",
            "kv_heads": "",
            "kv_head_dim": "",
            "kv_bytes_per_element": "",
            "kv_estimated_mb": "",
            "kv_estimated_gb": "",
        }

    # FP16/BF16 assumption.
    bytes_per_element = 2

    bytes_total = (
        2
        * int(layers)
        * int(kv_heads)
        * int(head_dim)
        * int(sequence_tokens)
        * bytes_per_element
    )

    return {

        "kv_layers":
            int(layers),

        "kv_heads":
            int(kv_heads),

        "kv_head_dim":
            int(head_dim),

        "kv_bytes_per_element":
            bytes_per_element,

        "kv_estimated_mb":
            bytes_total
            / (1024 ** 2),

        "kv_estimated_gb":
            bytes_total
            / (1024 ** 3),
    }


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

    "total_sequence_tokens",

    "ignore_eos",

    # Performance
    "load_time_ms",
    "ttft_ms",
    "tpot_ms_per_token",

    "generate_duration_ms",
    "inference_duration_ms",

    "tokenization_duration_ms",
    "detokenization_duration_ms",

    "throughput_tokens_per_second",

    # Wall time
    "wall_pipeline_load_s",
    "wall_generation_s",

    # Process memory
    "process_rss_before_mb",
    "process_rss_after_mb",
    "process_peak_rss_mb",
    "process_rss_delta_mb",

    "process_peak_swap_mb",

    # System memory
    "system_peak_ram_used_mb",
    "system_peak_swap_used_mb",

    # Disk / paging
    "process_disk_read_mb_generation",
    "process_disk_write_mb_generation",

    "system_pswpin_delta",
    "system_pswpout_delta",
    "system_pgmajfault_delta",

    # NPU
    "npu_allocated_mem_before_mb",
    "npu_allocated_mem_after_mb",
    "npu_allocated_mem_delta_mb",

    "npu_max_prompt_len_config",
    "npu_min_response_len_config",

    # KV analytical estimate
    "kv_layers",
    "kv_heads",
    "kv_head_dim",
    "kv_bytes_per_element",

    "kv_estimated_mb",
    "kv_estimated_gb",

    # Peak sequence
    "peak_sequence_tokens",
]


def append_row(
    path,
    row,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        if not exists:

            writer.writeheader()

        writer.writerow({
            key: row.get(
                key,
                "",
            )
            for key in FIELDNAMES
        })


def blank_row(args):

    return {

        "run_id":
            args.run_id,

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "status":
            "FAILED",

        "model":
            args.model,

        "device":
            args.device,

        "requested_input_tokens":
            args.input_tokens,

        "max_new_tokens":
            args.max_new_tokens,

        "ignore_eos":
            args.ignore_eos,

    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    validate_device(
        args.device
    )

    model_path = MODEL_DIRS[
        args.model
    ]

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model directory missing: "
            f"{model_path}"
        )

    row = blank_row(args)

    print("=" * 100)
    print(
        "OpenVINO GenAI "
        "EXPERIMENT 2 PROFILER"
    )
    print("=" * 100)

    print(
        f"Model           : "
        f"{args.model}"
    )

    print(
        f"Device          : "
        f"{args.device}"
    )

    print(
        f"Input tokens    : "
        f"{args.input_tokens}"
    )

    print(
        f"Output target   : "
        f"{args.max_new_tokens}"
    )

    print(
        f"Ignore EOS      : "
        f"{args.ignore_eos}"
    )

    print(
        f"CSV             : "
        f"{args.output_csv}"
    )

    print("=" * 100)

    pipeline_config = {}

    if args.device.upper() == "NPU":

        pipeline_config = {

            "MAX_PROMPT_LEN":
                int(
                    args.input_tokens
                ) + 64,

            "MIN_RESPONSE_LEN":
                max(
                    128,
                    int(
                        args.max_new_tokens
                    ),
                ),
        }

        row[
            "npu_max_prompt_len_config"
        ] = pipeline_config[
            "MAX_PROMPT_LEN"
        ]

        row[
            "npu_min_response_len_config"
        ] = pipeline_config[
            "MIN_RESPONSE_LEN"
        ]

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    stage = "pipeline_load"

    try:

        npu_before = (
            get_npu_allocated_memory_mb()
        )

        row[
            "npu_allocated_mem_before_mb"
        ] = (
            ""
            if npu_before is None
            else npu_before
        )

        load_start = (
            time.perf_counter()
        )

        pipe = ov.LLMPipeline(
            str(model_path),
            args.device,
            pipeline_config,
        )

        row[
            "wall_pipeline_load_s"
        ] = (
            time.perf_counter()
            - load_start
        )

        generation_config = (
            pipe.get_generation_config()
        )

        generation_config.max_new_tokens = (
            args.max_new_tokens
        )

        if args.ignore_eos:

            generation_config.ignore_eos = True

        pipe.set_generation_config(
            generation_config
        )

        tokenizer = (
            pipe.get_tokenizer()
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        stage = "build_prompt"

        prompt, actual_prompt_tokens = (
            build_prompt(
                tokenizer,
                args.input_tokens,
            )
        )

        row[
            "synthetic_prompt_tokens"
        ] = actual_prompt_tokens

        print(
            f"Synthetic prompt tokens: "
            f"{actual_prompt_tokens}"
        )

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        generation_kwargs = {

            "max_new_tokens":
                args.max_new_tokens,
        }

        if args.ignore_eos:

            generation_kwargs[
                "ignore_eos"
            ] = True

        stage = "generation"

        io_before = read_proc_io()
        vm_before = read_vmstat()
        proc_before = read_proc_status()

        row[
            "process_rss_before_mb"
        ] = proc_before.get(
            "VmRSS",
            "",
        )

        sampler = MemorySampler(
            args.sample_interval
        )

        sampler.start()

        generation_start = (
            time.perf_counter()
        )

        try:

            result = pipe.generate(
                [prompt],
                **generation_kwargs,
            )

        finally:

            wall_generation = (
                time.perf_counter()
                - generation_start
            )

            row[
                "wall_generation_s"
            ] = wall_generation

            sampler.stop()

        io_after = read_proc_io()
        vm_after = read_vmstat()
        proc_after = read_proc_status()

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        rss_before = proc_before.get(
            "VmRSS",
            0.0,
        )

        rss_after = proc_after.get(
            "VmRSS",
            0.0,
        )

        row[
            "process_rss_after_mb"
        ] = rss_after

        row[
            "process_rss_delta_mb"
        ] = (
            rss_after
            - rss_before
        )

        row[
            "process_peak_rss_mb"
        ] = sampler.peak_rss_mb

        row[
            "process_peak_swap_mb"
        ] = sampler.peak_swap_mb

        row[
            "system_peak_ram_used_mb"
        ] = sampler.peak_ram_used_mb

        row[
            "system_peak_swap_used_mb"
        ] = sampler.peak_swap_used_mb

        # ----------------------------------------------------
        # Disk
        # ----------------------------------------------------

        row[
            "process_disk_read_mb_generation"
        ] = max(
            0,
            io_after["read_bytes"]
            - io_before["read_bytes"],
        ) / (1024 ** 2)

        row[
            "process_disk_write_mb_generation"
        ] = max(
            0,
            io_after["write_bytes"]
            - io_before["write_bytes"],
        ) / (1024 ** 2)

        # ----------------------------------------------------
        # VM activity
        # ----------------------------------------------------

        row[
            "system_pswpin_delta"
        ] = max(
            0,
            vm_after["pswpin"]
            - vm_before["pswpin"],
        )

        row[
            "system_pswpout_delta"
        ] = max(
            0,
            vm_after["pswpout"]
            - vm_before["pswpout"],
        )

        row[
            "system_pgmajfault_delta"
        ] = max(
            0,
            vm_after["pgmajfault"]
            - vm_before["pgmajfault"],
        )

        # ----------------------------------------------------
        # PerfMetrics
        # ----------------------------------------------------

        metrics = (
            result.perf_metrics
        )

        input_tokens = (
            metrics.get_num_input_tokens()
        )

        generated_tokens = (
            metrics.get_num_generated_tokens()
        )

        total_sequence = (
            input_tokens
            + generated_tokens
        )

        row.update({

            "status":
                "SUCCESS",

            "input_tokens":
                input_tokens,

            "generated_tokens":
                generated_tokens,

            "total_sequence_tokens":
                total_sequence,

            "load_time_ms":
                metrics.get_load_time(),

            "ttft_ms":
                metrics.get_ttft().mean,

            "tpot_ms_per_token":
                metrics.get_tpot().mean,

            "generate_duration_ms":
                metrics
                .get_generate_duration()
                .mean,

            "inference_duration_ms":
                metrics
                .get_inference_duration()
                .mean,

            "tokenization_duration_ms":
                metrics
                .get_tokenization_duration()
                .mean,

            "detokenization_duration_ms":
                metrics
                .get_detokenization_duration()
                .mean,

            "throughput_tokens_per_second":
                metrics
                .get_throughput()
                .mean,

            "peak_sequence_tokens":
                total_sequence,
        })

        # ----------------------------------------------------
        # NPU after
        # ----------------------------------------------------

        npu_after = (
            get_npu_allocated_memory_mb()
        )

        row[
            "npu_allocated_mem_after_mb"
        ] = (
            ""
            if npu_after is None
            else npu_after
        )

        if (
            npu_before is not None
            and npu_after is not None
        ):

            row[
                "npu_allocated_mem_delta_mb"
            ] = (
                npu_after
                - npu_before
            )

        # ----------------------------------------------------
        # KV analytical estimate
        # ----------------------------------------------------

        kv = estimate_kv_cache(
            model_path,
            total_sequence,
        )

        row.update(kv)

        # ----------------------------------------------------
        # Partial generation
        # ----------------------------------------------------

        if (
            generated_tokens
            < args.max_new_tokens
        ):

            row[
                "status"
            ] = "PARTIAL"

            row[
                "error_message"
            ] = (
                f"Generated "
                f"{generated_tokens} / "
                f"{args.max_new_tokens}"
            )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print("\n" + "-" * 100)

        print(
            f"Status       : "
            f"{row['status']}"
        )

        print(
            f"Input tokens : "
            f"{input_tokens}"
        )

        print(
            f"Output tokens: "
            f"{generated_tokens}"
        )

        print(
            f"Sequence     : "
            f"{total_sequence}"
        )

        print(
            f"TTFT         : "
            f"{row['ttft_ms']:.2f} ms"
        )

        print(
            f"TPOT         : "
            f"{row['tpot_ms_per_token']:.2f} ms"
        )

        print(
            f"Throughput   : "
            f"{row['throughput_tokens_per_second']:.2f} tok/s"
        )

        print(
            f"Peak RSS     : "
            f"{row['process_peak_rss_mb']:.1f} MB"
        )

        print(
            f"RSS delta    : "
            f"{row['process_rss_delta_mb']:.1f} MB"
        )

        print(
            f"Peak swap    : "
            f"{row['process_peak_swap_mb']:.1f} MB"
        )

        print(
            f"Disk read    : "
            f"{row['process_disk_read_mb_generation']:.1f} MB"
        )

        print(
            f"Disk write   : "
            f"{row['process_disk_write_mb_generation']:.1f} MB"
        )

        print(
            f"Swap in/out  : "
            f"{row['system_pswpin_delta']} / "
            f"{row['system_pswpout_delta']}"
        )

        if row["kv_estimated_mb"] != "":

            print(
                f"Estimated KV : "
                f"{row['kv_estimated_mb']:.1f} MB "
                f"({row['kv_estimated_gb']:.3f} GB)"
            )

        print("-" * 100)

    except Exception as exc:

        row[
            "failure_stage"
        ] = stage

        row[
            "error_type"
        ] = type(exc).__name__

        row[
            "error_message"
        ] = str(exc).replace(
            "\n",
            " ",
        )[:4000]

        print("\n" + "!" * 100)

        print(
            f"FAILED during {stage}"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("!" * 100)

        traceback.print_exc()

    finally:

        append_row(
            args.output_csv,
            row,
        )

        print(
            f"\nResult written to: "
            f"{args.output_csv}"
        )

    if row["status"] == "FAILED":

        raise SystemExit(2)


if __name__ == "__main__":

    main()