from pathlib import Path
import csv
import argparse

import openvino_genai as ov
from openvino import Core


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

MODEL_DIRS = {
    "phi-3.5-mini-int4": MODELS_DIR / "phi-3.5-mini-int4",
    "qwen3-4b-fp16-ov": MODELS_DIR / "qwen3-4b-fp16-ov",
    "qwen3-8b-fp16-ov": MODELS_DIR / "qwen3-8b-fp16-ov",
    "qwen3-14b-fp16-ov": MODELS_DIR / "qwen3-14b-fp16-ov",
    "gemma-3-4b-it-fp16-ov": MODELS_DIR / "gemma-3-4b-it-fp16-ov",
}

DEFAULT_MODEL = "phi-3.5-mini-int4"
DEFAULT_DEVICES = ["CPU", "GPU", "NPU"]

DEFAULT_INPUT_TOKENS = 128
DEFAULT_MAX_NEW_TOKENS = 256

core = Core()


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenVINO GenAI workload profiler"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=MODEL_DIRS.keys(),
        help=f"Model name (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--device",
        type=str,
        action="append",
        dest="devices",
        metavar="DEVICE",
        help=(
            "Device/plugin configuration. Repeat for multiple devices. "
            "Examples: --device CPU --device GPU --device NPU "
            "--device HETERO:CPU,GPU,NPU"
        ),
    )

    parser.add_argument(
        "--input-tokens",
        type=int,
        default=DEFAULT_INPUT_TOKENS,
        help="Approximate number of input tokens",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of generated tokens",
    )

    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Continue generation after EOS until max_new_tokens is reached",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BASE_DIR / "results" / "perfmetrics_comparison_new.csv",
        help="CSV output file",
    )

    return parser.parse_args()


args = parse_args()

DEVICES = args.devices or DEFAULT_DEVICES


# ============================================================
# Validation
# ============================================================

def validate_device(device):
    if ":" not in device:
        requested_devices = [device]
    else:
        plugin, device_list = device.split(":", 1)

        if plugin.upper() not in {"MULTI", "HETERO"}:
            raise ValueError(
                f"Unsupported plugin prefix in {device!r}"
            )

        requested_devices = [
            item.strip()
            for item in device_list.split(",")
            if item.strip()
        ]

    for requested in requested_devices:
        base_device = (
            requested
            .split("(", 1)[0]
            .split(".", 1)[0]
            .upper()
        )

        if base_device not in core.available_devices:
            raise RuntimeError(
                f"{requested} is unavailable in {device}.\n"
                f"Available devices: {core.available_devices}"
            )


for device in DEVICES:
    validate_device(device)


# ============================================================
# Build controlled input workload
# ============================================================

def build_prompt(tokenizer, target_tokens):
    """
    Construct a synthetic prompt with approximately target_tokens.

    We repeatedly add the same technical paragraph and use the
    OpenVINO tokenizer to determine the resulting token count.
    """

    paragraph = (
        "Modern computer systems use heterogeneous processors to "
        "efficiently execute machine learning workloads. Large language "
        "models perform autoregressive generation where each generated "
        "token depends on previously generated tokens. During inference, "
        "the key-value cache stores intermediate attention information "
        "from earlier tokens and therefore grows with the sequence length. "
        "Efficient memory management and inference scheduling are important "
        "for deploying long-context language models on resource-constrained "
        "edge devices. "
    )

    text = ""

    while True:
        candidate = text + paragraph
        tokenized = tokenizer.encode(candidate)

        # Tokenizer output may be a Tensor-like object.
        try:
            count = int(tokenized.input_ids.shape[1])
        except AttributeError:
            try:
                count = tokenized.shape[-1]
            except AttributeError:
                count = len(tokenized)

        if count >= target_tokens:
            break

        text = candidate

    # Try to get as close as possible to requested length.
    tokenized = tokenizer.encode(text)

    try:
        actual_tokens = int(tokenized.input_ids.shape[1])
    except AttributeError:
        try:
            actual_tokens = tokenized.shape[-1]
        except AttributeError:
            actual_tokens = len(tokenized)

    return text, actual_tokens


# ============================================================
# Profiling
# ============================================================

model_name = args.model
model_path = MODEL_DIRS[model_name]

if not model_path.exists():
    raise FileNotFoundError(
        f"Model directory does not exist: {model_path}"
    )

print("=" * 80)
print("OpenVINO GenAI Workload Profiler")
print("=" * 80)
print(f"Model            : {model_name}")
print(f"Model path       : {model_path}")
print(f"Requested input  : {args.input_tokens} tokens")
print(f"Max new tokens   : {args.max_new_tokens}")
print(f"Ignore EOS       : {args.ignore_eos}")
print(f"Devices          : {', '.join(DEVICES)}")
print("=" * 80)


results = []


for device in DEVICES:

    print("\n" + "=" * 80)
    print(f"DEVICE: {device}")
    print("=" * 80)

    print("Loading pipeline...")

    pipe = ov.LLMPipeline(
        str(model_path),
        device
    )

    #newly added to set max_new_tokens and ignore_eos in generation config
    generation_config = pipe.get_generation_config()
    generation_config.max_new_tokens = args.max_new_tokens

    if args.ignore_eos:
        generation_config.ignore_eos = True

    pipe.set_generation_config(generation_config)

    print("Generation config:")
    print("  max_new_tokens:", generation_config.max_new_tokens)
    print("  ignore_eos    :", generation_config.ignore_eos)


    print("Pipeline loaded.")

    tokenizer = pipe.get_tokenizer()

    prompt, actual_input_tokens = build_prompt(
        tokenizer,
        args.input_tokens
    )

    print(f"Actual input tokens: {actual_input_tokens}")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
    }

    if args.ignore_eos:
        generation_kwargs["ignore_eos"] = True

    print("Generating...")

    result = pipe.generate(
        [prompt],
        **generation_kwargs
    )

    metrics = result.perf_metrics

    generated_tokens = metrics.get_num_generated_tokens()

    print("\n" + "-" * 80)
    print("PerfMetrics")
    print("-" * 80)

    print(
        f"Load Time              : "
        f"{metrics.get_load_time():.2f} ms"
    )

    print(
        f"Input Tokens           : "
        f"{metrics.get_num_input_tokens()}"
    )

    print(
        f"Generated Tokens       : "
        f"{generated_tokens}"
    )

    print(
        f"TTFT                   : "
        f"{metrics.get_ttft().mean:.2f} ms"
    )

    print(
        f"TPOT                   : "
        f"{metrics.get_tpot().mean:.2f} ms/token"
    )

    print(
        f"Generate Duration      : "
        f"{metrics.get_generate_duration().mean:.2f} ms"
    )

    print(
        f"Inference Duration     : "
        f"{metrics.get_inference_duration().mean:.2f} ms"
    )

    print(
        f"Tokenization Duration  : "
        f"{metrics.get_tokenization_duration().mean:.2f} ms"
    )

    print(
        f"Detokenization Duration: "
        f"{metrics.get_detokenization_duration().mean:.2f} ms"
    )

    print(
        f"Throughput             : "
        f"{metrics.get_throughput().mean:.2f} tokens/s"
    )

    results.append({
        "model": model_name,
        "device": device,
        "input_tokens": metrics.get_num_input_tokens(),
        "requested_input_tokens": args.input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "ignore_eos": args.ignore_eos,
        "load_time_ms": metrics.get_load_time(),
        "ttft_ms": metrics.get_ttft().mean,
        "tpot_ms_per_token": metrics.get_tpot().mean,
        "generate_duration_ms": metrics.get_generate_duration().mean,
        "inference_duration_ms": metrics.get_inference_duration().mean,
        "tokenization_duration_ms": metrics.get_tokenization_duration().mean,
        "detokenization_duration_ms": metrics.get_detokenization_duration().mean,
        "throughput_tokens_per_second": metrics.get_throughput().mean,
    })


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    f"{'Device':<25}"
    f"{'Input':>10}"
    f"{'Output':>10}"
    f"{'TTFT(ms)':>14}"
    f"{'TPOT(ms)':>14}"
    f"{'Throughput':>16}"
)

for row in results:
    print(
        f"{row['device']:<25}"
        f"{row['input_tokens']:>10}"
        f"{row['generated_tokens']:>10}"
        f"{row['ttft_ms']:>14.2f}"
        f"{row['tpot_ms_per_token']:>14.2f}"
        f"{row['throughput_tokens_per_second']:>16.2f}"
    )


# ============================================================
# CSV
# ============================================================

args.output_csv.parent.mkdir(
    parents=True,
    exist_ok=True
)

file_exists = args.output_csv.exists()

fieldnames = [
    "model",
    "device",
    "requested_input_tokens",
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
]

with args.output_csv.open(
    "a",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    if not file_exists:
        writer.writeheader()

    writer.writerows(results)


print(
    f"\nResults appended to: "
    f"{args.output_csv}"
)