from pathlib import Path
import time
import openvino_genai as ov_genai

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

MODEL_PATH = Path("../models/phi-3.5-mini-int4")
DEVICE = "CPU"

PROMPT = input("Enter prompt: ")

MAX_NEW_TOKENS = 2000

# ----------------------------------------------------
# Timing variables
# ----------------------------------------------------

first_token_time = None
token_counter = 0

# ----------------------------------------------------
# Stream callback
# ----------------------------------------------------

def streamer(token: str):

    global first_token_time
    global token_counter

    now = time.perf_counter()

    if first_token_time is None:
        first_token_time = now

    token_counter += 1

    print(token, end="", flush=True)

    return False


# ----------------------------------------------------
# Load pipeline
# ----------------------------------------------------

print("\nLoading pipeline...")

t0 = time.perf_counter()

pipe = ov_genai.LLMPipeline(
    str(MODEL_PATH),
    device=DEVICE
)

t1 = time.perf_counter()

print("Done.\n")

# ----------------------------------------------------
# Generate
# ----------------------------------------------------

generation_start = time.perf_counter()

response = pipe.generate(
    PROMPT,
    max_new_tokens=MAX_NEW_TOKENS,
    streamer=streamer
)

generation_end = time.perf_counter()

print("\n")

# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

init_time = t1 - t0
generation_time = generation_end - generation_start

ttft = (
    first_token_time - generation_start
    if first_token_time
    else None
)

print("\n")
print("=" * 60)
print("Streaming Profiling")
print("=" * 60)

print(f"Initialization Time : {init_time:.3f} s")
print(f"Generation Time     : {generation_time:.3f} s")

if ttft is not None:
    print(f"TTFT                : {ttft:.3f} s")

print(f"Callback Invocations: {token_counter}")

if generation_time > 0:
    print(f"Callback/sec        : {token_counter/generation_time:.2f}")
