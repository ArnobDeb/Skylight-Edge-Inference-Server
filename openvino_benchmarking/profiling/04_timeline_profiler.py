from pathlib import Path
import time
import openvino_genai as ov_genai

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path("../models/phi-3.5-mini-int4")

DEVICE = "CPU"

PROMPT = input("Prompt: ")

MAX_NEW_TOKENS = 50

# --------------------------------------------------
# Global state
# --------------------------------------------------

callback_events = []

generation_start = None

# --------------------------------------------------
# Stream callback
# --------------------------------------------------

def streamer(text):

    global callback_events

    now = time.perf_counter()

    elapsed = now - generation_start

    callback_events.append(
        {
            "time": elapsed,
            "text": text
        }
    )

    print(text, end="", flush=True)

    return False


# --------------------------------------------------
# Load pipeline
# --------------------------------------------------

print("\nLoading pipeline...")

t0 = time.perf_counter()

pipe = ov_genai.LLMPipeline(
    str(MODEL_PATH),
    device=DEVICE
)

t1 = time.perf_counter()

print("Done.\n")

# --------------------------------------------------
# Generate
# --------------------------------------------------

generation_start = time.perf_counter()

pipe.generate(
    PROMPT,
    max_new_tokens=MAX_NEW_TOKENS,
    streamer=streamer
)

generation_end = time.perf_counter()

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n")

print("=" * 80)
print("Callback Timeline")
print("=" * 80)

previous = None

for i, event in enumerate(callback_events):

    if previous is None:
        delta = 0
    else:
        delta = event["time"] - previous

    print(
        f"{i+1:02d} | "
        f"{event['time']:7.3f} s | "
        f"+{delta*1000:6.1f} ms | "
        f"{repr(event['text'])}"
    )

    previous = event["time"]

print()

print("=" * 80)
print("Summary")
print("=" * 80)

print(f"Callbacks      : {len(callback_events)}")
print(f"Generation Time: {generation_end-generation_start:.3f} s")
print(f"TTFT           : {callback_events[0]['time']:.3f} s")