from pathlib import Path
import openvino_genai as ov_genai

MODEL_PATH = Path("../models/phi-3.5-mini-int4")

pipe = ov_genai.LLMPipeline(
    str(MODEL_PATH),
    device="CPU"
)

print("=" * 80)
print("HELP FOR pipe.generate")
print("=" * 80)

help(pipe.generate)