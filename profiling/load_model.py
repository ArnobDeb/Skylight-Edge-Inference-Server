from pathlib import Path
import openvino_genai as ov_genai

# Path to the downloaded OpenVINO model
MODEL_PATH = Path("../models/phi-3.5-mini-int4")

print("=" * 60)
print("Loading OpenVINO GenAI Pipeline")
print("=" * 60)

pipeline = ov_genai.LLMPipeline(
    str(MODEL_PATH),
    device="[CPU, GPU, NPU]"
)

print("\n✓ Pipeline loaded successfully!")
print(f"Model path : {MODEL_PATH.resolve()}")
print("Device     : CPU")
