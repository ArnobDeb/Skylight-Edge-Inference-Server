from pathlib import Path
import openvino_genai as ov_genai

MODEL_PATH = Path("../models/TinyLlama_1_1b_v1_ov")

print("=" * 60)
print("Loading Pipeline...")
print("=" * 60)

pipe = ov_genai.LLMPipeline(
    str(MODEL_PATH),
    device="CPU"
)

print("Pipeline loaded successfully!\n")

prompt = "What is OpenVINO?"

print("=" * 60)
print("Prompt")
print("=" * 60)
print(prompt)

print("\nGenerating...\n")

response = pipe.generate(
    prompt,
    max_new_tokens=100
)

print("=" * 60)
print("Model Response")
print("=" * 60)
print(response)
