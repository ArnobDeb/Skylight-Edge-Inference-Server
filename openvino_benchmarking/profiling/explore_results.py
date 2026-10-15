from pathlib import Path
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline(
    "../models/phi-3.5-mini-int4",
    "CPU"
)

result = pipe.generate(
    "What is OpenVINO?",
    max_new_tokens=20
)

print("=" * 60)
print("Python type")
print("=" * 60)
print(type(result))

print()

print("=" * 60)
print("dir(result)")
print("=" * 60)
print(dir(result))

print()

print("=" * 60)
print("repr(result)")
print("=" * 60)
print(repr(result))