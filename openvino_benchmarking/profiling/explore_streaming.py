import inspect
import openvino_genai as ov_genai

print("=" * 60)
print("LLMPipeline.generate")
print("=" * 60)

print(inspect.signature(ov_genai.LLMPipeline.generate))

print("\n")

print("=" * 60)
print("TextStreamer")
print("=" * 60)

print(dir(ov_genai.TextStreamer))

print("\n")

print("=" * 60)
print("StreamerBase")
print("=" * 60)

print(dir(ov_genai.StreamerBase))