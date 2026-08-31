#To verify tokenizer and detokenizer work together inp -> tokenizer -> tokens -> detokenizer -> same inp
from pathlib import Path
import openvino_genai as ov_genai
MODEL_PATH = Path("../models/phi-3.5-mini-int4")

tokenizer = ov_genai.Tokenizer(str(MODEL_PATH))

prompt = input("Prompt: ")

encoded = tokenizer.encode(prompt)

ids = encoded.input_ids.data[0]

print("\n" + "=" * 60)
print("Tokenizer Explorer")
print("=" * 60)

print(f"Prompt            : {prompt}")
print(f"Batch size        : {encoded.input_ids.data.shape[0]}")
print(f"Input token count : {len(ids)}")

print("\nToken IDs:")

for i, token in enumerate(ids):
    print(f"{i:2d}: {token}")

decoded = tokenizer.decode(ids)

print("\nDecoded again:")
print(decoded)

# from pathlib import Path
# import openvino_genai as ov_genai

# MODEL_PATH = Path("../models/phi-3.5-mini-int4")

# tokenizer = ov_genai.Tokenizer(str(MODEL_PATH))

# prompt = input("Prompt: ")

# encoded = tokenizer.encode(prompt)

# print("\n")

# print("=" * 60)
# print("Tokenizer Information")
# print("=" * 60)

# print(f"Shape: {encoded.input_ids.data.shape}")

# print()

# print(encoded.input_ids.data)

# print()

# print(f"Batch size      : {encoded.input_ids.data.shape[0]}")
# print(f"Sequence length : {encoded.input_ids.data.shape[1]}")

# print()

# print(f"Token IDs:")

# for i, token in enumerate(encoded.input_ids.data[0]):
#     print(f"{i:2d} -> {token}")
