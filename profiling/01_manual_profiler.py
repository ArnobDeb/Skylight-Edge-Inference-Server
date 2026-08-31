from pathlib import Path
import time
import openvino_genai as ov_genai

MODEL_PATH = Path("../models/phi-3.5-mini-int4")

#PROMPT = input();

print("=" * 70)
print("OpenVINO GenAI Manual Profiler")
print("=" * 70)

###############################################################
# Measure Pipeline Initialization
###############################################################

t0 = time.perf_counter()

pipe = ov_genai.LLMPipeline(
    str(MODEL_PATH),
    device="CPU"
)

t1 = time.perf_counter()

###############################################################
# Measure Generation
###############################################################
PROMPT1 = input("Prompt 1: ")
PROMPT2 = input("Prompt 2: ") 
PROMPT3 = input("Prompt 3: ") 

t2 = time.perf_counter()

response1 = pipe.generate(
    PROMPT1,
    max_new_tokens=50
)

t3 = time.perf_counter()

response2 = pipe.generate(
    PROMPT2,
    max_new_tokens=50
)

t4 = time.perf_counter()

response3 = pipe.generate(
    PROMPT3,
    max_new_tokens=50
)

t5 = time.perf_counter()

###############################################################
# Statistics
###############################################################

init_time = t1 - t0
generation_time1 = t3 - t2
generation_time2 = t4 - t3
generation_time3 = t5 - t4
total_time = t5 - t2 + init_time

word_count1 = len(response1.split())
char_count1 = len(response1)

word_count2 = len(response2.split())
char_count2 = len(response2)

word_count3 = len(response3.split())
char_count3 = len(response3)

total_word_count = word_count1 + word_count2 + word_count3
total_char_count = char_count1 + char_count2 + char_count3


print("\n")
print("=" * 70)
print("Prompt1")
print("=" * 70)
print(PROMPT1)

print("\n")
print("=" * 70)
print("Response1")
print("=" * 70)
print(response1)

print("\n")
print("=" * 70)
print("Prompt2")
print("=" * 70)
print(PROMPT2)

print("\n")
print("=" * 70)
print("Response2")
print("=" * 70)
print(response2)

print("\n")
print("=" * 70)
print("Prompt3")
print("=" * 70)
print(PROMPT3)

print("\n")
print("=" * 70)
print("Response3")
print("=" * 70)
print(response3)


print("\n")
print("=" * 70)
print("Profiling Results")
print("=" * 70)

print(f"Pipeline initialization : {init_time:.3f} s")
print(f"Generation time1        : {generation_time1:.3f} s")
print(f"Generation time2        : {generation_time2:.3f} s")
print(f"Generation time3        : {generation_time3:.3f} s")
print(f"Total runtime           : {total_time:.3f} s")

print()

print(f"Characters generated1    : {char_count1}")
print(f"Words generated1         : {word_count1}")

print(f"Characters generated2    : {char_count2}")
print(f"Words generated2         : {word_count2}")

print(f"Characters generated3    : {char_count3}")
print(f"Words generated3         : {word_count3}")

print(f"Characters generatedT    : {total_char_count}")
print(f"Words generatedT         : {total_word_count}")


if generation_time1 > 0:
    print(f"Words/sec               : {word_count1 / generation_time1:.2f}")
if generation_time2 > 0:
    print(f"Words/sec               : {word_count2 / generation_time2:.2f}")
if generation_time3 > 0:
    print(f"Words/sec               : {word_count3 / generation_time3:.2f}")
if generation_time1 + generation_time2 + generation_time3 > 0:
    print(f"Words/sec               : {total_word_count / (generation_time1+generation_time2+generation_time3):.2f}")
