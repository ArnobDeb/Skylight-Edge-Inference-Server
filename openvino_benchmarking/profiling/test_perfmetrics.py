from pathlib import Path
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline(
    "../models/phi-3.5-mini-int4",
    "CPU"
)

result = pipe.generate(
   ["What is OpenVINO?"],
    max_new_tokens=20,
    return_decoded_results=True
)

print(type(result))
print(dir(result))
#metrics = result.perf_metrics

#print(metrics.get_ttft().mean)
#print(metrics.get_tpot().mean)
#print(metrics.get_throughput().mean)
#print(metrics.get_generate_duration().mean)
#print(metrics.get_num_generated_tokens())
#print(metrics.get_num_input_tokens())

