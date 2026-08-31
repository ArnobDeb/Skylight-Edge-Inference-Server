import numpy as np
import openvino as ov

core = ov.Core()

model = core.read_model("../models/phi-3.5-mini-int4/openvino_model.xml")

compiled = core.compile_model(model, "CPU")

request = compiled.create_infer_request()

# Dummy inputs (we only want profiling object structure)
inputs = {}

for inp in compiled.inputs:
    shape = [1 if d.is_dynamic else int(d) for d in inp.partial_shape]

    if inp.element_type.to_string() == "i64":
        inputs[inp] = np.zeros(shape, dtype=np.int64)
    elif inp.element_type.to_string() == "i32":
        inputs[inp] = np.zeros(shape, dtype=np.int32)
    else:
        inputs[inp] = np.zeros(shape, dtype=np.float32)

try:
    request.infer(inputs)
except Exception:
    # Dummy inference will fail because LLM inputs are inconsistent.
    # That's okay—we only want to see whether profiling API exists.
    pass

info = request.get_profiling_info()

print(type(info))

if len(info):
    print(type(info[0]))
    print(dir(info[0]))
else:
    print("Profiling list is empty (expected with dummy inputs).")
