import openvino as ov

core = ov.Core()

print("=" * 70)
print("CompiledModel methods")
print("=" * 70)

model = core.read_model("../models/phi-3.5-mini-int4/openvino_model.xml")
compiled = core.compile_model(model, "CPU")

print(dir(compiled))

print("\n")

print("=" * 70)
print("InferRequest methods")
print("=" * 70)

request = compiled.create_infer_request()

print(dir(request))
