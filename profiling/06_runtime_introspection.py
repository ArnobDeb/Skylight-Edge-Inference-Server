from pathlib import Path
import openvino as ov

MODEL_PATH = Path("../models/phi-3.5-mini-int4/openvino_model.xml")

core = ov.Core()

print("=" * 70)
print("Available Devices")
print("=" * 70)

for device in core.available_devices:
    print(device)

print()

print("=" * 70)
print("Reading Model")
print("=" * 70)

model = core.read_model(str(MODEL_PATH))

print(f"Number of operations : {len(model.get_ops())}")

print()

print("=" * 70)
print("Model Inputs")
print("=" * 70)

for inp in model.inputs:
    print(f"Name : {inp.get_any_name()}")
    print(f"Shape: {inp.partial_shape}")
    print(f"Type : {inp.element_type}")
    print()

print("=" * 70)
print("Model Outputs")
print("=" * 70)

for out in model.outputs:
    print(f"Name : {out.get_any_name()}")
    print(f"Shape: {out.partial_shape}")
    print(f"Type : {out.element_type}")
    print()
