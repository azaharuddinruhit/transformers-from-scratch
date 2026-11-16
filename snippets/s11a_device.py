import torch

# Pick the best available device and fall back to the CPU
if torch.cuda.is_available():              # NVIDIA, and AMD ROCm builds of PyTorch
    device = "cuda"
elif torch.backends.mps.is_available():    # Apple silicon
    device = "mps"
else:
    device = "cpu"

print("PyTorch:", torch.__version__)
print("built for CUDA:", torch.version.cuda)   # None means a CPU-only build
print("device:", device)
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

# Bytes per number, largest value and precision of each floating-point format
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dtype)
    print(f"{str(dtype):15} {fi.bits // 8} bytes  max {fi.max:9.3g}  gap after 1.0: {fi.eps:.3g}")

# float16 overflows where bfloat16 does not; bfloat16 rounds more coarsely
x = torch.tensor([70000.0, 1.001])
print("float16: ", x.to(torch.float16).tolist())
print("bfloat16:", x.to(torch.bfloat16).tolist())

# Weights move to the device once; inputs must be on the same device
torch.manual_seed(0)
layer = torch.nn.Linear(4096, 4096).to(device)
x = torch.randn(8, 4096, device=device)
with torch.autocast(device_type=device, dtype=torch.bfloat16):   # mixed precision
    y = layer(x)
print("weights stored as:", layer.weight.dtype, "| matmul output:", y.dtype, "on", y.device.type)
