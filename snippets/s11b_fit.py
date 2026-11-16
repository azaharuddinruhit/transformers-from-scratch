import math

# Vendor figures: memory (bytes), bandwidth (bytes/s), dense BF16 tensor FLOP/s
GPUS = {
    "RTX 4090": (24e9, 1008e9, 165.2e12),
    "L4": (24e9, 300e9, 121e12),
    "A100 80GB": (80e9, 2039e9, 312e12),
    "H100 SXM": (80e9, 3350e9, 989.5e12),
    "DGX Spark": (128e9, 273e9, None),     # unified memory; no BF16 figure published
}
BYTES = {"fp32": 4, "bf16": 2, "int8": 1, "int4": 0.5}


def memory_needed(N, fmt, L, d_model, T, B=1, allowance=1.2):
    weights = N * BYTES[fmt]
    kv = 2 * L * T * d_model * 2 * B       # keys and values, 16-bit, full width
    return weights, kv, allowance * (weights + kv)


def report(name, N, fmt, L, d_model, T):
    w, kv, total = memory_needed(N, fmt, L, d_model, T)
    print(f"{name} in {fmt}, T={T}: weights {w / 1e9:.1f} GB + KV cache "
          f"{kv / 1e9:.2f} GB, x1.2 -> {total / 1e9:.1f} GB")
    for gpu, (mem, bw, _) in GPUS.items():
        n = math.ceil(total / mem)           # GPUs needed, split evenly
        speed = n * bw / (w + kv)            # every step reads weights and cache
        print(f"   {gpu:9} x{n}: at most {speed:4.0f} tokens/s for one request")


report("7B", 7e9, "bf16", 32, 4096, 4096)
report("70B", 70e9, "bf16", 80, 8192, 4096)
report("70B", 70e9, "int4", 80, 8192, 4096)

# Decode reads 2 bytes per bf16 weight and does 2 FLOPs with it per request,
# so it only becomes compute-bound once the batch passes F / R
for gpu, (_, bw, flops) in GPUS.items():
    if flops:
        print(f"{gpu:9}: compute-bound beyond a batch of about {flops / bw:.0f}")

# Training with Adam in mixed precision: 2 (bf16 weights) + 2 (gradients)
# + 4 (fp32 master weights) + 8 (two Adam moments) = 16 bytes per parameter
print(f"training a 7B model: {16 * 7e9 / 1e9:.0f} GB before activations")
