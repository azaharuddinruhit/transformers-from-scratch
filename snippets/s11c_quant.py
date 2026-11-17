import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, (512, 512))      # a weight matrix, rows are inputs
W[rng.integers(0, 512, 8), rng.integers(0, 512, 8)] = 0.5   # a few large weights, as real models have
x = rng.normal(0, 1, (64, 512))          # 64 input rows
y = x @ W


def quantize(w, bits, group):
    """Symmetric round-to-nearest: one scale per `group` consecutive weights in each column."""
    q_max = 2 ** (bits - 1) - 1                       # 127 for 8 bits, 7 for 4 bits
    g = w.T.reshape(w.shape[1], -1, group)            # [columns, groups, group]
    scale = np.abs(g).max(axis=-1, keepdims=True) / q_max
    q = np.clip(np.round(g / scale), -q_max, q_max)   # the small integers that are stored
    return (q * scale).reshape(w.shape[1], -1).T      # dequantized weights


def rel(a, b):
    return np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())


print(f"{'bits':>4} {'group':>6} {'bits/weight':>12} {'weight error':>13} {'output error':>13}")
for bits, group in [(8, 512), (8, 32), (4, 512), (4, 32), (3, 32)]:
    Wq = quantize(W, bits, group)
    bpw = bits + 16 / group                            # plus one 16-bit scale per group
    print(f"{bits:>4} {group:>6} {bpw:>12.2f} {rel(Wq, W):>13.1%} {rel(x @ Wq, y):>13.1%}")

params = 8.03e9                                       # Llama 3.1 8B
for name, bpw in [("bf16", 16), ("int8", 8 + 16 / 32), ("int4", 4 + 16 / 32)]:
    print(f"8B model in {name:5}: {params * bpw / 8 / 1e9:5.1f} GB of weights")
