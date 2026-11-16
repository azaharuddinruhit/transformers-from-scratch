import numpy as np

rng = np.random.default_rng(0)
d_model, h, n_kv, T = 64, 8, 2, 6
d_k = d_model // h


def rms_norm(x, gain, eps=1e-6):
    # LayerNorm without the mean subtraction and without the shift
    return gain * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(z):
    return z / (1 + np.exp(-z))


def swiglu(x, W1, W3, W2):
    # one projection is passed through SiLU and gates the other, element by element
    return (silu(x @ W1) * (x @ W3)) @ W2


def rope(x, positions, base=10000.0):
    # rotate each pair of dimensions (2f, 2f+1) by the angle position * omega_f
    omega = base ** (-np.arange(0, x.shape[-1], 2) / x.shape[-1])
    angle = positions[:, None] * omega
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * np.cos(angle) - x2 * np.sin(angle)
    out[..., 1::2] = x1 * np.sin(angle) + x2 * np.cos(angle)
    return out


# RoPE: the score depends only on how far apart the query and the key are
q, k = rng.normal(size=d_k), rng.normal(size=d_k)
score = lambda i, j: rope(q[None], np.array([i]))[0] @ rope(k[None], np.array([j]))[0]
print(f"score(10, 7) = {score(10, 7):.4f}, score(103, 100) = {score(103, 100):.4f}")

# Grouped-query attention: 8 query heads share 2 key/value heads
X = rms_norm(rng.normal(size=(T, d_model)), np.ones(d_model))
Wq = rng.normal(size=(d_model, h * d_k)) / 8
Wk, Wv = (rng.normal(size=(d_model, n_kv * d_k)) / 8 for _ in range(2))
pos = np.arange(T)
Q = rope((X @ Wq).reshape(T, h, d_k).transpose(1, 0, 2), pos)       # [h, T, d_k]
K = rope((X @ Wk).reshape(T, n_kv, d_k).transpose(1, 0, 2), pos)    # [n_kv, T, d_k]
V = (X @ Wv).reshape(T, n_kv, d_k).transpose(1, 0, 2)
K, V = np.repeat(K, h // n_kv, axis=0), np.repeat(V, h // n_kv, axis=0)
s = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k) + np.triu(np.full((T, T), -np.inf), 1)
a = np.exp(s - s.max(-1, keepdims=True))
out = (a / a.sum(-1, keepdims=True)) @ V
print("queries", Q.shape, "cached keys per layer", (n_kv, T, d_k), "output", out.shape)

d_ff = 3 * d_model
W1, W3 = (rng.normal(size=(d_model, d_ff)) / 8 for _ in range(2))
W2 = rng.normal(size=(d_ff, d_model)) / 8
print("SwiGLU output", swiglu(X, W1, W3, W2).shape)

# KV cache for the Llama 3.1 8B shape: 32 layers, 32 query heads, 8 KV heads of width 128
for name, heads in (("multi-head", 32), ("grouped-query", 8), ("multi-query", 1)):
    per_token = 2 * 32 * heads * 128 * 2                                # keys and values, 2 bytes
    print(f"{name:13}: {per_token / 1024:4.0f} KiB per token, "
          f"{per_token * 131072 / 1e9:5.1f} GB for 131,072 tokens")

# Mixture of experts: a router picks 2 of 8 experts for each token
router = rng.normal(size=(d_model, 8)) / 8
logits = X @ router
top = np.argsort(-logits, axis=1)[:, :2]
gates = np.exp(np.take_along_axis(logits, top, 1))
gates /= gates.sum(1, keepdims=True)
print("experts chosen for each token:", top.tolist())
print("gates for the first token:", gates[0].round(3))
