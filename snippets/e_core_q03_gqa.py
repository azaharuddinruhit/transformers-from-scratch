import numpy as np


def softmax(S, axis=-1):
    e = np.exp(S - S.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


rng = np.random.default_rng(0)
B, T, d_k = 1, 6, 16
n_q_heads, n_kv_heads = 8, 2            # GQA: 4 query heads share each KV head
group = n_q_heads // n_kv_heads

Q = rng.normal(size=(B, n_q_heads, T, d_k))
K = rng.normal(size=(B, n_kv_heads, T, d_k))   # only these are cached
V = rng.normal(size=(B, n_kv_heads, T, d_k))

# Repeat each KV head so it lines up with its group of query heads
K_full = np.repeat(K, group, axis=1)            # [B, 8, T, d_k]
V_full = np.repeat(V, group, axis=1)
S = Q @ K_full.transpose(0, 1, 3, 2) / np.sqrt(d_k)
S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)
out = softmax(S) @ V_full
print("Q", Q.shape, " cached K", K.shape, " output", out.shape)


def kv_cache_bytes(L, T, kv_heads, d_k, bytes_per_number=2):
    return 2 * L * T * kv_heads * d_k * bytes_per_number


L, T_ctx, heads, d_k = 32, 8192, 32, 128
for name, kv in [("MHA", heads), ("GQA (8 KV heads)", 8), ("MQA", 1)]:
    gb = kv_cache_bytes(L, T_ctx, kv, d_k) / 1e9
    print(f"{name:17} KV cache per request: {gb:.2f} GB")
