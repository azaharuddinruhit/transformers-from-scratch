import numpy as np


def softmax(S, axis=-1):
    e = np.exp(S - S.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


rng = np.random.default_rng(0)
B, T, d_model, h = 2, 5, 8, 2
d_k = d_model // h                       # 4 dimensions per head

X = rng.normal(size=(B, T, d_model))
W_Q, W_K, W_V, W_O = (rng.normal(size=(d_model, d_model)) / np.sqrt(d_model)
                      for _ in range(4))


def split_heads(M):                      # [B, T, d_model] -> [B, h, T, d_k]
    return M.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)


Q, K, V = split_heads(X @ W_Q), split_heads(X @ W_K), split_heads(X @ W_V)
S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)          # [B, h, T, T]
S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)
heads = softmax(S) @ V                                  # [B, h, T, d_k]
concat = heads.transpose(0, 2, 1, 3).reshape(B, T, d_model)
out = concat @ W_O                                      # [B, T, d_model]

for name, arr in [("X", X), ("Q per head", Q), ("scores", S),
                  ("heads", heads), ("concat", concat), ("output", out)]:
    print(f"{name:11} {arr.shape}")

# One big matrix, sliced, is the same as separate per-head matrices
m = 1                                                   # check head 2
cols = slice(m * d_k, (m + 1) * d_k)
q, k, v = X @ W_Q[:, cols], X @ W_K[:, cols], X @ W_V[:, cols]
s = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)
s = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, s)
print("head 2 matches:", np.allclose(softmax(s) @ v, heads[:, m]))

# Same parameter count whatever h is
print("1 head :", 3 * d_model * d_model, "weights in W_Q, W_K, W_V")
print("2 heads:", h * 3 * d_model * d_k, "weights in W_Q, W_K, W_V")
