import numpy as np


def softmax(S, axis=-1):
    e = np.exp(S - S.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, causal=False):
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)                      # [T, T] scores
    if causal:
        T = S.shape[0]
        future = np.triu(np.ones((T, T), dtype=bool), k=1)  # j > i
        S = np.where(future, -np.inf, S)            # mask BEFORE softmax
    A = softmax(S)                                  # each row sums to 1
    return A @ V, A


Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 2, 0, 0],
              [0, 0, 1, 1], [0, 1, 2, 0]], dtype=float)
K = np.array([[1, 0, 0, 1], [0, 2, 1, 0], [1, 1, 0, -1],
              [0, 0, 2, 1], [-1, 1, 1, 0]], dtype=float)
V = np.array([[0.5, 0, 0, 1], [1, 2, 0, 0], [0, 1, 1, 0],
              [0, 0, 1, 2], [2, 0, 1, 1]], dtype=float)

np.set_printoptions(precision=3, suppress=True)
out, A = attention(Q, K, V)
print("weights, no mask (rows = queries):\n", A)
out, A = attention(Q, K, V, causal=True)
print("weights, causal mask:\n", A)
print("row sums:", A.sum(axis=1))
print("output shape:", out.shape)

# In a real model Q, K and V are projections of the same input X
rng = np.random.default_rng(0)
T, d_model, d_k = 5, 8, 4
X = rng.normal(size=(T, d_model))
W_Q, W_K, W_V = (rng.normal(size=(d_model, d_k)) for _ in range(3))
out, A = attention(X @ W_Q, X @ W_K, X @ W_V, causal=True)
print("X", X.shape, "-> weights", A.shape, "-> output", out.shape)
