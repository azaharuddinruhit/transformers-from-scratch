import numpy as np


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


rng = np.random.default_rng(0)
T, d_model, d_ff, n_experts, top_k = 6, 8, 16, 4, 2

W_router = rng.normal(size=(d_model, n_experts))
experts = [(rng.normal(0, 0.3, size=(d_model, d_ff)),
            rng.normal(0, 0.3, size=(d_ff, d_model))) for _ in range(n_experts)]


def expert_mlp(x, e):
    W1, W2 = experts[e]
    return np.maximum(0, x @ W1) @ W2


def moe_layer(X):
    logits = X @ W_router                              # [T, n_experts]
    chosen = np.argsort(-logits, axis=1)[:, :top_k]    # top-k experts per token
    out = np.zeros_like(X)
    for t in range(len(X)):
        gates = softmax(logits[t, chosen[t]])          # weights over the chosen
        for g, e in zip(gates, chosen[t]):
            out[t] += g * expert_mlp(X[t], e)          # only k experts run
    return out, chosen


X = rng.normal(size=(T, d_model))
out, chosen = moe_layer(X)
print("output shape:", out.shape)
print("experts chosen per token:", chosen.tolist())
print("tokens per expert (load):", np.bincount(chosen.ravel(), minlength=n_experts))
print(f"each token runs {top_k} of {n_experts} expert MLPs")
