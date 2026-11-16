import numpy as np


def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)          # 1. mean
    var = x.var(axis=-1, keepdims=True)          # 2. variance
    x_hat = (x - mu) / np.sqrt(var + eps)        # 3. normalise
    return gamma * x_hat + beta                  # 4. learned scale and shift


x = np.array([2.0, 4.0, 6.0, 8.0])
print("mean, variance =", x.mean(), x.var())
print("LayerNorm(x)   =", layer_norm(x, eps=0).round(4))

# Scale and offset disappear: only the pattern survives
print("x * 100        =", layer_norm(x * 100, eps=0).round(4))
print("x + 7          =", layer_norm(x + 7, eps=0).round(4))

# Applied to every token's vector independently: shape [T, d_model]
rng = np.random.default_rng(0)
X = rng.normal(5, 3, size=(3, 8))
Y = layer_norm(X)
print("per-token means:", Y.mean(axis=-1).round(6))
print("per-token stds: ", Y.std(axis=-1).round(4))
