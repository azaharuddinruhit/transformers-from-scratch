import numpy as np


def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(x.var(axis=-1, keepdims=True) + eps) + beta


def rms_norm(x, gamma=1.0, eps=1e-5):
    # No mean subtraction and no beta: divide by the root mean square
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return gamma * x / rms


x = np.array([2.0, 4.0, 6.0, 8.0])
print("LayerNorm:", layer_norm(x).round(4))
print("RMSNorm:  ", rms_norm(x).round(4))

# On a vector that already has mean 0 the two agree
x0 = x - x.mean()
print("same when the mean is 0:", np.allclose(layer_norm(x0), rms_norm(x0)))
