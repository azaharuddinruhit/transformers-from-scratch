import math
import numpy as np


def softmax(S, axis=-1):
    e = np.exp(S - S.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


def gelu(a):
    return a * 0.5 * (1 + np.vectorize(math.erf)(a / math.sqrt(2)))


def multi_head(x, p, h):
    T, d_model = x.shape
    d_k = d_model // h
    split = lambda M: M.reshape(T, h, d_k).transpose(1, 0, 2)  # [h, T, d_k]
    Q, K, V = split(x @ p["W_Q"]), split(x @ p["W_K"]), split(x @ p["W_V"])
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)
    heads = softmax(S) @ V
    return heads.transpose(1, 0, 2).reshape(T, d_model) @ p["W_O"]


def ffn(x, p):
    return gelu(x @ p["W1"]) @ p["W2"]


def block(x, p, h=2):
    x = x + multi_head(layer_norm(x), p, h)   # x' = x + MultiHead(LN(x))
    x = x + ffn(layer_norm(x), p)             # y  = x' + FFN(LN(x'))
    return x


rng = np.random.default_rng(0)
T, d_model, d_ff, L = 5, 8, 32, 4


def init(shape):
    return rng.normal(0, 0.02, size=shape)


layers = [{"W_Q": init((d_model, d_model)), "W_K": init((d_model, d_model)),
           "W_V": init((d_model, d_model)), "W_O": init((d_model, d_model)),
           "W1": init((d_model, d_ff)), "W2": init((d_ff, d_model))}
          for _ in range(L)]

x = rng.normal(size=(T, d_model))
print("input  shape:", x.shape)
for p in layers:
    x = block(x, p)
print("output shape:", x.shape, f"after {L} blocks: the shape never changes")
