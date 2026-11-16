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
    split = lambda M: M.reshape(T, h, d_k).transpose(1, 0, 2)
    Q, K, V = split(x @ p["W_Q"]), split(x @ p["W_K"]), split(x @ p["W_V"])
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)
    return (softmax(S) @ V).transpose(1, 0, 2).reshape(T, d_model) @ p["W_O"]


def block(x, p, h):
    x = x + multi_head(layer_norm(x), p, h)
    return x + gelu(layer_norm(x) @ p["W1"]) @ p["W2"]


def gpt_forward(ids, params, h=2):
    T = len(ids)
    x = params["E"][ids] + params["E_pos"][:T]  # x(0) = E[w_t] + E_pos[t]
    for p in params["blocks"]:                  # x(l) = Block_l(x(l-1))
        x = block(x, p, h)
    z = layer_norm(x) @ params["W_U"]           # z = LayerNorm(x(L)) W_U
    return softmax(z)                           # p = softmax(z)


vocab = ["the", "cat", "sat", "on", "mat", "sofa", "dog", "."]
n_vocab, d_model, d_ff, L, T_max = len(vocab), 8, 32, 2, 16
rng = np.random.default_rng(0)


def init(*shape):
    return rng.normal(0, 0.02, size=shape)


params = {
    "E": init(n_vocab, d_model), "E_pos": init(T_max, d_model),
    "blocks": [{"W_Q": init(d_model, d_model), "W_K": init(d_model, d_model),
                "W_V": init(d_model, d_model), "W_O": init(d_model, d_model),
                "W1": init(d_model, d_ff), "W2": init(d_ff, d_model)}
               for _ in range(L)],
    "W_U": init(d_model, n_vocab),
}

ids = [vocab.index(w) for w in "the cat sat on the".split()]
probs = gpt_forward(ids, params)
print("probabilities shape:", probs.shape, "(one row per position)")
print("each row sums to 1: ", np.allclose(probs.sum(axis=1), 1))
# The weights are random, so this is close to uniform (1/8 = 0.125)
print("next-token distribution after the last position:")
print({w: round(float(p), 3) for w, p in zip(vocab, probs[-1])})
