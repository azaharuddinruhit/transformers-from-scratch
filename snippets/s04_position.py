import numpy as np


def sinusoidal_pe(T, d_model):
    t = np.arange(T)[:, None]                 # positions, shape [T, 1]
    i = np.arange(d_model // 2)[None, :]      # pair index, shape [1, d_model/2]
    angle = t / 10000 ** (2 * i / d_model)    # shape [T, d_model/2]
    pe = np.zeros((T, d_model))
    pe[:, 0::2] = np.sin(angle)               # even dimensions: sine
    pe[:, 1::2] = np.cos(angle)               # odd dimensions: cosine
    return pe


print(sinusoidal_pe(3, 4).round(4))

# Add positions to token vectors: x_t = E[w_t] + PE(t)
rng = np.random.default_rng(0)
vocab = ["the", "cat", "sat", "on", "mat"]
E = rng.normal(0, 0.5, size=(len(vocab), 8))
ids = [0, 1, 2, 3, 0]                         # "the cat sat on the"
X_tokens = E[ids]
X = X_tokens + sinusoidal_pe(len(ids), 8)

print("'the' rows equal without positions:", np.allclose(X_tokens[0], X_tokens[4]))
print("'the' rows equal with positions:   ", np.allclose(X[0], X[4]))
print("shape is unchanged:", X.shape)
