import numpy as np

rng = np.random.default_rng(0)
T, d_k, tile = 1024, 64, 128
Q, K, V = (rng.normal(size=(T, d_k)) for _ in range(3))


def plain_attention(Q, K, V):
    S = Q @ K.T / np.sqrt(d_k)                        # the full T x T grid of scores
    S[np.triu_indices(T, 1)] = -np.inf                # causal mask
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V, S.size


def tiled_attention(Q, K, V):
    out, largest = np.zeros_like(Q), 0
    for q0 in range(0, T, tile):                      # one tile of queries at a time
        q = Q[q0:q0 + tile]
        m = np.full(len(q), -np.inf)                  # running largest score
        r = np.zeros(len(q))                          # running sum of exponentials
        o = np.zeros((len(q), V.shape[1]))            # running weighted sum of values
        for k0 in range(0, q0 + tile, tile):          # tiles wholly above the diagonal are skipped
            s = q @ K[k0:k0 + tile].T / np.sqrt(d_k)  # one tile of scores, never stored
            rows, cols = np.arange(q0, q0 + len(q))[:, None], np.arange(k0, k0 + tile)[None, :]
            s[cols > rows] = -np.inf
            largest = max(largest, s.size)
            m_new = np.maximum(m, s.max(axis=1))
            scale = np.exp(m - m_new)                 # rescale the old totals to the new maximum
            e = np.exp(s - m_new[:, None])
            r = r * scale + e.sum(axis=1)
            o = o * scale[:, None] + e @ V[k0:k0 + tile]
            m = m_new
        out[q0:q0 + tile] = o / r[:, None]
    return out, largest


a, grid = plain_attention(Q, K, V)
b, held = tiled_attention(Q, K, V)
print("same output as plain      :", bool(np.abs(a - b).max() < 1e-12))
print("scores stored, plain      :", f"{grid:,}")
print("scores held at once, tiled:", f"{held:,}")

# The worked example: scores [1, 2, 3, 0], values [10, 20, 30, 40], tiles of two
s, v = np.array([1.0, 2, 3, 0]), np.array([10.0, 20, 30, 40])
w = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
print("softmax weights           :", w.round(3), " output", round(float(w @ v), 2))
