import numpy as np


def rope(x, t, base=10000):
    """Rotate each pair of dimensions (2i, 2i+1) by the angle t * theta_i."""
    d = x.shape[-1]
    theta = base ** (-2 * np.arange(d // 2) / d)   # one speed per pair
    cos, sin = np.cos(t * theta), np.sin(t * theta)
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * cos - x2 * sin
    out[1::2] = x1 * sin + x2 * cos
    return out


rng = np.random.default_rng(0)
q, k = rng.normal(size=8), rng.normal(size=8)

# The score depends only on the offset between the two positions
print("q at 5,  k at 3 :", round(rope(q, 5) @ rope(k, 3), 6))
print("q at 12, k at 10:", round(rope(q, 12) @ rope(k, 10), 6))
print("q at 12, k at 3 :", round(rope(q, 12) @ rope(k, 3), 6))

# A rotation never changes a vector's length
print("length kept:", np.isclose(np.linalg.norm(rope(q, 7)), np.linalg.norm(q)))
