import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())  # subtracting the max changes nothing but avoids overflow
    return e / e.sum()


z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
print("p              =", p.round(4))
print("sum            =", round(p.sum(), 6))
print("p[0] / p[1]    =", round(p[0] / p[1], 4))  # a gap of 1 gives a ratio of e
print("softmax(z + 5) =", softmax(z + 5).round(4))  # only the gaps matter

# Why subtract the max: exp(1000) overflows to infinity
big = np.array([1000.0, 999.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("naive          =", naive)
print("stable         =", softmax(big).round(4))

# Sampling: draw tokens in proportion to their probabilities
rng = np.random.default_rng(0)
tokens = np.array(["mat", "sofa", "floor"])
draws = rng.choice(tokens, size=1000, p=p)
print("1000 draws     =", {str(t): int((draws == t).sum()) for t in tokens})
