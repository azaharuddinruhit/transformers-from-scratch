import numpy as np

rng = np.random.default_rng(0)
vocab = ["mat", "rug", "chair", "bed", "floor"]
p = np.array([0.60, 0.20, 0.10, 0.06, 0.04])   # the big model's next-token probabilities
q = np.array([0.40, 0.35, 0.05, 0.15, 0.05])   # the draft model's
leftover = np.maximum(p - q, 0) / np.maximum(p - q, 0).sum()


def speculate(n_samples):
    """The draft samples from q; keep its token with probability min(1, p/q), else sample from the leftover of p."""
    x = rng.choice(len(q), size=n_samples, p=q)
    accepted = rng.random(n_samples) < np.minimum(1, p[x] / q[x])
    return np.where(accepted, x, rng.choice(len(p), size=n_samples, p=leftover)), accepted


# 1. The tokens that come out follow p, the big model, not q
out, accepted = speculate(1_000_000)
for i, word in enumerate(vocab):
    print(f"{word:6} big model {p[i]:.2f}  draft {q[i]:.2f}  speculative output {np.mean(out == i):.3f}")

# 2. Tokens per big-model pass when n tokens are drafted (here every position has the same p and q)
a = np.minimum(p, q).sum()                       # the chance that one draft token is accepted
print(f"\nacceptance rate: measured {accepted.mean():.3f}, sum of min(p, q) = {a:.2f}")
for n in [1, 2, 4, 8]:
    kept = speculate(200_000 * n)[1].reshape(-1, n)
    run = np.cumprod(kept, axis=1).sum(axis=1)   # accepted until the first rejection
    print(f"draft {n}: {np.mean(run + 1):.2f} tokens per pass, formula {(1 - a ** (n + 1)) / (1 - a):.2f}")
