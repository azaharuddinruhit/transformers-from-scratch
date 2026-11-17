import numpy as np

rng = np.random.default_rng(0)

# Training data: 300 points on a spiral. Each "image" here is just two numbers.
t = 1.5 + np.sqrt(rng.random(300)) * 2.6 * np.pi
data = np.stack([np.cos(t), np.sin(t)], axis=1) * (t / (1.5 + 2.6 * np.pi) * 1.3)[:, None]
data += rng.normal(0, 0.025, data.shape)


def alpha_bar(level):
    """Share of the clean example kept: 1 at level 0 (clean), near 0 at level 1 (pure noise)."""
    return np.maximum(1e-6, np.cos(np.pi / 2 * np.minimum(level, 0.9999)) ** 2)


# Training examples: mix clean points with noise (the network would learn to predict eps)
u0 = data[:4]
eps = rng.normal(size=u0.shape)
a = alpha_bar(0.5)
u_noisy = np.sqrt(a) * u0 + np.sqrt(1 - a) * eps
print("clean      ", u0.round(2).tolist())
print("level 0.5  ", u_noisy.round(2).tolist())


def guess_clean(u, a):
    """The best guess at u0 for these 300 points: a weighted average of them.
    A real model learns to approximate this with a network."""
    d2 = ((u[:, None, :] - np.sqrt(a) * data[None]) ** 2).sum(-1)
    w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * (1 - a)))
    return (w / w.sum(1, keepdims=True)) @ data


def generate(steps, n=300):
    u = rng.normal(size=(n, 2))                       # start from pure noise
    for k in range(steps, 0, -1):
        a, a_next = alpha_bar(k / steps), (alpha_bar((k - 1) / steps) if k > 1 else 1.0)
        u0_hat = guess_clean(u, a)
        eps_hat = (u - np.sqrt(a) * u0_hat) / np.sqrt(1 - a)
        sigma = np.sqrt((1 - a_next) / (1 - a) * (1 - a / a_next))
        u = np.sqrt(a_next) * u0_hat + np.sqrt(max(0.0, 1 - a_next - sigma**2)) * eps_hat \
            + sigma * rng.normal(size=u.shape)        # move towards the guess, add a little fresh noise
    return u


for steps in (1, 3, 10, 50):
    u = generate(steps)
    dist = np.sqrt(((u[:, None] - data[None]) ** 2).sum(-1)).min(1)
    print(f"{steps:2} step{'s' if steps > 1 else ' '}: spread {u.std(0).mean():.2f}, "
          f"on the spiral {np.mean(dist < 0.03):4.0%}")
