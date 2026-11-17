import numpy as np


def paper_lr(step, d_model=512, warmup=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


for step in (1, 4000, 100_000):
    print(f"step {step:>7,}: lr = {paper_lr(step):.4g}")
print("peak = (d_model * warmup) ** -0.5 =", round((512 * 4000) ** -0.5, 7))


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


def post_norm(x, sublayer):          # 2017 paper: normalize after the add
    return layer_norm(x + sublayer(x))


def pre_norm(x, sublayer):           # GPT-style: normalize a copy first
    return x + sublayer(layer_norm(x))


rng = np.random.default_rng(0)
W = rng.normal(0, 0.1, size=(8, 8))
sublayer = lambda x: np.tanh(x @ W)
x = rng.normal(3, 2, size=(4, 8))

print("post-norm stream: mean", post_norm(x, sublayer).mean().round(3),
      "std", post_norm(x, sublayer).std().round(3), "(rescaled every layer)")
print("pre-norm stream:  mean", pre_norm(x, sublayer).mean().round(3),
      "std", pre_norm(x, sublayer).std().round(3), "(the input passes through)")
