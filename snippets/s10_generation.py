from collections import Counter
import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def next_token_probs(logits, tau=1.0, top_k=None, top_p=None):
    p = softmax(np.asarray(logits, dtype=float) / tau)    # temperature
    order = np.argsort(-p)                                # most probable first
    keep = np.ones(p.size, dtype=bool)
    if top_k is not None:
        keep[order[top_k:]] = False                       # top-k
    if top_p is not None:
        cum = np.cumsum(p[order])
        n = np.searchsorted(cum, top_p) + 1               # smallest set >= p
        keep[order[n:]] = False                           # top-p
    p = np.where(keep, p, 0.0)
    return p / p.sum()                                    # renormalize


z = [2.0, 1.0, 0.1]
for tau in (1, 0.5, 2):
    print(f"tau={tau:<3}", next_token_probs(z, tau).round(4))

probs = np.array([0.5, 0.3, 0.15, 0.05])
logits = np.log(probs)              # logits whose softmax is exactly `probs`
print("top-k=2  ", next_token_probs(logits, top_k=2).round(4))
print("top-p=0.9", next_token_probs(logits, top_p=0.9).round(4))

# The generation loop. The "model" here is a table of character counts that
# looks at the last two characters. It stands in for a trained Transformer.
text = "the cat sat on the mat. the dog sat on the sofa. a cat saw the dog. "
chars = sorted(set(text))
counts = Counter((text[i:i + 2], text[i + 2]) for i in range(len(text) - 2))
model = lambda context: np.log(                     # logits for the next char
    np.array([counts[(context[-2:], c)] for c in chars]) + 0.01)


def generate(prompt, n, rng, greedy=False, **settings):
    out = prompt
    for _ in range(n):
        p = next_token_probs(model(out), **settings)
        i = p.argmax() if greedy else rng.choice(len(p), p=p)
        out += chars[i]                                   # append, repeat
    return out


rng = np.random.default_rng(0)
print("greedy:   ", repr(generate("the ", 30, rng, greedy=True)))
print("tau=0.7:  ", repr(generate("the ", 30, rng, tau=0.7, top_p=0.9)))
print("tau=2.0:  ", repr(generate("the ", 30, rng, tau=2.0)))
