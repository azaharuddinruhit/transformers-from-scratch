import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# Teacher forcing: the targets are the inputs shifted by one place
tokens = "the cat sat on the mat".split()
inputs, targets = tokens[:-1], tokens[1:]
for i in range(len(inputs)):
    print(f"given {' '.join(inputs[:i + 1])!r:26} predict {targets[i]!r}")

# Cross-entropy: the average of -log p(correct next token)
p_correct = np.array([0.5, 0.25, 0.9])
per_token = -np.log(p_correct)
print("per-token losses:", per_token.round(4))
print("mean loss       :", round(per_token.mean(), 4))
print("uniform baseline, 21 symbols: ln 21 =", round(np.log(21), 4))

# The gradient of the loss with respect to the logits is p - onehot(target)
rng = np.random.default_rng(0)
z, y = rng.normal(size=5), 2
loss = lambda z: -np.log(softmax(z)[y])

analytic = softmax(z)
analytic[y] -= 1
h = 1e-6
numeric = np.array([(loss(z + h * np.eye(5)[i]) - loss(z - h * np.eye(5)[i]))
                    / (2 * h) for i in range(5)])
print("p - onehot      :", analytic.round(4))
print("finite diffs    :", numeric.round(4))

# One gradient-descent step on a single weight
theta, grad, eta = 2.0, 0.5, 0.1
print("theta after one step:", theta - eta * grad)
