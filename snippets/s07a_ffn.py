import math
import numpy as np


def gelu(a):
    # a * Phi(a), where Phi is the standard normal CDF
    return a * 0.5 * (1 + np.vectorize(math.erf)(a / math.sqrt(2)))


def relu(a):
    return np.maximum(0, a)


def ffn(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2


x = np.array([1.0, 2.0])
W1 = np.array([[1, 0.5, -1], [-1, 2, 1]])
W2 = np.array([[1, 0], [0, 1], [1, -1]])
b1, b2 = np.zeros(3), np.zeros(2)

print("x W1      =", x @ W1)
print("GELU      =", gelu(x @ W1).round(4))
print("FFN(x)    =", ffn(x, W1, b1, W2, b2).round(4))

# Without the non-linearity, two matrices collapse into one
print("linear only = one matrix:", np.allclose((x @ W1) @ W2, x @ (W1 @ W2)))

a = np.array([-3, -1, -0.5, 0, 0.5, 1, 3])
print("a         =", a)
print("GELU(a)   =", gelu(a).round(4))
print("ReLU(a)   =", relu(a))
