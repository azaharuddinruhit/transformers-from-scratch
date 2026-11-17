import numpy as np

tokens = ["the", "cat", "sat", "on", "mat"]
K = np.array([[1, 0, 0, 1], [0, 2, 1, 0], [1, 1, 0, -1],
              [0, 0, 2, 1], [-1, 1, 1, 0]], dtype=float)
V = np.array([[0.5, 0, 0, 1], [1, 2, 0, 0], [0, 1, 1, 0],
              [0, 0, 1, 2], [2, 0, 1, 1]], dtype=float)
q = np.array([1, 2, 0, 0], dtype=float)  # the query of "sat"
d_k = q.size

scores = K @ q                      # 1. one dot product per key
scaled = scores / np.sqrt(d_k)      # 2. divide by sqrt(d_k)
e = np.exp(scaled)                  # 3. exponentiate ...
alpha = e / e.sum()                 #    ... and normalize: softmax
c = alpha @ V                       # 4. weighted sum of the values

print("dot products  =", scores)
print("scaled        =", scaled)
print("exp           =", e.round(4), " sum =", round(e.sum(), 4))
print("alpha         =", alpha.round(4))
print("context c     =", c.round(4))
print("most attended =", tokens[alpha.argmax()])
