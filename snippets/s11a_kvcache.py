import numpy as np


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


rng = np.random.default_rng(0)
d_model, P, n = 8, 3, 5               # prompt of 3 tokens, generate 5
W_Q, W_K, W_V = (rng.normal(size=(d_model, d_model)) for _ in range(3))
X = rng.normal(size=(P + n - 1, d_model))  # token vectors that will be processed


def attend(q, K, V):
    return softmax(K @ q / np.sqrt(d_model)) @ V


# Without a cache: every pass recomputes keys and values for all positions
no_cache, work_no_cache = [], 0
for t in range(P - 1, P + n - 1):     # one pass per generated token
    seen = X[:t + 1]
    K, V = seen @ W_K, seen @ W_V     # recomputed every time
    work_no_cache += len(seen)
    no_cache.append(attend(X[t] @ W_Q, K, V))

# With a cache: prefill the prompt once, then add one row per step
K_cache, V_cache = X[:P] @ W_K, X[:P] @ W_V
work_cache = P
with_cache = [attend(X[P - 1] @ W_Q, K_cache, V_cache)]
for t in range(P, P + n - 1):
    K_cache = np.vstack([K_cache, X[t] @ W_K])        # append the new key
    V_cache = np.vstack([V_cache, X[t] @ W_V])        # append the new value
    work_cache += 1
    with_cache.append(attend(X[t] @ W_Q, K_cache, V_cache))

print("identical outputs:", np.allclose(no_cache, with_cache))
print("positions computed without cache:", work_no_cache,
      "=", P * n + n * (n - 1) // 2)
print("positions computed with cache:   ", work_cache, "=", P + n - 1)
