import numpy as np


def softmax(S, axis=-1):
    e = np.exp(S - S.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V):              # the same function as self-attention
    A = softmax(Q @ K.T / np.sqrt(Q.shape[-1]))
    return A @ V, A


source = ["le", "chat", "noir", "dort"]           # encoder side
written = ["<s>", "the", "black"]                 # decoder input so far
Q_dec = np.array([[2, 1, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0]], dtype=float)
K_enc = np.array([[2, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 0, 2]],
                 dtype=float)
V_enc = np.array([[1, 0, 0, 0.5], [0, 2, 0, 1], [0, 0, 2, -1], [0.5, 1, 0, 2]])

# Queries from the decoder; keys and values from the encoder. No causal mask.
context, weights = attention(Q_dec, K_enc, V_enc)
print("weights shape:", weights.shape, "= [T_tgt, T_src], a rectangle")

i = 1                                             # decoder has written "the"
print("after 'the', alpha =", weights[i].round(4))
print("context c         =", context[i].round(4))
print("most attended     =", source[weights[i].argmax()])
for word, row in zip(written, weights):
    print(f"query from {word!r:8} attends most to {source[row.argmax()]!r}")
