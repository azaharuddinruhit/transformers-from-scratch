import numpy as np

# A toy WordPiece-style vocabulary. "##" marks a piece that continues a word.
letters = "abcdefghijklmnopqrstuvwxyz"
vocab = (["[UNK]", "the", "cat", "sat", "on", "mat", "play", "jump", "un",
          "##s", "##ed", "##ing", "##happy", "##ly"]
         + list(letters) + ["##" + c for c in letters])
token_id = {tok: i for i, tok in enumerate(vocab)}


def tokenize_word(word):
    """Greedy longest match from the left, as WordPiece does."""
    pieces, start = [], 0
    while start < len(word):
        for end in range(len(word), start, -1):  # try the longest piece first
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in token_id:
                pieces.append(piece)
                start = end
                break
        else:
            return ["[UNK]"]  # no piece fits, not even one letter
    return pieces


def tokenize(text):
    return [p for word in text.lower().split() for p in tokenize_word(word)]


for text in ["the cats played", "unhappy", "zebra"]:
    toks = tokenize(text)
    print(f"{text!r:18} -> {toks} -> ids {[token_id[t] for t in toks]}")

# Embedding lookup: each ID selects one row of the table E
E = np.array([[0.1, 0.3], [-0.2, 0.5], [0.7, -0.4], [0.0, 0.9], [0.6, 0.6]])
w_t = 2
print("E[2]            =", E[w_t])

one_hot = np.zeros(len(E))
one_hot[w_t] = 1
print("onehot(2) @ E   =", one_hot @ E)  # the same row, by a matrix product

# The model's input: one row per token, shape [T, d_model]
rng = np.random.default_rng(0)
E_big = rng.normal(0, 0.02, size=(len(vocab), 8))  # untrained: random numbers
ids = [token_id[t] for t in tokenize("the cat sat on the mat")]
X = E_big[ids]
print("X.shape         =", X.shape)
print("both 'the' rows equal:", np.array_equal(X[0], X[4]))
