from collections import Counter

# Word counts from the text in this section: "low" 5 times, "lower" twice, and so on
words = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
corpus = {tuple(w): n for w, n in words.items()}   # every word starts as single characters


def count_pairs(corpus):
    pairs = Counter()
    for pieces, n in corpus.items():
        for a, b in zip(pieces, pieces[1:]):
            pairs[a, b] += n
    return pairs


def merge(pieces, pair):
    out, i = [], 0
    while i < len(pieces):
        if pieces[i:i + 2] == pair:
            out.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return tuple(out)


merges = []
for step in range(6):
    pairs = count_pairs(corpus)
    best = max(pairs, key=pairs.get)                # ties go to the pair seen first
    merges.append(best)
    corpus = {merge(p, best): n for p, n in corpus.items()}
    print(f"merge {step + 1}: {' + '.join(best):8} ({pairs[best]} times) ->",
          "  ".join(" ".join(p) for p in corpus))


def encode(word):
    """Apply the learned merges, in the order they were learned."""
    pieces = tuple(word)
    for pair in merges:
        pieces = merge(pieces, pair)
    return list(pieces)


for w in ["lowest", "newer", "wider"]:
    print(f"{w:7} -> {encode(w)}")
