import re
import numpy as np

docs = [
    "Reset your password from the account settings page.",
    "Expense claims must be filed within 30 days of purchase.",
    "The VPN client needs to be updated before connecting.",
    "Annual leave requests go to your line manager for approval.",
    "Forgotten password? Use the self-service reset link.",
]
words = lambda text: re.findall(r"[a-z0-9]+", text.lower())
vocab = sorted({w for d in docs for w in words(d)})


def embed(text):
    # Word counts stand in for a real embedding model here; the search code
    # below is the same whichever model produces the vectors.
    v = np.array([words(text).count(w) for w in vocab], dtype=float)
    return v / (np.linalg.norm(v) or 1.0)          # normalize to length 1


doc_vectors = np.stack([embed(d) for d in docs])   # computed once, offline


def search(query, k=2):
    scores = doc_vectors @ embed(query)            # cosine = dot of unit vectors
    best = np.argsort(-scores)[:k]
    return [(round(float(scores[i]), 3), docs[i]) for i in best]


for score, doc in search("how do I reset my password"):
    print(score, doc)
