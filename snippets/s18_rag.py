import math
import re

handbook = """Passwords. Reset your password from the account settings page. Forgotten password? Use the self-service reset link on the sign-in page.

Expenses. Expense claims must be filed within 30 days of purchase, with receipts. Flights and hotels over 500 need approval from finance before booking.

Remote work. Update the VPN client before connecting from home. Error E-1042 on the VPN client means your certificate has expired: ask the service desk for a new one."""

STOP = set("a an the i my me to do does how what of on in for from your you is are be it and or with have has can must".split())


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if w not in STOP]


def chunk(text, size=12, overlap=4):
    """Chunks of `size` words, each sharing `overlap` words with the one before, and each keeping its section title."""
    chunks = []
    for section in text.split("\n\n"):
        title, body = section.split(". ", 1)
        body = body.split()
        for start in range(0, len(body), size - overlap):
            chunks.append(f"{title}: " + " ".join(body[start:start + size]))
            if start + size >= len(body):
                break
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.2, b=0.75):
        self.docs = [words(c) for c in chunks]
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        self.k1, self.b = k1, b

    def idf(self, w):                            # rare words count more
        n = sum(w in d for d in self.docs)
        return math.log(1 + (len(self.docs) - n + 0.5) / (n + 0.5))

    def score(self, query, d):
        s = 0.0
        for w in set(words(query)):
            f = d.count(w)                       # repeats help, but less and less
            s += self.idf(w) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return s

    def search(self, query, k=2):
        scores = [self.score(query, d) for d in self.docs]
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [i for i in best if scores[i] > 0], scores    # a chunk with none of the words is no match


chunks = chunk(handbook)
print(f"{len(chunks)} chunks, for example:\n  {chunks[0]}\n  {chunks[1]}\n")
index = BM25(chunks)
for question in ["what does error E-1042 mean", "how many days to file expense claims"]:
    best, scores = index.search(question)
    print(question)
    for i in best:
        print(f"  {scores[i]:.2f}  {chunks[i]}")
