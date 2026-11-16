from collections import defaultdict


def reciprocal_rank_fusion(rankings, k=60):
    """Merge ranked lists: each list adds 1 / (k + rank) to a document."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


vector_hits = ["doc7", "doc2", "doc9", "doc4"]   # semantic search order
keyword_hits = ["doc4", "doc7", "doc1"]          # BM25 order (exact terms)

print(reciprocal_rank_fusion([vector_hits, keyword_hits]))
