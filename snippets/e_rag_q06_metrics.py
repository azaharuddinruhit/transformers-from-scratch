def recall_at_k(retrieved, relevant, k):
    """Share of the relevant chunks that appear in the top k."""
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    """1 / position of the first relevant chunk (0 if none was found)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


# One labelled test query per row: what was retrieved, and what is correct
eval_set = [
    (["c3", "c8", "c1", "c5"], {"c1"}),
    (["c2", "c4", "c9", "c7"], {"c2", "c7"}),
    (["c6", "c5", "c0", "c3"], {"c4"}),
]

for k in (1, 3):
    mean = sum(recall_at_k(r, rel, k) for r, rel in eval_set) / len(eval_set)
    print(f"recall@{k} = {mean:.3f}")
mrr = sum(reciprocal_rank(r, rel) for r, rel in eval_set) / len(eval_set)
print(f"MRR      = {mrr:.3f}")
