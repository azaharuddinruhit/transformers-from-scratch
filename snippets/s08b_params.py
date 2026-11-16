def count_params(n_vocab, d_model, T, L, d_ff, tied=False):
    token_emb = n_vocab * d_model
    pos_emb = T * d_model
    attention = 4 * (d_model * d_model + d_model)          # W_Q, W_K, W_V, W_O
    mlp = d_model * d_ff + d_ff + d_ff * d_model + d_model  # W1, b1, W2, b2
    block_norms = 2 * 2 * d_model                           # two LayerNorms
    final_norm = 2 * d_model
    output = 0 if tied else d_model * n_vocab
    total = token_emb + pos_emb + L * (attention + mlp + block_norms) \
        + final_norm + output
    return {"token emb": token_emb, "position emb": pos_emb,
            "blocks": L * (attention + mlp + block_norms),
            "final norm": final_norm, "output": output, "total": total}


print(count_params(n_vocab=50, d_model=8, T=16, L=1, d_ff=32))

gpt2_small = count_params(n_vocab=50257, d_model=768, T=1024, L=12,
                          d_ff=3072, tied=True)
print(f"GPT-2 small: {gpt2_small['total']:,} parameters")

# Doubling the width roughly quadruples the cost of the blocks
for d in (768, 1536):
    blocks = count_params(50257, d, 1024, 12, 4 * d, tied=True)["blocks"]
    print(f"d_model={d}: blocks hold {blocks:,} parameters")
