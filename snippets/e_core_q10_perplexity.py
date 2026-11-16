import math

p_correct = [0.5, 0.25, 0.9]           # probability given to each true token
loss = -sum(math.log(p) for p in p_correct) / len(p_correct)
print(f"loss {loss:.4f}  ->  perplexity {math.exp(loss):.3f}")

# An untrained model guesses uniformly: loss ln(n_vocab), perplexity n_vocab
n_vocab = 21
print(f"uniform: loss {math.log(n_vocab):.4f}  ->  perplexity "
      f"{math.exp(math.log(n_vocab)):.1f}")
