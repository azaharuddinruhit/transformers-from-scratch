import torch
import torch.nn.functional as F

torch.manual_seed(0)

# A character "tokenizer" keeps the example self-contained
prompt, response = "Q: 2+2? A:", " 4"
chars = sorted(set(prompt + response))
ids = torch.tensor([chars.index(c) for c in prompt + response])

# Labels: copy the inputs, then hide the prompt with -100 (ignored by the loss)
labels = ids.clone()
labels[:len(prompt)] = -100

# Random logits stand in for a model's output: [T, n_vocab]
logits = torch.randn(len(ids), len(chars))

# Position t predicts token t+1, so shift logits and labels by one
loss = F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)
full = F.cross_entropy(logits[:-1], ids[1:])

counted = (labels[1:] != -100).sum().item()
print(f"positions in the loss: {counted} of {len(ids) - 1} (response only)")
print(f"masked loss {loss.item():.4f}   unmasked loss {full.item():.4f}")

# Check: the masked loss is the mean over response positions only
manual = -F.log_softmax(logits[:-1], dim=-1)[torch.arange(len(ids) - 1), ids[1:]]
print("matches manual mean:", torch.isclose(loss, manual[len(prompt) - 1:].mean()).item())
