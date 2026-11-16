import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Training text: sentences like the toy model's, read character by character
subjects = ["the cat", "the dog", "a bird", "the fox"]
verbs = ["sat on", "saw", "ran to", "likes"]
objects = ["the mat", "a box", "the tree", "the sofa"]
text = " ".join(f"{s} {v} {o}." for s in subjects for v in verbs for o in objects)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n_vocab, T, B = len(chars), 16, 32


class TinyGPT(nn.Module):
    def __init__(self, d_model=48, n_head=2, n_layer=2):
        super().__init__()
        self.tok = nn.Embedding(n_vocab, d_model)
        self.pos = nn.Embedding(T, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, 4 * d_model, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)          # pre-norm block
        self.blocks = nn.TransformerEncoder(layer, n_layer,
                                            enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_vocab)        # W_U

    def forward(self, idx):                            # idx: [B, T]
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t))
        mask = nn.Transformer.generate_square_subsequent_mask(t)
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(self.ln(x))                   # logits: [B, T, n_vocab]


def get_batch():
    starts = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([data[s:s + T] for s in starts])
    y = torch.stack([data[s + 1:s + T + 1] for s in starts])  # shifted by one
    return x, y


model = TinyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters,",
      f"uniform baseline ln {n_vocab} = {math.log(n_vocab):.3f}")

for step in range(601):
    x, y = get_batch()
    logits = model(x)                                         # 1. forward
    loss = F.cross_entropy(logits.reshape(-1, n_vocab), y.reshape(-1))  # 2. loss
    opt.zero_grad()
    loss.backward()                                           # 3. backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # cap the gradient
    opt.step()                                                # 4. update
    if step % 150 == 0:
        print(f"step {step:3}  loss {loss.item():.3f}")

# Greedy generation from the trained model
idx = torch.tensor([[stoi[c] for c in "the "]])
with torch.no_grad():
    for _ in range(40):
        next_id = model(idx[:, -T:])[0, -1].argmax()
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
print("sample:", "".join(chars[i] for i in idx[0]))
