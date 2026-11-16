import torch
import torch.nn as nn

torch.manual_seed(0)


class LoRALinear(nn.Module):
    """W' = W + (alpha / r) * B A, with W frozen and only A, B trained."""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)                      # freeze W and b
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # starts at 0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self) -> nn.Linear:
        """Fold B A into W: no extra cost at inference time."""
        out = nn.Linear(self.base.in_features, self.base.out_features)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            out.bias.copy_(self.base.bias)
        return out


base = nn.Linear(512, 512)              # PyTorch stores W as [out, in]
layer = LoRALinear(base, r=4)
x = torch.randn(3, 512)
print("same output as base at start:", torch.allclose(layer(x), base(x)))

trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print(f"trainable {trainable:,} of {trainable + base.weight.numel():,} weights",
      f"= r * (d_in + d_out) = {4 * (512 + 512):,}")

# A few training steps on a made-up target; only A and B change
opt = torch.optim.AdamW([layer.A, layer.B], lr=1e-2)
target = torch.randn(3, 512)
for _ in range(20):
    loss = ((layer(x) - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print("base weights unchanged:", torch.equal(layer.base.weight, base.weight))
print("merged layer matches adapter:",
      torch.allclose(layer.merged()(x), layer(x), atol=1e-5))
