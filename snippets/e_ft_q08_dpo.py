import torch
import torch.nn.functional as F


def sequence_logprob(logits, labels):
    """Sum of log p(token) over a response. logits [T, V], labels [T]."""
    logp = F.log_softmax(logits, dim=-1)
    return logp[torch.arange(len(labels)), labels].sum()


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Inputs are summed log-probabilities of whole responses, shape [batch]."""
    margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


torch.manual_seed(0)
logits, labels = torch.randn(5, 10), torch.randint(0, 10, (5,))
print(f"log p(response) = {sequence_logprob(logits, labels).item():.3f}")

ref_c, ref_r = torch.tensor([-12.0]), torch.tensor([-11.0])  # frozen reference

# Policy equal to the reference: loss = ln 2
print(f"policy = reference: {dpo_loss(ref_c, ref_r, ref_c, ref_r).item():.4f}")

# Policy now prefers the chosen answer more than the reference does
pi_c, pi_r = torch.tensor([-9.0]), torch.tensor([-14.0])
print(f"policy prefers chosen: {dpo_loss(pi_c, pi_r, ref_c, ref_r).item():.4f}")

# The gradient raises the chosen log-prob and lowers the rejected one
pi_c = torch.tensor([-12.0], requires_grad=True)
pi_r = torch.tensor([-11.0], requires_grad=True)
dpo_loss(pi_c, pi_r, ref_c, ref_r).backward()
print(f"step direction: chosen {-pi_c.grad.item():+.3f}, "
      f"rejected {-pi_r.grad.item():+.3f}")
