import math
import re

# 1. A chat template turns a conversation into one token sequence (ChatML-style markers,
#    as Qwen uses). Supervised fine-tuning puts the loss only on the assistant's reply.
conversation = [("system", "You are a helpful assistant."),
                ("user", "What did the cat sit on?"),
                ("assistant", "The cat sat on the mat.")]
tokens, trained = [], []
for role, text in conversation:
    words = re.findall(r"\w+|[^\w\s]", text)          # stand-in for a real tokenizer
    is_reply = role == "assistant"
    for piece, learn in ([("<|im_start|>", False), (role, False), ("\n", False)]
                         + [(w, is_reply) for w in words]
                         + [("<|im_end|>", is_reply), ("\n", False)]):
        tokens.append(piece)
        trained.append(learn)
print(f"{len(tokens)} tokens, loss on {sum(trained)}:",
      " ".join(t for t, k in zip(tokens, trained) if k))

# 2. Preference tuning: the best policy for "reward minus lambda x KL from the reference"
#    reweights the reference by exp(reward / lambda).
answers = ["The mat.", "Great question! ... a soft woven mat ...", "The sofa."]
ref = [0.5, 0.3, 0.2]                                  # the SFT model's probabilities
reward = [2.0, 1.0, -2.0]                              # a careful reward model


def tune(ref, reward, lam):
    w = [p * math.exp(r / lam) for p, r in zip(ref, reward)]
    return [x / sum(w) for x in w]


for lam in (4, 1, 0.25):
    pi = tune(ref, reward, lam)
    kl = sum(p * math.log(p / q) for p, q in zip(pi, ref))
    print(f"lambda={lam:<4} probabilities {[round(p, 3) for p in pi]}  KL {kl:.3f} nats")

# 3. DPO trains on a pair (chosen, rejected). At the start the policy equals the
#    reference and the loss is log 2; at the optimum above the margin is the reward gap.
def dpo_loss(pi, ref, lam, chosen, rejected):
    margin = lam * (math.log(pi[chosen] / ref[chosen]) - math.log(pi[rejected] / ref[rejected]))
    return -math.log(1 / (1 + math.exp(-margin)))


lam = 1.0
print(f"DPO loss before tuning {dpo_loss(ref, ref, lam, 0, 2):.3f}, "
      f"at the optimum {dpo_loss(tune(ref, reward, lam), ref, lam, 0, 2):.3f}")

# 4. Test-time compute: sample several attempts and take the majority answer.
q = 0.6                                                # chance one attempt is right
for n in (1, 5, 9):
    p = sum(math.comb(n, k) * q**k * (1 - q)**(n - k) for k in range(n // 2 + 1, n + 1))
    print(f"majority of {n} independent attempts is right with probability {p:.3f}")
