import re

# A chat template turns a conversation into one token sequence (ChatML-style markers,
# as Qwen uses). Supervised fine-tuning puts the loss only on the assistant's reply.
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
