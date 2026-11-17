# Llama 3.1 8B, from its config.json
d_model, d_kv, d_ff, layers = 4096, 1024, 14336, 32   # keys and values: 8 KV heads of 128
vocab = 128256
r = 16                                                # LoRA rank

shapes = {"query": (d_model, d_model), "key": (d_model, d_kv), "value": (d_model, d_kv),
          "output": (d_model, d_model), "gate": (d_model, d_ff), "up": (d_model, d_ff), "down": (d_ff, d_model)}

linear = layers * sum(d_in * d_out for d_in, d_out in shapes.values())
norms = (2 * layers + 1) * d_model
tables = 2 * vocab * d_model                          # token embeddings and the output layer (not tied)
N = linear + norms + tables
lora = layers * sum(r * (d_in + d_out) for d_in, d_out in shapes.values())
print(f"parameters      {N:,}")
print(f"LoRA, rank {r}   {lora:,} = {lora / N:.2%} of the model")

GB = 1e9
nf4 = 4 + 8 / 64 + 32 / (64 * 256)                    # 4 bits, plus scales quantized again ("double quantization")
print(f"\nfull fine-tuning  {16 * N / GB:6.1f} GB  (16 bytes per parameter)")
print(f"LoRA over bf16    {(2 * N + 16 * lora) / GB:6.1f} GB")
print(f"QLoRA             {(linear * nf4 / 8 + 2 * (tables + norms) + 16 * lora) / GB:6.1f} GB  ({nf4:.3f} bits per block weight)")
print("(all before activations)")
