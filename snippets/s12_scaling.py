N = 1e9        # parameters
D = 2e10       # training tokens

print(f"training compute ~ 6ND = {6 * N * D:.1e} FLOPs")
print(f"inference ~ 2N = {2 * N:.0e} FLOPs per generated token")
print(f"tokens per parameter: {D / N:.0f} (the compute-optimal rule of thumb is ~20)")
print(f"Llama 3 8B, over 15e12 tokens: {15e12 / 8e9:.0f} tokens per parameter")
