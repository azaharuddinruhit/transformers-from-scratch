N = 1e9        # parameters
D = 2e10       # training tokens

print(f"training compute ~ 6ND = {6 * N * D:.1e} FLOPs")
print(f"weights at 16 bits: {N * 2 / 1e9:.1f} GB, at 4 bits: {N * 0.5 / 1e9:.1f} GB")
print(f"inference ~ 2N = {2 * N:.0e} FLOPs per generated token")
print(f"tokens per parameter: {D / N:.0f} (the Chinchilla rule of thumb is ~20)")


def loss(N, D, L_inf=1.69, a=406.4, r=0.34, b=410.7, s=0.28):
    # Hoffmann et al. (2022), Appendix D.2, Eq. 10: their fitted constants
    return L_inf + a / N ** r + b / D ** s


# Growing only N leaves the data term untouched, so the loss stops improving
for n in (1e8, 1e9, 1e10, 1e11, 1e12):
    print(f"N={n:.0e}, D=2e10: loss {loss(n, 2e10):.3f}")
print(f"floor with D=2e10 however large N is: {loss(float('inf'), 2e10):.3f}")
