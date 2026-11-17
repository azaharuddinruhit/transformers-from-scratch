import numpy as np

rng = np.random.default_rng(0)
height, width, patch, d_model = 336, 336, 14, 64

image = rng.random((height, width, 3))  # stands in for a photo: values 0 to 1 for red, green, blue

# Cut into a grid of patches, then flatten each patch into one row of numbers
rows, cols = height // patch, width // patch
patches = (image.reshape(rows, patch, cols, patch, 3)
                .transpose(0, 2, 1, 3, 4)       # [rows, cols, patch, patch, 3]
                .reshape(rows * cols, patch * patch * 3))
print("image          ", image.shape)
print("patches        ", patches.shape, " one row per image token")

# Project every patch to the model's width with one learned matrix (random here, untrained)
W_patch = rng.normal(0, 0.02, (patch * patch * 3, d_model))
tokens = patches @ W_patch
print("image tokens   ", tokens.shape, " the same width as text token vectors")

# The first patch is the top-left 14 x 14 corner, read row by row
assert np.allclose(patches[0], image[:patch, :patch].reshape(-1))

# Token cost for some common sizes
for side in (224, 336, 448, 672):
    n = (side // patch) ** 2
    print(f"{side} x {side} pixels -> {n:5,} tokens")
