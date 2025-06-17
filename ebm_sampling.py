import os
import torch
import torch.nn as nn
from model.ExplicitIWAE import *
from model.PytorchIWAE import *
import numpy as np
from ebm_model import SmallEBM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model selection
Explicit = False
Implicit = True

# Load IWAE model
if Explicit:
    net = AnalyticalIWAE(1024, 512, 32).to(device)
if Implicit:
    net = PytorchIWAE(1024, 512, 10).to(device)

model_save_path = "./saved_models/iwae_model.pth"
if os.path.exists(model_save_path):
    net.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Loaded model from {model_save_path}")
else:
    print("No checkpoint found, starting from scratch")

# Freeze VAE parameters
for param in net.parameters():
    param.requires_grad = False

# Load EBM model
ebm = SmallEBM().to(device)
ebm_save_path = "./saved_models/ebm_model.pth"
if os.path.exists(ebm_save_path):
    ebm.load_state_dict(torch.load(ebm_save_path, map_location=device))
    print(f"Loaded EBM model from {ebm_save_path}")
else:
    print("No EBM checkpoint found")


iwae_samples = net.sample(100)
ebm_energy = ebm.energy(iwae_samples)
ebm_energy = (-ebm_energy).exp()

normalized_ebm_energy = ebm_energy / ebm_energy.sum()
weighted_energy = normalized_ebm_energy.unsqueeze(1) * iwae_samples

import matplotlib.pyplot as plt

# Reshape and save samples as 28x28 images

# Reshape weighted_energy to 28x28 images (assuming it's flattened MNIST-like data)
num_samples = weighted_energy.shape[0]
images = weighted_energy.detach().cpu().numpy().reshape(num_samples, 28, 28)

# Get original IWAE samples for comparison
original_images = iwae_samples.detach().cpu().numpy().reshape(num_samples, 28, 28)

# Create a grid plot showing original and weighted images side by side
fig, axes = plt.subplots(4, 10, figsize=(24, 10))  # 4 rows, 10 columns (5 pairs)

for i in range(min(20, num_samples)):
    row = i // 5
    col = (i % 5) * 2

    # Original image
    axes[row, col].imshow(original_images[i], cmap="gray")
    axes[row, col].axis("off")
    axes[row, col].set_title(f"Original {i+1}")

    # Weighted image
    axes[row, col + 1].imshow(images[i], cmap="gray")
    axes[row, col + 1].axis("off")
    axes[row, col + 1].set_title(f"Weighted {i+1}")

plt.tight_layout()
plt.savefig("./comparison_images.png", dpi=300, bbox_inches="tight")
plt.show()

# Save individual comparison images
os.makedirs("./comparison_samples", exist_ok=True)
for i in range(num_samples):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    ax1.imshow(original_images[i], cmap="gray")
    ax1.axis("off")
    ax1.set_title("Original")

    ax2.imshow(images[i], cmap="gray")
    ax2.axis("off")
    ax2.set_title("Weighted")

    plt.tight_layout()
    plt.savefig(
        f"./comparison_samples/comparison_{i:03d}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

print(f"Saved {num_samples} comparison images to ./comparison_samples/")
