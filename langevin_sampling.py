import os
import torch
import torch.nn as nn
import torchvision
from Utils import filter_dataset
from model.ExplicitIWAE import *
from model.PytorchIWAE import *
import numpy as np
from ebm_model import SmallEBM
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model selection
Explicit = False
Implicit = True
batch_size = 100  # Set batch size for sampling

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


class SGLD:
    def __init__(
        self,
        tau,
        std,
        n_iter,
        step_size,
        threshold,
        msgld=True,
    ):
        """
        MSGLD sampler.
        Args:
        tau: limit step taken taken
        std: std
        n_iter: number of iterations
        step_size: similar to learning rate in from of gradient
        Otherwise, use max_std
        """
        self.tau = tau
        self.std = std
        self.n_iter = n_iter
        self.step_size = step_size
        self.msgld = msgld

    def __call__(
        self,
        log_pdf,
        init,
    ):
        """
        Do SGLD sampling for self.n_iter steps.

        Args:
        log_pdf : Guidance energy
        init: initial sample
        """
        out = init.detach().clone().requires_grad_(True)
        for i in range(self.n_iter):
            lp = log_pdf(out).sum()
            lp.backward()
            out.data = out + self.step_size * torch.clamp(
                out.grad, -self.tau, self.tau
            )  # Gradient ascent
            out.grad.zero_()
        return out.detach().clone()


# Freeze VAE parameters
for param in net.parameters():
    param.requires_grad = False


x = net.sample(64).requires_grad_(True)  # Initial sample from IWAE
# Store samples from each iteration
samples_list = []
samples_list.append(x.detach().clone())  # Store initial sample
ebm_steps = 250

for i in range(ebm_steps):
    mu, log_var = net.encoder(x)
    std = log_var.mul(0.5).exp_()
    qz_Gx_obs = td.Normal(loc=mu, scale=std)
    iwae_z = qz_Gx_obs.sample((10,))  # Shape: (10, batch_size, latent_dim)

    # q(z|x) is the posterior distribution of z given x
    log_prob_qz_Gx = qz_Gx_obs.log_prob(iwae_z).sum(-1)  # Shape: (10, batch_size)
    prior = td.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(std))

    # p(z) is the prior distribution of z
    log_prob_z = prior.log_prob(iwae_z).sum(-1)  # Shape: (10, batch_size)
    mu_x, var_x = net.decoder(iwae_z)
    px_Gz = td.Normal(loc=mu_x, scale=var_x.mul(0.5).exp_())
    log_prob_x = px_Gz.log_prob(x).sum(-1)  # Shape

    approximate_qx = torch.logsumexp(log_prob_x + log_prob_z - log_prob_qz_Gx, dim=0)
    energy_x = ebm(x).squeeze()  # Shape: (batch_size,)

    approximate_px = approximate_qx - energy_x
    approximate_px = approximate_px.sum()
    grad_x = torch.autograd.grad(approximate_px, x, retain_graph=True)[0]
    x = x + 0.1 * grad_x + torch.randn_like(x) * 0.01  # Add noise to the sample

    # Store sample after each iteration
    samples_list.append(x.detach().clone())

# Visualize evolution of multiple samples across 10 evenly spaced steps
num_steps_to_plot = 10  # Number of time steps to show
num_samples_to_show = 10  # Number of different samples to show
step_interval = max(
    1, len(samples_list) // num_steps_to_plot
)  # Calculate step interval
selected_step_indices = [i * step_interval for i in range(num_steps_to_plot)]

# Ensure we don't exceed the available samples
selected_step_indices = [
    min(idx, len(samples_list) - 1) for idx in selected_step_indices
]

fig, axes = plt.subplots(
    num_samples_to_show, num_steps_to_plot, figsize=(20, 2 * num_samples_to_show)
)

for sample_idx in range(num_samples_to_show):
    for step_idx, time_step in enumerate(selected_step_indices):
        # Get the sample at this time step
        sample = samples_list[time_step]

        # Extract the specific sample from the batch
        img = sample[sample_idx].cpu().view(28, 28)

        axes[sample_idx, step_idx].imshow(img, cmap="gray")

        # Add title only for the first row to show time steps
        if sample_idx == 0:
            axes[sample_idx, step_idx].set_title(f"Step {time_step}", fontsize=8)

        # Add ylabel for the first column to show sample number
        if step_idx == 0:
            axes[sample_idx, step_idx].set_ylabel(
                f"Sample {sample_idx + 1}", fontsize=8
            )

        axes[sample_idx, step_idx].axis("off")

plt.tight_layout()
plt.savefig("langevin_sampling_evolution_grid.png", dpi=150, bbox_inches="tight")
plt.show()
