import os
import torchvision
import torch.optim as optim
import seaborn as sns
from model.ExplicitIWAE import *
from model.PytorchIWAE import *
from Utils import *
import sys

sns.set_style("darkgrid")

# Hyperparameters
gif_pics = True
batch_size = 250
lr = 1e-4
ebm_epochs = 50
train_log = []
test_log = {}
k = 0
num_samples = 5
beta = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
Explicit = False
Implicit = True

if (Explicit + Implicit) > 1:
    print("More than one model enabled")
    sys.exit()

if Explicit:
    net = AnalyticalIWAE(1024, 512, 32).to(device)
if Implicit:
    net = PytorchIWAE(1024, 512, 32).to(device)
    # Load from checkpoint if available

model_save_path = "./saved_models/iwae_model.pth"
if os.path.exists(model_save_path):
    net.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Loaded model from {model_save_path}")
else:
    print("No checkpoint found, starting from scratch")

# Freeze VAE parameters
for param in net.parameters():
    param.requires_grad = False

# Data loading
t = torchvision.transforms.transforms.ToTensor()
train_data = torchvision.datasets.MNIST(
    "./", train=True, transform=t, target_transform=None, download=True
)
test_data = torchvision.datasets.MNIST(
    "./", train=False, transform=t, target_transform=None, download=True
)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)



###### Energy-Based Model Implementation ######
class SmallEBM(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super(SmallEBM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()

    def energy(self, x):
        return self.forward(x)


# Initialize EBM
ebm = SmallEBM().to(device)
ebm_optimizer = optim.Adam(ebm.parameters(), lr=1e-3)

# EBM Training
print("Training Energy-Based Model...")
for epoch in range(ebm_epochs):
    for idx, (batch, _) in enumerate(train_loader):
        batch = batch.view(batch.size(0), -1).to(device)

        # Positive samples (real data)
        pos_energy = ebm.energy(batch)

        # Negative samples (from IWAE)
        with torch.no_grad():
            neg_samples = net.sample(batch.shape[0])
        neg_energy = ebm.energy(neg_samples)

        # Contrastive loss
        loss = pos_energy.mean() - neg_energy.mean() 

        # Regularize the norm of the energy :
        reg_loss = torch.mean(pos_energy**2) + torch.mean(neg_energy**2)

        # Regularize the gradients in between in-data and ood samples
        interp = torch.rand(batch.shape[0], 1, device=device)
        x_interp = interp * batch + (1 - interp) * neg_samples
        x_interp.requires_grad_(True)
        energy_interp = torch.mean(ebm(x_interp))
        grad_interp = torch.autograd.grad(
            outputs=energy_interp,
            inputs=x_interp,
            create_graph=True,
            retain_graph=True
        )[0]

        grad_reg_loss = grad_interp.norm(2, dim=1) 

        # Add to loss
        loss += 0.5 * reg_loss + 0.5 * grad_reg_loss.mean()

        ebm_optimizer.zero_grad()
        loss.backward()
        ebm_optimizer.step()
        
        if epoch == 0 and idx == 0:
            print(f"neg_samples shape: {neg_samples.shape}")
            print(f"neg_samples min: {neg_samples.min().item():.4f}, max: {neg_samples.max().item():.4f}")
            
            # Save samples as a plot
            neg_canvas = create_canvas(neg_samples.cpu())
            plt.figure(figsize=(10, 5))
            plt.imshow(neg_canvas, cmap="gray")
            plt.title(f"IWAE Generated Samples - Epoch {epoch}, Batch {idx}")
            plt.axis("off")
            plt.savefig(f"./iwae_samples_epoch_{epoch}_batch_{idx}.png", bbox_inches="tight")
            plt.close()
            print(f"Saved IWAE samples plot to ./iwae_samples_epoch_{epoch}_batch_{idx}.png")

    print(
        f"EBM Epoch {epoch+1}/{ebm_epochs}, Loss: {loss.item():.4f}, pos_energy: {pos_energy.mean().item():.4f}, neg_energy: {neg_energy.mean().item():.4f}"
    )

# Save the trained EBM model
ebm_save_path = "./saved_models/ebm_model.pth"
os.makedirs(os.path.dirname(ebm_save_path), exist_ok=True)
torch.save(ebm.state_dict(), ebm_save_path)
print(f"Saved EBM model to {ebm_save_path}")
