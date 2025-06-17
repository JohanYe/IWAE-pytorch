import os
import torchvision
import torch.optim as optim
import seaborn as sns
from model.ExplicitIWAE import *
from model.PytorchIWAE import *
from Utils import *

sns.set_style("darkgrid")

# Hyperparameters
gif_pics = True
batch_size = 250
lr = 1e-4
num_epochs = 50
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
optimizer = optim.Adam(net.parameters(), lr=lr)

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


# Data loading with skewed distribution
def skew_dataset(dataset, skew_digits, remove_ratio):
    """
    Skew the dataset by removing a percentage of specified digits.
    Args:
        dataset: The dataset to skew (e.g., MNIST).
        skew_digits: List of digits to skew (e.g., [1, 2, 3, 4, 5]).
        remove_ratio: Fraction of samples to remove for each digit (e.g., 0.8 for 80%).
    Returns:
        Skewed dataset.
    """
    targets = dataset.targets.numpy()
    data = dataset.data.numpy()

    mask = np.ones(len(targets), dtype=bool)
    for digit in skew_digits:
        digit_indices = np.where(targets == digit)[0]
        remove_count = int(len(digit_indices) * remove_ratio)
        remove_indices = np.random.choice(digit_indices, remove_count, replace=False)
        mask[remove_indices] = False

    dataset.targets = torch.tensor(targets[mask])
    dataset.data = torch.tensor(data[mask])
    return dataset


# Skew the training dataset
skew_digits = [1, 2, 3, 4, 5]  # Digits to skew
remove_ratio = 0.8  # Remove 80% of these digits
train_data = skew_dataset(train_data, skew_digits, remove_ratio)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)


for epoch in range(num_epochs):
    for idx, train_iter in enumerate(train_loader):
        batch, label = train_iter[0], train_iter[1]
        batch = batch.view(batch.size(0), -1)  # flatten
        batch = batch.expand(num_samples, batch.shape[0], -1).to(
            device
        )  # make num_samples copies

        batch_loss = net.calc_loss(batch, beta)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_log.append(batch_loss.item())
        if beta < 2:
            beta += 0.001  # Warm-up
        k += 1

    loss_batch_mean = []
    for idx, test_iter in enumerate(test_loader):
        batch, label = train_iter[0], train_iter[1]
        batch = batch.view(batch.size(0), -1)  # flatten
        batch = batch.expand(num_samples, batch.shape[0], batch.shape[1]).to(
            device
        )  # make num_samples copies

        test_loss = net.calc_loss(batch, beta)

        loss_batch_mean.append(test_loss.detach().item())

    test_log[k] = np.mean(loss_batch_mean)
    if gif_pics and epoch % 2 == 0:
        if Explicit:
            batch = batch[0, :100, :].squeeze()
            recon_x = net(batch)
        else:
            batch = batch[0, :100, :].unsqueeze(0)
            recon_x = net(batch)[0].squeeze()  # get mu only

        samples = net.sample(100).detach().cpu()
        fig, axs = plt.subplots(1, 2, figsize=(5, 10))

        # Reconstructions
        recon_x = create_canvas(recon_x.detach().cpu())
        axs[0].set_title("Epoch {} Reconstructions".format(epoch + 1))
        axs[0].axis("off")
        axs[0].imshow(recon_x, cmap="gray")

        # Samples
        samples = create_canvas(samples)
        axs[1].set_title("Epoch {} Sampled Samples".format(epoch + 1))
        axs[1].axis("off")
        axs[1].imshow(samples, cmap="gray")
        save_path = "./Figure/GIF/gif_pic" + str(epoch + 1) + ".jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    print(
        "[Epoch: {}/{}][Step: {}]\tTrain Loss: {},\tTest Loss: {}".format(
            epoch + 1, num_epochs, k, round(train_log[k - 1], 2), round(test_log[k], 2)
        )
    )

###### Loss Curve Plotting ######
Plot_loss_curve(train_log, test_log)
plt.savefig("./Figure/Figure_1.png", bbox_inches="tight")
plt.close()

###### Sampling #########
x = next(iter(train_loader))[0].to(device)
x = x.view(x.size(0), -1)[:100]  # flatten and limit to 100
if Explicit:
    recon_x = net(x)
else:
    recon_x = net(x)[0]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
x_true = create_canvas(x.detach().cpu())
axs[0].set_title("Ground Truth MNIST Digits")
axs[0].axis("off")
axs[0].imshow(x_true, cmap="gray")

recon_x = create_canvas(recon_x.detach().cpu())
axs[1].set_title("Reconstructed MNIST Digits")
axs[1].axis("off")
axs[1].imshow(recon_x, cmap="gray")

samples = net.sample(100).detach().cpu()
samples = create_canvas(samples)
axs[2].set_title("Sampled MNIST Digits")
axs[2].axis("off")
axs[2].imshow(samples, cmap="gray")
plt.savefig("./Figure/Figure_2.png", bbox_inches="tight")
plt.close()

# Save the model
model_save_path = "./saved_models/iwae_model.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(net.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")