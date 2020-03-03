import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.optim as optim
import seaborn as sns
from AnalyticalIWAE import *
from Utils import *
sns.set_style("darkgrid")



# Hyperparameters
gif_pics = True
batch_size = 250
lr = 1e-4
num_epochs = 60
train_log = []
test_log = {}
k = 0
num_samples = 5
beta = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AnalyticalIWAE(1024, 512, 32).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)

# Data loading
t = torchvision.transforms.transforms.ToTensor()
train_data = torchvision.datasets.MNIST('./', train=True, transform=t, target_transform=None, download=True)
test_data = torchvision.datasets.MNIST('./', train=False, transform=t, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


for epoch in range(num_epochs):
    for idx, train_iter in enumerate(train_loader):
        batch, label = train_iter[0], train_iter[1]
        batch = batch.view(batch.size(0), -1)  # flatten
        batch = batch.expand(num_samples, batch.shape[0], -1).to(device)  # make num_samples copies

        batch_loss = net.calc_loss(batch, beta)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_log.append(batch_loss.item())
        if beta < 2:
            beta += 0.001 # Warm-up
        k += 1

    loss_batch_mean = []
    for idx, test_iter in enumerate(test_loader):
        batch, label = train_iter[0], train_iter[1]
        batch = batch.view(batch.size(0), -1)  # flatten
        batch = batch.expand(num_samples, batch.shape[0], batch.shape[1]).to(device)  # make num_samples copies

        test_loss = net.calc_loss(batch, 1)
        loss_batch_mean.append(test_loss.detach().item())
    test_log[k] = np.mean(loss_batch_mean)

    if gif_pics and epoch % 2 == 0:
        batch = batch[0,:100,:].squeeze()
        recon_x = net(batch).detach().cpu()
        samples = net.sample(100).detach().cpu()
        fig, axs = plt.subplots(1,2,figsize=(5,10))

        #Reconstructions
        recon_x = create_canvas(recon_x)
        axs[0].set_title('Epoch {} Reconstructions'.format(epoch+1))
        axs[0].axis('off')
        axs[0].imshow(recon_x, cmap='gray')

        #Samples
        samples = net.sample(100).detach().cpu()
        samples = create_canvas(samples)
        axs[1].set_title('Epoch {} Sampled Samples'.format(epoch+1))
        axs[1].axis('off')
        axs[1].imshow(samples, cmap='gray')
        save_path = './Figure/GIF/gif_pic' + str(epoch+1) + '.jpg'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    print('[Epoch: {}/{}][Step: {}]\tTrain Loss: {},\tTest Loss: {}'.format(
        epoch+1, num_epochs, k, round(train_log[k - 1], 2), round(test_log[k], 2)))

###### Loss Curve Plotting ######
Plot_loss_curve(train_log, test_log)
plt.savefig('./Figure/Figure_1.png', bbox_inches='tight')
plt.close()

###### Sampling #########
x = next(iter(train_loader))[0].to(device)
x = x.view(x.size(0), -1)[:100]  # flatten and limit to 100
recon_x = net(x).detach().cpu()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
x_true = create_canvas(x.cpu())
axs[0].set_title('Ground Truth MNIST Digits')
axs[0].axis('off')
axs[0].imshow(x_true, cmap='gray')

recon_x = create_canvas(recon_x)
axs[1].set_title('Reconstructed MNIST Digits')
axs[1].axis('off')
axs[1].imshow(recon_x, cmap='gray')

samples = net.sample(100).detach().cpu()
samples = create_canvas(samples)
axs[2].set_title('Sampled MNIST Digits')
axs[2].axis('off')
axs[2].imshow(samples, cmap='gray')
plt.savefig('./Figure/Figure_2.png', bbox_inches='tight')
plt.close()

