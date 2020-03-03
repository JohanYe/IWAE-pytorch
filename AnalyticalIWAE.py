import torch
import torch.nn as nn
import numpy as np

# define network
class AnalyticalIWAE(nn.Module):

    def __init__(self, num_hidden1, num_hidden2, latent_space):
        super(AnalyticalIWAE, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=784, out_features=num_hidden1),
            # IWAE_BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(),
            # nn.BatchNorm1d(num_hidden2), #This comma is ok!? makes it easier to add and remove modules
        )

        self.fc21 = nn.Linear(in_features=num_hidden2, out_features=latent_space)
        self.fc22 = nn.Linear(in_features=num_hidden2, out_features=latent_space)

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=latent_space, out_features=num_hidden2),
            nn.ReLU(),
            # nn.BatchNorm1d(num_hidden2),
            nn.Linear(num_hidden2, num_hidden1),
            nn.ReLU(),
            # nn.BatchNorm1d(num_hidden1),
        )
        self.decode = nn.Linear(in_features=num_hidden1, out_features=784)
        self.latent = latent_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def encode(self, x):
        x = self.fc1(x)
        mu = self.fc21(x)
        log_var = self.fc22(x)

        # Reparameterize
        eps = torch.randn_like(log_var)
        h = mu + torch.exp(log_var * 0.5) * eps
        return mu, log_var, h, eps

    def forward(self, x):
        """
        Purely to see reconstruction, not for calculating loss.
        """

        # Encode
        mu, log_var, h, eps = self.encode(x)

        # decode
        recon_X = self.fc3(h)
        recon_X = torch.sigmoid(self.decode(recon_X))
        return recon_X

    def calc_loss(self, x, beta):

        # Encode
        mu, log_var, h, eps = self.encode(x)

        # Calculating P(x,h)
        log_Ph = torch.sum(-0.5 * h ** 2 - 0.5 * torch.log(2 * h.new_tensor(np.pi)),
                           -1)  # equivalent to lognormal if mu=0,std=1 (i think)
        recon_X = torch.sigmoid(self.decode(self.fc3(h)))  # Creating reconstructions
        log_PxGh = torch.sum(x * torch.log(recon_X) + (1 - x) * torch.log(1 - recon_X),
                             -1)  # Bernoulli decoder: Appendix c.1 Kingma p(x|h)
        log_Pxh = log_Ph + log_PxGh  # log(p(x,h))
        log_QhGx = torch.sum(-0.5 * (eps) ** 2 - 0.5 * torch.log(2 * h.new_tensor(np.pi)) - 0.5 * log_var,
                             -1)  # Evaluation in lognormal

        # Weighting according to equation 13 from IWAE paper
        log_weight = (log_Pxh - log_QhGx).detach().data
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)

        # scaling
        loss = torch.mean(-torch.sum(weight * (log_Pxh - log_QhGx), 0))
        return loss

    def sample(self, n_samples):
        eps = torch.randn((n_samples, self.latent)).to(self.device)
        sample = self.fc3(eps)
        sample = torch.sigmoid(self.decode(sample))
        return sample
