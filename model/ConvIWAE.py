import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributions as td

# NOT INTEGRATED WITH MAIN.PY, ONLY FOR SHOW


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 50 * 7 * 7)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 50, 7, 7)


class ConvIWAE(nn.Module):
    def __init__(self, z_dim=20, bs):
        super(ConvIWAE, self).__init__()
        self.z_dim = z_dim
        self.analytic_kl = True
        self.batch_size = bs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(240, 160, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 80, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.BatchNorm2d(80),
            nn.Conv2d(80, 50, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            Flatten(),
            nn.Linear(50 * 7 * 7, 700),
            nn.ReLU())

        self.mu = nn.Linear(700, z_dim)
        self.std = nn.Linear(700, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim, 700), nn.ReLU(),
            nn.Linear(700, 50 * 7 * 7), nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose2d(50, 220, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.BatchNorm2d(220),
            nn.ConvTranspose2d(220, 160, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.ConvTranspose2d(160, 60, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.ConvTranspose2d(60, 2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())  # not sure why we sigmoid variance
        #

    def encoder(self, x):
        x = self.block1(x)
        mu = self.mu(x)
        std = nn.functional.softplus(self.std(x))
        return mu, std

    def reparameterize(self, mu, std, K, training):
        if training == True:
            qz_Gx_obs = td.Normal(loc=mu, scale=std)
            z = qz_Gx_obs.rsample(torch.Size([K]))  # if we do [m,k] we get (m,k,batch,zdim)
        else:
            z = mu.view(self.batch_size, -1)
        return z

    def decoder(self, z):
        K, bs = z.size(0), z.size(1)
        z = z.view([K * bs, -1])
        x = self.dec(z)
        x = x.view([K, bs, 2, 28, 28])
        x_mean = x[:, :, :1, :, :]  # (K, bs, param, dim1, dim2)
        x_std = nn.functional.softplus(x[:, :, 1:, :, :])  # (K, bs, param, dim1, dim2)
        return x_mean, x_std

    def elbo(self, mu_dec, std_dec, mu_enc, std_enc, x, z, K, beta):
        qz_Gx_obs = td.Normal(loc=mu_enc, scale=std_enc)  # z_dist
        p_z = td.Normal(torch.zeros([self.z_dim]).to(self.device), torch.ones([self.z_dim]).to(self.device))

        if self.analytic_kl:
            kl = td.kl_divergence(qz_Gx_obs, p_z).sum(-1)
        else:
            lpz = p_z.log_prob(z).sum(-1)
            lqzx = qz_Gx_obs.log_prob(z).sum(-1)
            kl = lqzx - lpz

        xgz = td.Normal(loc=mu_dec, scale=std_dec)  # x_dist
        lpxgz = xgz.log_prob(x).sum([-1, -2, -3])  # (K,bs)
        elbo = lpxgz - beta * kl

        loss = -torch.mean(torch.logsumexp(elbo, 0))
        return loss

    def forward(self, x, K, beta, training):
        mu_enc, std_enc = self.encoder(x)
        z = self.reparameterize(mu_enc, std_enc, K, training)
        mu_dec, std_dec = self.decoder(z)  # (K, bs, dim1, dim2)
        loss = self.elbo(mu_dec, std_dec, mu_enc, std_enc, x, z, K, beta)

        return loss, mu_enc, std_enc, mu_dec, std_dec