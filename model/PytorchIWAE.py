import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F


# define network
class PytorchIWAE(nn.Module):
    # Network uses in-built pytorch function for variational inference, instead of having to explicitly
    # calculate it
    def __init__(self, num_hidden1, num_hidden2, latent, in_dim=784):
        super(PytorchIWAE, self).__init__()
        self.latent = latent
        self.out_dec = in_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.block1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(),
        )
        self.mu_enc = nn.Linear(in_features=num_hidden2, out_features=self.latent)
        self.lvar_enc = nn.Linear(in_features=num_hidden2, out_features=self.latent)

        self.block2 = nn.Sequential(
            nn.Linear(in_features=self.latent, out_features=num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_hidden1),
            nn.ReLU(),
        )

        self.mu_dec = nn.Linear(in_features=num_hidden1, out_features=self.out_dec)
        self.lvar_dec = nn.Linear(in_features=num_hidden1, out_features=self.out_dec)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def encoder(self, x):
        x = self.block1(x)
        mu = self.mu_enc(x)
        log_var = self.lvar_enc(x)
        return mu, log_var

    def decoder(self, z):
        h = self.block2(z)

        mu_x = torch.sigmoid(self.mu_dec(h))
        var_x = torch.sigmoid(self.lvar_dec(h))  # Stability reasons

        return (mu_x, var_x)


    def reparameterize(self, mu, std):
        qz_Gx_obs = td.Normal(loc=mu, scale=std)
        # find z|x
        z_Gx = qz_Gx_obs.rsample()
        return z_Gx, qz_Gx_obs

    def forward(self, x, train=True):
        mu, log_var = self.encoder(x)
        if train:
            std = log_var.mul(0.5).exp_()
            z, _ = self.reparameterize(mu, std)
        else:
            z = mu
        x_recon = self.decoder(z)
        # can also show mu
        return x_recon

    def calc_loss(self, x, beta):
        # Encode
        mu_enc, log_var_enc = self.encoder(x)
        std_enc = torch.exp(0.5 * log_var_enc)

        # Reparameterize:
        z_Gx, qz_Gx_obs = self.reparameterize(mu_enc, std_enc)
        mu_dec, log_var_dec = self.decoder(z_Gx)

        # Find q(z|x)
        log_QhGx = qz_Gx_obs.log_prob(z_Gx)
        log_QhGx = torch.sum(log_QhGx, -1)

        # Find p(z)
        mu_prior = torch.zeros(self.latent).to(self.device)
        std_prior = torch.ones(self.latent).to(self.device)
        p_z = td.Normal(loc=mu_prior, scale=std_prior)
        log_Ph = torch.sum(p_z.log_prob(z_Gx), -1)

        # Find p(x|z)
        std_dec = log_var_dec.mul(0.5).exp_()
        px_Gz = td.Normal(loc=mu_dec, scale=std_dec).log_prob(x)
        log_PxGh = torch.sum(px_Gz, -1)
        # print(log_PxGh, log_Ph, log_QhGx)
        # Calculate loss

        w = log_PxGh + (log_Ph - log_QhGx)*beta
        loss = -torch.mean(torch.logsumexp(w, 0))
        

        return loss

    def sample(self, num_samples):
        z_dist = td.Normal(loc=torch.zeros([num_samples, self.latent]), scale=1)
        z_sample = z_dist.sample().unsqueeze(0).to(self.device)
        samples = self.decoder(z_sample)[0].view(num_samples, -1)
        return samples
