# This is a sample Python script to create a Variational Autoencoder.
import torch
from torch import nn


# Input image --> hidden dimension --> mean, sd --> Parametrization trick --> decoder --> output dim
class VariationalAutoEncoder:
    def __init__(self, input_dim, hid_dim=200, z_dim=20):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU()
        )

        # This will push it towards standard gaussian
        self.hd_to_mean = nn.Linear(hid_dim, z_dim)
        self.hd_to_sd = nn.Linear(hid_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, input_dim),
            nn.Sigmoid()
        )

    def parameters(self):
        # Collect parameters from encoder and decoder
        enc_params = list(self.encoder.parameters())
        dec_params = list(self.decoder.parameters())
        return enc_params + dec_params

    def encode(self, x):
        # q_phi(z|x)
        h = self.encoder(x)
        mu = self.hd_to_mean(h)
        sd = self.hd_to_sd(h)
        return mu, sd
        pass

    def decode(self, z):
        # p_theta(x|z)
        return self.decoder(z)

    def forward(self, x):
        mu, sd = self.encode(x)
        epsilon = torch.randn_like(sd)
        z_reparametrized = mu + sd * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sd


if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sd = vae.forward(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sd.shape)
