# This is a sample Python script to create a Convolutional Variational Autoencoder.
import torch
from torch import nn



class VariationalAutoEncoder(nn.Module):
    def __init__(self, image_size, z_dim=128):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        hid_dim = image_size * 16 * 16
        self.hd_to_mean = nn.Linear(hid_dim, z_dim)
        self.hd_to_sd = nn.Linear(hid_dim, z_dim)

        self.fc = nn.Linear(z_dim, image_size * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
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
        z = self.fc(z)
        z = z.view(-1, 128, 8, 8)
        return self.decoder(z)

    def forward(self, x):
        # Input image --> hidden dimension --> mean, sd --> Parametrization trick --> decoder --> output dim
        mu, sd = self.encode(x)
        epsilon = torch.randn_like(sd)
        z_reparametrized = mu + sd * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sd

