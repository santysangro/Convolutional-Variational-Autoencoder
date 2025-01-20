# This is a sample Python script to create a Convolutional Variational Autoencoder.
import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, image_size, z_dim=128):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )

        # Fully connected layers for latent space
        self.hid_dim = 512 * (image_size // 16) * (image_size // 16)
        self.hd_to_mean = nn.Linear(self.hid_dim, z_dim)
        self.hd_to_sd = nn.Linear(self.hid_dim, z_dim)
        self.fc = nn.Linear(z_dim, self.hid_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
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
        z = z.view(-1, 512, 8, 8)
        return self.decoder(z)

    def forward(self, x):
        # Input image --> hidden dimension --> mean, sd --> Parametrization trick --> decoder --> output dim
        mu, sd = self.encode(x)
        epsilon = torch.randn_like(sd)
        z_reparametrized = mu + sd * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sd
