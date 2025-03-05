import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 2))

        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder

        # self.fc3 = nn.Linear(latent_dim, 128 * 4 * 4)

        self.fc3 = nn.Linear(latent_dim, 1024 * 2 * 2)


        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def encode(self, x):
        x = F.relu(self.bn16(self.enc_conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn32(self.enc_conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn64(self.enc_conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn128(self.enc_conv4(x)))
        # x = self.pool(x)

        x = self.adaptive_pool(x)  # Assure une sortie de 1024

        x = x.view(x.size(0), -1)

        mu, logvar = self.fc1(x), self.fc2(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc3(z)
        x = x.view(x.size(0), 1024, 2, 2)

        x = self.upsample(F.relu(self.dec_conv1(x)))
        x = self.upsample(F.relu(self.dec_conv2(x)))
        x = self.upsample(F.relu(self.dec_conv3(x)))
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss