import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class VAE_Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=512):
        super(VAE_Encoder, self).__init__()

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

        self.fc1 = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn16(self.enc_conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn32(self.enc_conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn64(self.enc_conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn128(self.enc_conv4(x)))

        x = self.adaptive_pool(x)  # Assure une sortie de 1024

        x = x.view(x.size(0), -1)

        mu, logvar = self.fc1(x[:, :512]), self.fc2(x[:, 512:])
        return mu, logvar

class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=512, output_channels=3):
        super(VAE_Decoder, self).__init__()

        # Fully Connected Layer pour transformer le vecteur latent en tenseur 4D
        self.fc = nn.Linear(latent_dim, 128 * 2 * 2)

        # Première opération d'upsampling après le view
        self.initial_upsample = nn.Upsample(scale_factor=4, mode='nearest')
        # scale_factor=4 : sortie 64 x 64
        # scale_factor=8 : sortie 128 x 128
        # scale_factor=14 : sortie 224 x 224

        # Déconvolution & Upsampling progressifs
        self.deconv_blocks = nn.Sequential(
            #nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(True),
            #nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        # Projection du vecteur latent en tenseur 4D
        x = self.fc(z)
        # x = x.view(x.size(0), 128, 4, 4) pour avoir image 64 x 64 en sortie
        #x = x.view(x.size(0), 128, 14, 14) # pour avoir image 224 x 224 en sortie
        #x = x.view(x.size(0), 128, 8, 8) # 128 x 128 en sortie
        x = x.view(x.size(0), 128, 2, 2)

        # **Upsampling avant la première déconv**
        x = self.initial_upsample(x)

        # Upsampling & Convolutions transposées
        x = self.deconv_blocks(x)

        return x

# Fonction de réparamétrisation de z
def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std) # Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
    return mu + eps * std  # z = μ + ε * σ

# Classe complète du VAE
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=512):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_channels, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, input_channels)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    r_loss = recon_loss(recon_x, x)
    # kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    prior = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    posterior = Normal(mu, torch.exp(0.5 * logvar))
    kld_loss = kl_divergence(posterior, prior).sum(dim=1).mean()

    return r_loss + kld_loss

def recon_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')