import torch
from torch import nn
import torch.nn.functional as F

class VaeModel(nn.Module):
    def __init__(self, img_dim, hidden_dim, z_dim):
        super().__init__()

        #encoding
        self.img2hid = nn.Linear(img_dim, hidden_dim)
        self.hid2mu = nn.Linear(hidden_dim, z_dim)
        self.hid2logvar = nn.Linear(hidden_dim, z_dim)

        #decoding
        self.z2hid = nn.Linear(z_dim, hidden_dim)
        self.hid2img = nn.Linear(hidden_dim, img_dim)

    def encode(self, img):
        x = F.relu(self.img2hid(img))
        mu = self.hid2mu(x)
        logvar = self.hid2logvar(x)

        return mu,logvar

    def decode(self, z):
        x = F.relu(self.z2hid(z))
        return F.sigmoid(self.hid2img(x))

    # Returns mu, sigma, recon
    def forward(self, img):
        mu, logvar = self.encode(img)
        # Reparameterization  trick
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z)

        return mu, logvar, recon