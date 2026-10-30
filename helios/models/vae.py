"""TinyVAE —— 逐帧 2D 卷积 VAE，8x 空间下采样。

早期版本说明：视频按帧独立编解码（后续版本再换因果 3D VAE）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.n1(x)))
        h = self.conv2(F.silu(self.n2(h)))
        return x + h


class TinyVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = 8
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # Encoder: H,W -> H/8, W/8
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            ResBlock(c1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            ResBlock(c2),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            ResBlock(c3),
            nn.Conv2d(c3, c3, 3, stride=2, padding=1),
            ResBlock(c3),
        )
        self.to_latent = nn.Conv2d(c3, latent_channels * 2, 3, padding=1)

        # Decoder
        self.from_latent = nn.Conv2d(latent_channels, c3, 3, padding=1)
        self.dec = nn.Sequential(
            ResBlock(c3),
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1),
            ResBlock(c2),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),
            ResBlock(c1),
            nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1),
            ResBlock(c1),
        )
        self.to_image = nn.Conv2d(c1, in_channels, 3, padding=1)

    def encode(self, x):
        """x: [B,3,H,W] -> mean, logvar: [B,C,H/8,W/8]"""
        h = self.to_latent(self.enc(x))
        return h.chunk(2, dim=1)

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        return mean + std * torch.randn_like(std)

    def decode(self, z):
        return self.to_image(self.dec(self.from_latent(z)))

    def forward(self, x):
        mean, logvar = self.encode(x)
        return self.decode(self.reparameterize(mean, logvar)), mean, logvar


def vae_loss(x, recon, mean, logvar, kl_weight: float = 1e-6):
    l1 = F.l1_loss(recon, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mean.pow(2) - logvar.exp())
    return l1 + kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}
