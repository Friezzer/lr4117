import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, channels_img: int = 3, features_g: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # Вход: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.z_dim, 1, 1)
        return self.net(z)