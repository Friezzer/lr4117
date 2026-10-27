import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img: int = 3, features_d: int = 64, norm_type: str = "instance") -> None:
        super().__init__()
        
        def get_norm(features: int):
            if norm_type == "instance":
                return nn.InstanceNorm2d(features, affine=True)
            elif norm_type == "layer":
                return nn.GroupNorm(1, features)
            else:
                return nn.Identity()

        self.net = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            get_norm(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            get_norm(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            get_norm(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1 (скалярная оценка без сигмоиды)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)