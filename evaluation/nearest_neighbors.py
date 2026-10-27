from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

class PerceptualLossEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        base_resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return torch.flatten(feats, 1)

class NearestNeighborFinder:
    def __init__(self, train_loader: DataLoader, device: str = "cuda") -> None:
        self.train_loader = train_loader
        self.device = device
        self.embedder = PerceptualLossEmbedder().to(self.device)

    def find_pixel_l2_neighbors(
        self, 
        query_image: torch.Tensor, 
        top_k: int = 1
    ) -> Tuple[List[torch.Tensor], List[float]]:
        min_dist = float('inf')
        best_match = None
        q_flat = query_image.flatten().to(self.device)

        with torch.no_grad():
            for real_imgs, _ in self.train_loader:
                real_imgs = real_imgs.to(self.device)
                dists = torch.norm(real_imgs.view(real_imgs.size(0), -1) - q_flat, dim=1)
                val, idx = torch.min(dists, dim=0)
                if val.item() < min_dist:
                    min_dist = val.item()
                    best_match = real_imgs[idx].cpu()

        return [best_match], [min_dist]

    def find_perceptual_neighbors(
        self, 
        query_image: torch.Tensor, 
        top_k: int = 1
    ) -> Tuple[List[torch.Tensor], List[float]]:
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)

        with torch.no_grad():
            q_feat = self.embedder(query_image.to(self.device))
            q_feat = q_feat / q_feat.norm(dim=1, keepdim=True)

        best_dist = float('inf')
        best_match = None

        with torch.no_grad():
            for real_imgs, _ in self.train_loader:
                real_imgs = real_imgs.to(self.device)
                feats = self.embedder(real_imgs)
                feats = feats / feats.norm(dim=1, keepdim=True)
                
                # Косинусное расстояние: 1 - cos_sim
                cos_dist = 1.0 - torch.mm(q_feat, feats.t()).squeeze(0)
                val, idx = torch.min(cos_dist, dim=0)
                if val.item() < best_dist:
                    best_dist = val.item()
                    best_match = real_imgs[idx].cpu()

        return [best_match], [best_dist]