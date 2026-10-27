import torch
import torch.nn as nn
from torchvision import models

class AttributeClassifier(nn.Module):
    def __init__(self, num_attributes: int = 4, pretrained: bool = True, dropout_rate: float = 0.3) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)
        
        # Сверточный экстрактор до avgpool
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])
        in_features = base_resnet.fc.in_features
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_attributes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return torch.flatten(feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.fc(feats)