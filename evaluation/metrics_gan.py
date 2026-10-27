from typing import Tuple
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class FeatureExtractorInception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        self.model = model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)

class GANMetricsEvaluator:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.feature_extractor = FeatureExtractorInception().to(self.device)

    def extract_statistics(self, images: torch.Tensor, batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = images.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        features = []

        with torch.no_grad():
            for i in range(n_batches):
                batch = images[i * batch_size : (i + 1) * batch_size].to(self.device)
                feat = self.feature_extractor(batch).cpu().numpy()
                features.append(feat)

        features = np.vstack(features)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def compute_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor, batch_size: int = 50) -> float:
        mu_r, sigma_r = self.extract_statistics(real_images, batch_size)
        mu_g, sigma_g = self.extract_statistics(fake_images, batch_size)

        diff = mu_r - mu_g
        covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * np.trace(covmean)
        return float(fid)

    def compute_inception_score(self, fake_images: torch.Tensor, splits: int = 10, batch_size: int = 50) -> Tuple[float, float]:
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False).to(self.device)
        model.eval()

        n_samples = fake_images.shape[0]
        preds = []
        with torch.no_grad():
            for i in range(int(np.ceil(n_samples / batch_size))):
                batch = fake_images[i * batch_size : (i + 1) * batch_size].to(self.device)
                batch = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                prob = torch.softmax(model(batch), dim=1).cpu().numpy()
                preds.append(prob)

        preds = np.vstack(preds)
        split_scores = []
        chunk_size = preds.shape[0] // splits

        for k in range(splits):
            part = preds[k * chunk_size : (k + 1) * chunk_size, :]
            p_y = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
            split_scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

        return float(np.mean(split_scores)), float(np.std(split_scores))