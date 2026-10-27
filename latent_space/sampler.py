from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

class LatentSampler:
    def __init__(self, generator: nn.Module, classifier: nn.Module, device: str = "cuda") -> None:
        self.generator = generator
        self.classifier = classifier
        self.device = device

    def sample_and_annotate(
        self, 
        num_samples: int = 10000, 
        z_dim: int = 128, 
        batch_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.generator.eval()
        self.classifier.eval()

        z_list, labels_list, conf_list = [], [], []
        iterations = int(np.ceil(num_samples / batch_size))

        with torch.no_grad():
            for _ in range(iterations):
                z = torch.randn(batch_size, z_dim, 1, 1, device=self.device)
                fake_imgs = self.generator(z)

                logits = self.classifier(fake_imgs)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int()

                z_list.append(z.view(batch_size, z_dim).cpu().numpy())
                labels_list.append(preds.cpu().numpy())
                conf_list.append(probs.cpu().numpy())

        latent_z = np.vstack(z_list)[:num_samples]
        binary_labels = np.vstack(labels_list)[:num_samples]
        confidence_scores = np.vstack(conf_list)[:num_samples]

        return latent_z, binary_labels, confidence_scores