from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs.config import TrainingConfig
from evaluation.metrics_cls import compute_classification_metrics

class ClassifierTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        target_indices: List[int],
        target_names: List[str],
        train_config: TrainingConfig
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_indices = target_indices
        self.target_names = target_names
        self.cfg = train_config
        self.device = self.cfg.device
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr_cls, weight_decay=1e-4)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            targets = labels[:, self.target_indices].float().to(self.device)
            targets = torch.clamp(targets, min=0.0)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader: DataLoader) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                targets = labels[:, self.target_indices].float().to(self.device)
                targets = torch.clamp(targets, min=0.0).cpu().numpy()

                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(float)

                all_preds.append(preds)
                all_targets.append(targets)

        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_preds)
        return compute_classification_metrics(y_true, y_pred, self.target_names)

    def fit(self) -> nn.Module:
        for epoch in range(self.cfg.cls_epochs):
            loss = self.train_epoch()
            metrics = self.evaluate(self.val_loader)
            macro_f1 = metrics["macro_avg"]["f1"]
            macro_acc = metrics["macro_avg"]["accuracy"]
            print(f"[Classifier Epoch {epoch+1}/{self.cfg.cls_epochs}] Loss: {loss:.4f} | Val Acc: {macro_acc:.4f} | Val F1: {macro_f1:.4f}")
        return self.model