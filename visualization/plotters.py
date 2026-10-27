import os
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn

def plot_wgan_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history["critic_loss"], label="Critic Loss", color="tab:red")
    axes[0].plot(history["gen_loss"], label="Generator Loss", color="tab:blue")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("WGAN Training Losses")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["wasserstein_dist"], label="Wasserstein Distance", color="tab:green")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("W-Distance")
    axes[1].set_title("Estimated Convergence")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_quality_vs_latent_dim(dims: List[int], fid_values: List[float], is_values: List[float], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Размерность z_dim')
    ax1.set_ylabel('FID (ниже - лучше)', color=color)
    ax1.plot(dims, fid_values, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Inception Score (выше - лучше)', color=color)
    ax2.plot(dims, is_values, marker='s', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Качество генерации в зависимости от z_dim")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, attr_names: List[str], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(attr_names), figsize=(4 * len(attr_names), 4))
    if len(attr_names) == 1:
        axes = [axes]

    for idx, name in enumerate(attr_names):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[idx].set_title(f"{name}")
        axes[idx].set_xlabel("Pred")
        axes[idx].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_trajectory_image_grid(images: np.ndarray, titles: List[str], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = len(images)
    fig, axes = plt.subplots(1, steps, figsize=(2.5 * steps, 3))

    for i in range(steps):
        img = (images[i] + 1.0) / 2.0
        axes[i].imshow(np.clip(np.transpose(img, (1, 2, 0)), 0, 1))
        axes[i].axis('off')
        if i < len(titles):
            axes[i].set_title(titles[i], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_nearest_neighbors_comparison(
    generated_img: torch.Tensor, 
    neighbor_img: torch.Tensor, 
    distance: float, 
    save_path: str
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    gen = (generated_img.cpu().squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
    nn_img = (neighbor_img.cpu().squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0

    axes[0].imshow(np.clip(gen, 0, 1))
    axes[0].set_title("Generated G(z)")
    axes[0].axis('off')

    axes[1].imshow(np.clip(nn_img, 0, 1))
    axes[1].set_title(f"Nearest Train Sample\nDist = {distance:.3f}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_semantic_editing_grid(
    images: np.ndarray, 
    alphas: List[float], 
    attr_name: str, 
    save_path: str
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(alphas), figsize=(3 * len(alphas), 3))

    for idx, a in enumerate(alphas):
        img = (images[idx] + 1.0) / 2.0
        axes[idx].imshow(np.clip(np.transpose(img, (1, 2, 0)), 0, 1))
        axes[idx].set_title(f"alpha = {a:+.1f}")
        axes[idx].axis('off')

    plt.suptitle(f"Семантическое редактирование: {attr_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compare_weight_distributions(critic_clipping: nn.Module, critic_gp: nn.Module, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    weights_clip = [p.detach().cpu().numpy().flatten() for n, p in critic_clipping.named_parameters() if 'weight' in n]
    weights_gp = [p.detach().cpu().numpy().flatten() for n, p in critic_gp.named_parameters() if 'weight' in n]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.concatenate(weights_clip), bins=80, color='crimson', range=[-0.02, 0.02])
    axes[0].set_title("Weight Clipping")
    axes[0].grid(True)

    axes[1].hist(np.concatenate(weights_gp), bins=80, color='royalblue')
    axes[1].set_title("Gradient Penalty")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()