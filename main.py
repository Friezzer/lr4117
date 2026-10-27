import os
import argparse
import numpy as np
import torch

from configs.config import DatasetConfig, ModelConfig, TrainingConfig, LatentAnalysisConfig
from data.dataset import get_celeba_dataloaders
from models.generator import Generator
from models.critic import Critic
from models.classifier import AttributeClassifier
from training.trainer_wgan import WGANTrainer
from training.trainer_classifier import ClassifierTrainer
from latent_space.sampler import LatentSampler
from latent_space.directions import compute_difference_of_means, compute_svm_hyperplane_direction
from latent_space.trajectories import spherical_interpolation, apply_semantic_edit
from evaluation.nearest_neighbors import NearestNeighborFinder
from visualization.plotters import (
    plot_wgan_training_curves,
    plot_confusion_matrices,
    save_trajectory_image_grid,
    save_semantic_editing_grid,
    save_nearest_neighbors_comparison
)

def run_pipeline():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="WGAN Semantic Latent Editing Pipeline")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--exp_dir", type=str, default="./experiments")
    parser.add_argument("--weights_dir", type=str, default="./weights")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--gan_epochs", type=int, default=12)
    parser.add_argument("--cls_epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    cls_weights_path = os.path.join(args.weights_dir, "classifier.pth")
    gen_weights_path = os.path.join(args.weights_dir, "generator.pth")
    critic_weights_path = os.path.join(args.weights_dir, "critic.pth")

    data_cfg = DatasetConfig(data_root=args.data_root)
    model_cfg = ModelConfig(z_dim=args.z_dim)
    train_cfg = TrainingConfig(device=args.device, gan_epochs=args.gan_epochs, cls_epochs=args.cls_epochs)
    latent_cfg = LatentAnalysisConfig()

    print(f"Используемое устройство: {train_cfg.device}")

    # 1. Загрузка данных
    print("[1/6] Подготовка датасетов...")
    cls_train_loader, wgan_train_loader, val_loader, test_loader, target_names, target_indices = get_celeba_dataloaders(data_cfg)
    
    # 2. Инициализация и загрузка/обучение классификатора
    print("[2/6] Классификатор атрибутов...")
    classifier = AttributeClassifier(num_attributes=len(target_indices)).to(train_cfg.device)
    
    if os.path.exists(cls_weights_path):
        print(f"--> Найдены сохраненные веса классификатора. Загрузка из {cls_weights_path}...")
        classifier.load_state_dict(torch.load(cls_weights_path, map_location=train_cfg.device))
    else:
        print("--> Сохраненные веса не найдены. Старт обучения классификатора...")
        cls_trainer = ClassifierTrainer(classifier, cls_train_loader, val_loader, target_indices, target_names, train_cfg)
        classifier = cls_trainer.fit()
        torch.save(classifier.state_dict(), cls_weights_path)
        print(f"--> Веса классификатора сохранены в {cls_weights_path}")

    # 3. Инициализация и загрузка/обучение WGAN-GP
    print("[3/6] Модель WGAN-GP...")
    generator = Generator(z_dim=model_cfg.z_dim).to(train_cfg.device)
    critic = Critic().to(train_cfg.device)

    if os.path.exists(gen_weights_path) and os.path.exists(critic_weights_path):
        print(f"--> Найдены сохраненные веса WGAN. Загрузка из {args.weights_dir}...")
        generator.load_state_dict(torch.load(gen_weights_path, map_location=train_cfg.device))
        critic.load_state_dict(torch.load(critic_weights_path, map_location=train_cfg.device))
    else:
        print("--> Сохраненные веса не найдены. Старт обучения WGAN-GP...")
        wgan_trainer = WGANTrainer(generator, critic, wgan_train_loader, train_cfg, model_cfg)
        history = wgan_trainer.fit()
        torch.save(generator.state_dict(), gen_weights_path)
        torch.save(critic.state_dict(), critic_weights_path)
        print(f"--> Веса WGAN-GP сохранены в {args.weights_dir}")
        plot_wgan_training_curves(history, os.path.join(args.exp_dir, "wgan_losses.png"))

    # 4. Анализ латентного пространства
    print("[4/6] Сэмплирование латентного датасета и разметка классификатором...")
    sampler = LatentSampler(generator, classifier, device=train_cfg.device)
    latent_z, labels, confidences = sampler.sample_and_annotate(
        num_samples=latent_cfg.num_samples, 
        z_dim=model_cfg.z_dim
    )

    plot_confusion_matrices(labels, (confidences >= 0.5).astype(int), target_names, os.path.join(args.exp_dir, "confusion_matrices.png"))

    # 5. Семантическое редактирование и интерполяция
    print("[5/6] Поиск семантических направлений и интерполяция...")
    generator.eval()
    for attr_idx, attr_name in enumerate(target_names):
        v_norm, z_0, z_1 = compute_difference_of_means(latent_z, confidences, attr_idx, latent_cfg.top_percent_confidence)
        
        traj_vectors = spherical_interpolation(z_0, z_1, steps=latent_cfg.trajectory_steps)
        with torch.no_grad():
            t_tensor = torch.tensor(traj_vectors, dtype=torch.float32, device=train_cfg.device)
            traj_imgs = generator(t_tensor).cpu().numpy()

        titles = [f"t={i/(latent_cfg.trajectory_steps-1):.2f}" for i in range(latent_cfg.trajectory_steps)]
        save_trajectory_image_grid(traj_imgs, titles, os.path.join(args.exp_dir, f"traj_{attr_name}.png"))

        sample_z = np.random.randn(model_cfg.z_dim)
        edited_vectors = apply_semantic_edit(sample_z, v_norm, latent_cfg.alphas_editing)
        with torch.no_grad():
            e_tensor = torch.tensor(edited_vectors, dtype=torch.float32, device=train_cfg.device)
            edited_imgs = generator(e_tensor).cpu().numpy()

        save_semantic_editing_grid(edited_imgs, latent_cfg.alphas_editing, attr_name, os.path.join(args.exp_dir, f"edit_{attr_name}.png"))

    # 6. Проверка на запоминание обучающей выборки
    print("[6/6] Поиск ближайших обучающих соседей (Memorization Check)...")
    nn_finder = NearestNeighborFinder(wgan_train_loader, device=train_cfg.device)
    test_z = torch.randn(1, model_cfg.z_dim, 1, 1, device=train_cfg.device)
    with torch.no_grad():
        test_img = generator(test_z)
    
    matches, dists = nn_finder.find_perceptual_neighbors(test_img.squeeze(0), top_k=1)
    save_nearest_neighbors_comparison(test_img, matches[0], dists[0], os.path.join(args.exp_dir, "nearest_neighbor.png"))

    print(f"Эксперименты завершены. Все артефакты сохранены в: {args.exp_dir}")

if __name__ == "__main__":
    run_pipeline()