from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DatasetConfig:
    data_root: str = "./data"
    image_size: int = 64
    crop_size: int = 140
    batch_size: int = 64
    num_workers: int = 4
    wgan_max_samples: int = 10000 # Ограничение для WGAN
    target_attributes: List[str] = field(default_factory=lambda: ["Smiling", "Male", "Eyeglasses", "Young"])

@dataclass
class ModelConfig:
    z_dim: int = 128
    channels_img: int = 3
    features_g: int = 64
    features_d: int = 64
    norm_critic: str = "instance"  # "instance", "layer", "none"

@dataclass
class TrainingConfig:
    device: str = "cuda"
    gan_epochs: int = 12
    cls_epochs: int = 3
    lr_gan: float = 2e-4
    betas_gan: Tuple[float, float] = (0.0, 0.9)
    lr_cls: float = 1e-4
    lambda_gp: float = 10.0
    critic_iterations: int = 3
    weight_clip_value: float = 0.01
    method_lipschitz: str = "gradient_penalty"  # "gradient_penalty" или "weight_clipping"

@dataclass
class LatentAnalysisConfig:
    num_samples: int = 5000
    top_percent_confidence: float = 0.20
    trajectory_steps: int = 8
    alphas_editing: List[float] = field(default_factory=lambda: [-3.0, -1.5, 0.0, 1.5, 3.0])