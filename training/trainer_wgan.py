from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs.config import TrainingConfig, ModelConfig
from training.losses import compute_gradient_penalty, critic_loss_wgan, generator_loss_wgan

class WGANTrainer:
    def __init__(
        self, 
        generator: nn.Module, 
        critic: nn.Module, 
        train_loader: DataLoader, 
        train_config: TrainingConfig,
        model_config: ModelConfig
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.train_loader = train_loader
        self.cfg = train_config
        self.model_cfg = model_config
        self.device = self.cfg.device

        if self.cfg.method_lipschitz == "gradient_penalty":
            self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.lr_gan, betas=self.cfg.betas_gan)
            self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_gan, betas=self.cfg.betas_gan)
        else:
            self.opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=5e-5)
            self.opt_c = torch.optim.RMSprop(self.critic.parameters(), lr=5e-5)

    def train_epoch_gp(self, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.critic.train()
        total_loss_c, total_loss_g, total_w_dist = 0.0, 0.0, 0.0
        batches = len(self.train_loader)

        for batch_idx, (real, _) in enumerate(self.train_loader):
            real = real.to(self.device)
            b_size = real.shape[0]

            for _ in range(self.cfg.critic_iterations):
                noise = torch.randn(b_size, self.model_cfg.z_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                
                c_real = self.critic(real)
                c_fake = self.critic(fake.detach())
                gp = compute_gradient_penalty(self.critic, real, fake.detach(), device=self.device)
                loss_c = critic_loss_wgan(c_real, c_fake) + self.cfg.lambda_gp * gp
                
                self.opt_c.zero_grad()
                loss_c.backward()
                self.opt_c.step()

            noise = torch.randn(b_size, self.model_cfg.z_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            gen_fake = self.critic(fake)
            loss_g = generator_loss_wgan(gen_fake)

            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()

            w_dist = (torch.mean(c_real) - torch.mean(c_fake)).item()
            total_loss_c += loss_c.item()
            total_loss_g += loss_g.item()
            total_w_dist += w_dist

        return {
            "critic_loss": total_loss_c / batches,
            "gen_loss": total_loss_g / batches,
            "w_dist": total_w_dist / batches
        }

    def train_epoch_clipping(self, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.critic.train()
        total_loss_c, total_loss_g = 0.0, 0.0
        batches = len(self.train_loader)

        for batch_idx, (real, _) in enumerate(self.train_loader):
            real = real.to(self.device)
            b_size = real.shape[0]

            for _ in range(self.cfg.critic_iterations):
                noise = torch.randn(b_size, self.model_cfg.z_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                c_real = self.critic(real)
                c_fake = self.critic(fake.detach())
                loss_c = critic_loss_wgan(c_real, c_fake)

                self.opt_c.zero_grad()
                loss_c.backward()
                self.opt_c.step()

                for p in self.critic.parameters():
                    p.data.clamp_(-self.cfg.weight_clip_value, self.cfg.weight_clip_value)

            noise = torch.randn(b_size, self.model_cfg.z_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            loss_g = generator_loss_wgan(self.critic(fake))

            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()

            total_loss_c += loss_c.item()
            total_loss_g += loss_g.item()

        return {
            "critic_loss": total_loss_c / batches,
            "gen_loss": total_loss_g / batches,
            "w_dist": 0.0
        }

    def fit(self) -> Dict[str, List[float]]:
        history = {"critic_loss": [], "gen_loss": [], "wasserstein_dist": []}
        for epoch in range(self.cfg.gan_epochs):
            if self.cfg.method_lipschitz == "gradient_penalty":
                res = self.train_epoch_gp(epoch)
            else:
                res = self.train_epoch_clipping(epoch)
                
            history["critic_loss"].append(res["critic_loss"])
            history["gen_loss"].append(res["gen_loss"])
            history["wasserstein_dist"].append(res["w_dist"])
            print(f"[GAN Epoch {epoch+1}/{self.cfg.gan_epochs}] Loss C: {res['critic_loss']:.4f} | Loss G: {res['gen_loss']:.4f} | W-Dist: {res['w_dist']:.4f}")
        return history