import torch
import torch.nn as nn

def compute_gradient_penalty(
    critic: nn.Module, 
    real_samples: torch.Tensor, 
    fake_samples: torch.Tensor, 
    device: str = "cuda"
) -> torch.Tensor:
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    grad_outputs = torch.ones(d_interpolates.shape, device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
    return gradient_penalty

def critic_loss_wgan(critic_real: torch.Tensor, critic_fake: torch.Tensor) -> torch.Tensor:
    return -(torch.mean(critic_real) - torch.mean(critic_fake))

def generator_loss_wgan(critic_fake: torch.Tensor) -> torch.Tensor:
    return -torch.mean(critic_fake)