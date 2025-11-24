"""Simplified DDPM (Denoising Diffusion Probabilistic Model) for quick validation.

Implements core training (q_sample) and sampling (p_sample_loop) methods.
Simplified from gaussian_diffusion.py in DiT-SR.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class SimpleDiffusion:
    """Simplified Gaussian Diffusion for fast prototyping.

    Uses basic linear beta schedule and DDPM sampling (no DDIM).
    """

    def __init__(self, num_timesteps=4, beta_start=0.0001, beta_end=0.02, device='cuda'):
        """Initialize diffusion process.

        Args:
            num_timesteps: number of diffusion steps (small for fast verification)
            beta_start: minimum noise level
            beta_end: maximum noise level
            device: torch device
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # Precompute constants for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.tensor(
            np.sqrt(alphas_cumprod), dtype=torch.float32, device=device
        )
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32, device=device
        )

        # Precompute constants for reverse diffusion p(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.tensor(
            np.sqrt(1.0 / alphas), dtype=torch.float32, device=device
        )

        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = torch.tensor(
            posterior_variance, dtype=torch.float32, device=device
        )

        # Clip variance (avoid numerical issues at t=0)
        self.posterior_log_variance_clipped = torch.tensor(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32, device=device
        )

        # For computing mean of q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef1 = torch.tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=torch.float32, device=device
        )
        self.posterior_mean_coef2 = torch.tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=torch.float32, device=device
        )

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 to get x_t.

        Implements: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: [B, C, H, W] clean image
            t: [B] timestep indices (0 to num_timesteps-1)
            noise: [B, C, H, W] Gaussian noise (optional, will be sampled if None)

        Returns:
            [B, C, H, W] noisy image x_t
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Extract coefficients for batch
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def p_mean_variance(self, model, x_t, t, lq=None, clip_denoised=True):
        """Compute mean and variance of p(x_{t-1} | x_t).

        Args:
            model: denoising network
            x_t: [B, C, H, W] noisy image at timestep t
            t: [B] timestep indices
            lq: [B, C, H_lq, W_lq] low-res condition (optional)
            clip_denoised: whether to clip predicted x0 to [-1, 1]

        Returns:
            dict with 'mean' and 'variance' keys
        """
        # Predict x0 from x_t
        x0_pred = model(x_t, t, lq)

        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Compute posterior mean: mu_theta(x_t, t) using predicted x0
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t

        # Posterior variance (fixed)
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return {
            'mean': mean,
            'variance': variance,
            'log_variance': log_variance,
            'x0_pred': x0_pred
        }

    def p_sample(self, model, x_t, t, lq=None, clip_denoised=True):
        """Single reverse diffusion step: sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model: denoising network
            x_t: [B, C, H, W] noisy image at timestep t
            t: [B] timestep indices
            lq: [B, C, H_lq, W_lq] low-res condition (optional)
            clip_denoised: whether to clip predicted x0

        Returns:
            [B, C, H, W] image at timestep t-1
        """
        out = self.p_mean_variance(model, x_t, t, lq, clip_denoised)
        mean = out['mean']
        variance = out['variance']

        # Add noise (except at t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  # [B, 1, 1, 1]
        sample = mean + nonzero_mask * torch.sqrt(variance) * noise

        return sample

    def p_sample_loop(self, model, shape, lq=None, progress=True, clip_denoised=True):
        """Full sampling loop: x_T → x_{T-1} → ... → x_0.

        Args:
            model: denoising network
            shape: tuple (B, C, H, W) for output
            lq: [B, C, H_lq, W_lq] low-res condition (optional)
            progress: whether to show progress bar
            clip_denoised: whether to clip predicted x0

        Returns:
            [B, C, H, W] sampled clean image
        """
        # Get device from model parameters or use self.device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = self.device  # Fallback to diffusion's device
        b = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Iteratively denoise
        timesteps = reversed(range(self.num_timesteps))
        if progress:
            timesteps = tqdm(timesteps, desc='DDPM Sampling', total=self.num_timesteps)

        for i in timesteps:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, lq, clip_denoised)

        return x

    def _extract(self, coef, t, x_shape):
        """Extract coefficients at timestep t and reshape to broadcast with x.

        Args:
            coef: [num_timesteps] precomputed coefficients
            t: [B] timestep indices
            x_shape: tuple (B, C, H, W)

        Returns:
            [B, 1, 1, 1] extracted coefficients
        """
        out = coef.gather(-1, t)
        # Reshape to [B, 1, 1, 1] for broadcasting
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def create_diffusion(num_timesteps=4, beta_start=0.0001, beta_end=0.02, device='cuda'):
    """Factory function to create SimpleDiffusion instance."""
    return SimpleDiffusion(num_timesteps, beta_start, beta_end, device)


if __name__ == '__main__':
    # Unit test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create diffusion process
    diffusion = create_diffusion(num_timesteps=4, device=device)

    # Test forward diffusion (add noise)
    x0 = torch.randn(2, 2, 64, 64).to(device)
    t = torch.randint(0, 4, (2,)).to(device)
    x_t = diffusion.q_sample(x0, t)

    print(f'x0 shape: {x0.shape}, range: [{x0.min():.2f}, {x0.max():.2f}]')
    print(f'x_t shape: {x_t.shape}, range: [{x_t.min():.2f}, {x_t.max():.2f}]')

    # Test reverse diffusion (mock model)
    class MockModel(torch.nn.Module):
        def forward(self, x, t, lq=None):
            return x * 0.5  # Simple identity-like function

    model = MockModel().to(device)
    x_sample = diffusion.p_sample_loop(model, shape=(2, 2, 64, 64), progress=False)

    print(f'Sampled x shape: {x_sample.shape}')
    print('✅ SimpleDiffusion test passed!')
