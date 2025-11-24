"""Minimal DiT (Diffusion Transformer) for quick validation.

Simplified from DiT-SR for fast prototyping on 2-channel flow field data.
"""

import math
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: [B] tensor of timestep indices
        dim: embedding dimension
        max_period: maximum period for sinusoidal encoding

    Returns:
        [B, dim] tensor of positional embeddings
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiTBlock(nn.Module):
    """Simplified DiT Transformer block with AdaLN-Zero modulation.

    Uses standard Multi-Head Attention instead of Swin for simplicity.
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

        # AdaLN-Zero: modulate with timestep conditioning
        # Outputs 6 * dim: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, c):
        """Forward pass with timestep conditioning.

        Args:
            x: [B, N, C] input features
            c: [B, C] timestep conditioning embedding

        Returns:
            [B, N, C] output features
        """
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention with modulation
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with modulation
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


@ARCH_REGISTRY.register()
class DiTMinimal(nn.Module):
    """Minimal DiT for fast validation on 2-channel data.

    Key simplifications:
    - Standard attention (not Swin)
    - Fewer layers (3 vs 6+)
    - Smaller embedding dim (96 vs 160+)
    - Direct pixel-space diffusion (no VAE)
    """

    def __init__(
        self,
        in_channels=2,
        img_size=256,          # Can be tuple (H, W) or single int
        patch_size=16,
        embed_dim=96,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        cond_lq=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.cond_lq = cond_lq

        # Support both square and rectangular images
        if isinstance(img_size, (tuple, list)):
            self.img_size_h, self.img_size_w = img_size
        else:
            self.img_size_h = self.img_size_w = img_size

        num_patches = (self.img_size_h // patch_size) * (self.img_size_w // patch_size)

        # Input projection: image → patch embeddings
        self.x_embedder = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Timestep embedding network
        self.t_embedder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # LQ condition embedder (optional)
        if cond_lq:
            self.lq_embedder = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer: normalize + project to pixels
        self.final_layer = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(embed_dim, patch_size * patch_size * in_channels, bias=True)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following DiT paper."""
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embedding conv
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)

        if self.cond_lq:
            nn.init.xavier_uniform_(self.lq_embedder.weight)
            nn.init.constant_(self.lq_embedder.bias, 0)

        # Zero-initialize adaLN modulation (critical for stability)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-initialize final layer (start from identity mapping)
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def unpatchify(self, x):
        """Convert patches back to image.

        Args:
            x: [B, N, patch_size^2 * C] patches

        Returns:
            [B, C, H, W] image
        """
        p = self.patch_size
        h = self.img_size_h // p
        w = self.img_size_w // p
        c = self.in_channels

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, h, p, w, p]
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward(self, x, t, lq=None):
        """Forward diffusion model.

        Args:
            x: [B, C, H, W] noisy image
            t: [B] timestep indices
            lq: [B, C, H_lq, W_lq] low-res condition (optional, will be upsampled)

        Returns:
            [B, C, H, W] predicted clean image
        """
        B = x.shape[0]

        # 1. Patchify input
        x = self.x_embedder(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # 2. Add positional embedding
        x = x + self.pos_embed

        # 3. Timestep conditioning
        t_emb = timestep_embedding(t, self.embed_dim)  # [B, embed_dim]
        t_emb = self.t_embedder(t_emb)  # [B, embed_dim]

        # 4. Optional: fuse LQ condition
        if self.cond_lq and lq is not None:
            # Upsample LQ to match GT resolution
            target_size = (self.img_size_h, self.img_size_w)
            if lq.shape[2:] != target_size:
                lq = nn.functional.interpolate(lq, size=target_size, mode='bilinear', align_corners=False)

            lq_emb = self.lq_embedder(lq).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            x = x + lq_emb  # Simple additive fusion

        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # 6. Final projection
        x = self.final_layer(x)  # [B, num_patches, patch_size^2 * C]

        # 7. Unpatchify to image
        x = self.unpatchify(x)  # [B, C, H, W]

        return x


if __name__ == '__main__':
    # Unit test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DiTMinimal(
        in_channels=2,
        img_size=256,
        patch_size=16,
        embed_dim=96,
        depth=3,
        num_heads=8,
        cond_lq=True
    ).to(device)

    x = torch.randn(2, 2, 256, 256).to(device)
    t = torch.randint(0, 4, (2,)).to(device)
    lq = torch.randn(2, 2, 64, 64).to(device)

    with torch.no_grad():
        out = model(x, t, lq)

    print(f'Input shape: {x.shape}')
    print(f'Output shape: {out.shape}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print('✅ DiTMinimal test passed!')
