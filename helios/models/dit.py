"""HeliosDiT —— 图片/视频统一的 Diffusion Transformer 主干。

latent [B,C,T,H,W]（图片 T=1）按空间 patch 切 token，时间维不打 patch；
全量时空自注意力 + 文本交叉注意力 + adaLN-Zero 时间步调制。
早期版本为统一全注意力，后续可换时空分解注意力省显存。
"""

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """正弦时间步嵌入，t ∈ [0,1]，1 为纯噪声。"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def patchify(z: torch.Tensor, p: int) -> torch.Tensor:
    """[B,C,T,H,W] -> [B, T*h*w, C*p*p]"""
    B, C, T, H, W = z.shape
    h, w = H // p, W // p
    z = z.reshape(B, C, T, h, p, w, p).permute(0, 2, 3, 5, 1, 4, 6)
    return z.reshape(B, T * h * w, C * p * p)


def unpatchify(tokens: torch.Tensor, p: int, C: int, T: int, H: int, W: int) -> torch.Tensor:
    """[B, T*h*w, C*p*p] -> [B,C,T,H,W]"""
    B = tokens.shape[0]
    h, w = H // p, W // p
    x = tokens.reshape(B, T, h, w, C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
    return x.reshape(B, C, T, H, W)


class Block(nn.Module):
    """adaLN-Zero: 自注意力 + 文本交叉注意力 + MLP，9 路调制。"""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, c, ctx, ctx_mask=None):
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.mod(c).chunk(9, dim=-1)
        h = modulate(self.norm1(x), s1, c1)
        x = x + g1.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), s2, c2)
        x = x + g2.unsqueeze(1) * self.cross(h, ctx, ctx, key_padding_mask=ctx_mask, need_weights=False)[0]
        h = modulate(self.norm3(x), s3, c3)
        x = x + g3.unsqueeze(1) * self.mlp(h)
        return x


class HeliosDiT(nn.Module):
    def __init__(self, in_channels: int = 4, dim: int = 384, depth: int = 8, heads: int = 6,
                 patch_size: int = 2, text_dim: int = 256, max_t: int = 16,
                 max_h: int = 16, max_w: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.p = patch_size
        self.dim = dim
        self.patch_in = nn.Linear(in_channels * self.p * self.p, dim)
        self.pos_t = nn.Parameter(torch.zeros(max_t, dim))
        self.pos_h = nn.Parameter(torch.zeros(max_h, dim))
        self.pos_w = nn.Parameter(torch.zeros(max_w, dim))
        for pos in (self.pos_t, self.pos_h, self.pos_w):
            nn.init.trunc_normal_(pos, std=0.02)

        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ctx_proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.final_proj = nn.Linear(dim, in_channels * self.p * self.p)
        nn.init.zeros_(self.final_mod[1].weight)
        nn.init.zeros_(self.final_mod[1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, ctx_mask=None) -> torch.Tensor:
        """z: [B,C,T,H,W]，t: [B] ∈ [0,1]，ctx: [B,L,text_dim] -> 速度场 [B,C,T,H,W]"""
        B, C, T, H, W = z.shape
        h, w = H // self.p, W // self.p
        assert T <= self.pos_t.shape[0] and h <= self.pos_h.shape[0] and w <= self.pos_w.shape[0], \
            f"输入尺寸 T={T}, grid={h}x{w} 超出位置表上限"

        x = self.patch_in(patchify(z, self.p))
        pos = (self.pos_t[:T, None, None, :] + self.pos_h[None, :h, None, :] + self.pos_w[None, None, :w, :])
        x = x + pos.reshape(1, T * h * w, self.dim)

        c = self.t_mlp(timestep_embedding(t, self.dim))
        ctx = self.ctx_proj(ctx)
        for blk in self.blocks:
            x = blk(x, c, ctx, ctx_mask)

        s, sc = self.final_mod(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), s, sc)
        return unpatchify(self.final_proj(x), self.p, C, T, H, W)
