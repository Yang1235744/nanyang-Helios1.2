"""Rectified Flow：x_t = (1-t)·x0 + t·ε，模型预测速度场 v = ε - x0。

t=1 为纯噪声，t=0 为数据；采样用 Euler 从 t=1 积分到 t=0。
"""

import torch
import torch.nn.functional as F


def flow_loss(model, x0: torch.Tensor, ctx: torch.Tensor, ctx_mask=None, t: torch.Tensor = None) -> torch.Tensor:
    b = x0.shape[0]
    if t is None:
        t = torch.rand(b, device=x0.device)
    eps = torch.randn_like(x0)
    t5 = t.view(b, *([1] * (x0.ndim - 1)))
    xt = (1.0 - t5) * x0 + t5 * eps
    v = model(xt, t, ctx, ctx_mask)
    return F.mse_loss(v, eps - x0)


@torch.no_grad()
def euler_sample(model, ctx_cond, mask_cond, ctx_uncond, mask_uncond,
                 shape, steps: int = 30, cfg_scale: float = 4.0, device="cpu") -> torch.Tensor:
    """ctx_* 需与 shape[0] batch 对齐。cfg_scale=1 时退化为普通条件采样。"""
    model.eval()
    x = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = torch.full((shape[0],), ts[i].item(), device=device)
        v = model(x, t, ctx_cond, mask_cond)
        if cfg_scale != 1.0:
            v_u = model(x, t, ctx_uncond, mask_uncond)
            v = v_u + cfg_scale * (v - v_u)
        x = x + (ts[i + 1] - ts[i]).item() * v
    return x
