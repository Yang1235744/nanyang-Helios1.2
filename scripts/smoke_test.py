"""端到端冒烟测试：随机数据跑通 VAE -> 文本编码 -> DiT -> flow loss -> 采样 -> 解码。

运行（项目根目录）：python scripts/smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from helios.diffusion.flow import euler_sample, flow_loss
from helios.models import ByteTextEncoder, HeliosDiT, TinyVAE
from helios.models.text_encoder import PAD_ID, encode_text


def main():
    torch.manual_seed(0)

    # 1) VAE 前向 + 重建损失 + 反传
    vae = TinyVAE(latent_channels=4, base_channels=16)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    recon, mean, logvar = vae(x)
    assert recon.shape == x.shape and mean.shape == (2, 4, 4, 4)
    loss = F.l1_loss(recon, x) - 1e-6 * 0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss.backward()
    print(f"[1] VAE ok, recon loss={loss.item():.4f}")

    # 2) 文本编码（含中文）
    text = ByteTextEncoder(dim=32, depth=2, heads=2, max_len=16)
    ids = torch.tensor([encode_text("a red circle 红色圆形", 16), encode_text("", 16)])
    ctx = text(ids)
    mask = ids == PAD_ID
    assert ctx.shape == (2, 16, 32)
    print(f"[2] ByteTextEncoder ok, ctx={tuple(ctx.shape)}")

    # 3) DiT + flow loss + 反传（T=2 视频 latent）
    dit = HeliosDiT(in_channels=4, dim=64, depth=2, heads=2, patch_size=2,
                    text_dim=32, max_t=4, max_h=4, max_w=4)
    z = torch.randn(2, 4, 2, 4, 4)
    loss = flow_loss(dit, z, ctx, mask)
    loss.backward()
    print(f"[3] HeliosDiT ok, flow loss={loss.item():.4f}")

    # 4) Euler 采样（T=1 图片）+ VAE 逐帧解码
    with torch.no_grad():
        z_img = euler_sample(dit, ctx[:1], mask[:1], ctx[1:], mask[1:],
                             shape=(1, 4, 1, 4, 4), steps=3, cfg_scale=2.0)
        px = vae.decode(z_img[:, :, 0])
    assert px.shape == (1, 3, 32, 32)
    print(f"[4] euler_sample + decode ok, out={tuple(px.shape)}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
