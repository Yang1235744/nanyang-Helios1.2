"""采样入口（在项目根目录运行）。

图片：python -m helios.sample --ckpt runs/helios_tiny/latest_dit.pt --prompt "一只在雪地里的红狐"
视频：python -m helios.sample --ckpt runs/helios_tiny/latest_dit.pt --prompt "海浪拍打礁石" --type video --out out.gif
"""

import argparse

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from helios.diffusion.flow import euler_sample
from helios.models import ByteTextEncoder, HeliosDiT, TinyVAE
from helios.models.text_encoder import PAD_ID, encode_text


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """[B*T,3,H,W] in [-1,1] -> [B*T,H,W,3] uint8"""
    x = (frames.clamp(-1, 1) + 1.0) * 127.5
    return x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--neg-prompt", default="")
    ap.add_argument("--type", choices=["image", "video"], default="image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=None, help="视频帧数，默认取训练配置")
    ap.add_argument("--size", type=int, default=None, help="边长，默认取训练配置")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg-scale", type=float, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    assert ckpt.get("dit") is not None, "该 checkpoint 没有 DiT 权重（先用 --stage dit 训练）"

    vae = TinyVAE(**cfg["vae"]).to(device).eval()
    text_enc = ByteTextEncoder(**cfg["text"]).to(device).eval()
    dit = HeliosDiT(text_dim=cfg["text"]["dim"], **cfg["dit"]).to(device).eval()
    vae.load_state_dict(ckpt["vae"])
    text_enc.load_state_dict(ckpt["text"])
    dit.load_state_dict(ckpt["dit"])
    vae.requires_grad_(False)

    size = args.size or cfg["train"]["image_size"]
    frames = 1 if args.type == "image" else (args.frames or cfg["train"]["num_frames"])
    steps = args.steps or cfg["sample"]["steps"]
    cfg_scale = args.cfg_scale or cfg["sample"]["cfg_scale"]
    out = args.out or ("sample.png" if args.type == "image" else "sample.gif")

    max_len = cfg["text"]["max_len"]
    ids_c = torch.tensor([encode_text(args.prompt, max_len)], device=device)
    ids_u = torch.tensor([encode_text(args.neg_prompt, max_len)], device=device)
    ctx_c, ctx_u = text_enc(ids_c), text_enc(ids_u)

    h = w = size // vae.downsample_factor
    z = euler_sample(dit, ctx_c, ids_c == PAD_ID, ctx_u, ids_u == PAD_ID,
                     shape=(1, cfg["vae"]["latent_channels"], frames, h, w),
                     steps=steps, cfg_scale=cfg_scale, device=device)
    z = z / cfg["latent_scale"]

    pixels = vae.decode(z.permute(0, 2, 1, 3, 4).reshape(frames, -1, h, w))
    arr = to_uint8(pixels)
    if args.type == "image":
        Image.fromarray(arr[0]).save(out)
    else:
        iio.imwrite(out, arr, duration=int(1000 / cfg["sample"]["fps"]), loop=0)
    print(f"已保存: {out} (prompt={args.prompt!r}, steps={steps}, cfg={cfg_scale})")


if __name__ == "__main__":
    main()
