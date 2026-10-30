"""训练入口（在项目根目录运行）。

阶段一（VAE 重建）：  python -m helios.train --config configs/helios_tiny.yaml --stage vae
阶段二（DiT 扩散）：  python -m helios.train --config configs/helios_tiny.yaml --stage dit
"""

import argparse
import itertools
import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from helios.data.dataset import ImageDataset, VideoDataset
from helios.diffusion.flow import flow_loss
from helios.models import ByteTextEncoder, HeliosDiT, TinyVAE
from helios.models.text_encoder import PAD_ID, encode_text
from helios.models.vae import vae_loss
from helios.utils import get_lr, load_config, save_checkpoint, set_seed


def build_loaders(cfg):
    t = cfg["train"]
    loaders = {}
    if os.path.isdir(t["image_data"]):
        try:
            ds = ImageDataset(t["image_data"], t["image_size"])
            loaders["image"] = DataLoader(ds, batch_size=t["batch_size"], shuffle=True,
                                          num_workers=t["num_workers"], drop_last=True)
        except RuntimeError:
            pass
    if os.path.isdir(t["video_data"]):
        try:
            ds = VideoDataset(t["video_data"], t["image_size"], t["num_frames"])
            loaders["video"] = DataLoader(ds, batch_size=t["batch_size"], shuffle=True,
                                          num_workers=t["num_workers"], drop_last=True)
        except RuntimeError:
            pass
    if not loaders:
        raise RuntimeError(
            f"没有找到训练数据，请把图片放进 {t['image_data']} 或视频放进 {t['video_data']}")
    return loaders


def cycle(loader):
    return itertools.cycle(loader)


def encode_latents(vae, x, scale):
    """x: [B,3,T,H,W] -> z: [B,C,T,H/8,W/8]（逐帧编码，取均值不加噪）"""
    B, _, T, H, W = x.shape
    with torch.no_grad():
        mean, _ = vae.encode(x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W))
    z = mean.reshape(B, T, vae.latent_channels, H // 8, W // 8).permute(0, 2, 1, 3, 4)
    return z * scale


def encode_captions(text_enc, captions, max_len, device, dropout=0.0):
    caps = ["" if random.random() < dropout else c for c in captions]
    ids = torch.tensor([encode_text(c, max_len) for c in caps], device=device)
    return text_enc(ids), ids == PAD_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["vae", "dit"], required=True)
    ap.add_argument("--resume", default=None, help="latest.pt 路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    t = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    loaders = build_loaders(cfg)
    print(f"[{cfg['model_name']}] stage={args.stage} device={device} data={list(loaders)}")

    vae = TinyVAE(**cfg["vae"]).to(device)
    text_enc = ByteTextEncoder(**cfg["text"]).to(device)
    dit = HeliosDiT(text_dim=cfg["text"]["dim"], **cfg["dit"]).to(device)

    start = 0
    if args.stage == "vae":
        params = list(vae.parameters())
    else:
        vae.requires_grad_(False).eval()
        params = list(dit.parameters()) + list(text_enc.parameters())
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["vae"])
        if args.stage == "dit" and ckpt.get("dit") is not None:
            dit.load_state_dict(ckpt["dit"])
            text_enc.load_state_dict(ckpt["text"])
        start = ckpt.get("step", 0)
        print(f"已从 {args.resume} 恢复 (step={start})")

    img_it = cycle(loaders["image"]) if "image" in loaders else None
    vid_it = cycle(loaders["video"]) if "video" in loaders else None
    pbar = tqdm(range(start, t["steps"]), initial=start, total=t["steps"])

    for step in pbar:
        lr = get_lr(step, t["lr"], t["warmup_steps"], t["steps"])
        for g in opt.param_groups:
            g["lr"] = lr

        # 每个 step 按 p_video 选一种模态，保持 batch 内形状一致
        use_video = vid_it is not None and (img_it is None or random.random() < t["p_video"])
        x, captions = next(vid_it if use_video else img_it)
        x = x.to(device, non_blocking=True)  # [B,3,T,S,S]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if args.stage == "vae":
                B, _, T, H, W = x.shape
                frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
                recon, mean, logvar = vae(frames)
                loss, parts = vae_loss(frames, recon, mean, logvar, t["kl_weight"])
            else:
                z = encode_latents(vae, x, cfg["latent_scale"])
                ctx, mask = encode_captions(text_enc, captions, cfg["text"]["max_len"],
                                            device, t["caption_dropout"])
                loss = flow_loss(dit, z, ctx, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
        opt.step()

        if (step + 1) % t["log_every"] == 0:
            extra = f" l1={parts['l1']:.4f} kl={parts['kl']:.4f}" if args.stage == "vae" else ""
            pbar.set_postfix_str(f"loss={loss.item():.4f}{extra} lr={lr:.2e}")

        if (step + 1) % t["save_every"] == 0 or (step + 1) == t["steps"]:
            state = {
                "step": step + 1,
                "config": cfg,
                "vae": vae.state_dict(),
                "text": text_enc.state_dict() if args.stage == "dit" else None,
                "dit": dit.state_dict() if args.stage == "dit" else None,
            }
            save_checkpoint(os.path.join(t["out_dir"], f"ckpt_{args.stage}_{step + 1}.pt"), state)
            save_checkpoint(os.path.join(t["out_dir"], f"latest_{args.stage}.pt"), state)

    print(f"训练完成，checkpoint 保存在 {t['out_dir']}")


if __name__ == "__main__":
    main()
