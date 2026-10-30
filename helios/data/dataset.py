"""数据集：图片目录 + 视频目录。

- 图片返回 [3, 1, S, S]（视为单帧视频），视频返回 [3, T, S, S]，值域 [-1, 1]
- caption 取同名 .txt（如 a.png -> a.txt），没有则为空字符串
"""

import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".gif", ".mov", ".webm", ".avi", ".mkv"}


def _list_files(root: str, exts: set) -> list[str]:
    out = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in exts:
                out.append(os.path.join(dirpath, n))
    return out


def _load_caption(path: str) -> str:
    txt = os.path.splitext(path)[0] + ".txt"
    if os.path.isfile(txt):
        with open(txt, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def _center_crop_square(x: torch.Tensor) -> torch.Tensor:
    """x: [..., H, W] 对最后两维中心裁剪成正方形。"""
    h, w = x.shape[-2], x.shape[-1]
    m = min(h, w)
    top, left = (h - m) // 2, (w - m) // 2
    return x[..., top:top + m, left:left + m]


class ImageDataset(Dataset):
    def __init__(self, root: str, size: int):
        self.files = _list_files(root, IMG_EXT)
        self.size = size
        if not self.files:
            raise RuntimeError(f"图片目录为空: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        arr = _center_crop_square(arr)
        arr = F.interpolate(arr.unsqueeze(0), size=(self.size, self.size),
                            mode="bilinear", align_corners=False).squeeze(0)
        return arr.unsqueeze(1) * 2.0 - 1.0, _load_caption(path)  # [3,1,S,S]


class VideoDataset(Dataset):
    def __init__(self, root: str, size: int, num_frames: int):
        self.files = _list_files(root, VID_EXT)
        self.size = size
        self.num_frames = num_frames
        if not self.files:
            raise RuntimeError(f"视频目录为空: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        frames = iio.imread(path)  # [T,H,W,C] uint8
        if frames.ndim == 3:  # 单帧兜底
            frames = frames[None]
        idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        x = torch.from_numpy(np.ascontiguousarray(frames[idx])).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]
        x = _center_crop_square(x)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return x.permute(1, 0, 2, 3) * 2.0 - 1.0, _load_caption(path)  # [3,T,S,S]
