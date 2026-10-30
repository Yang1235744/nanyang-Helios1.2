"""字节级文本编码器 —— 不依赖任何外部 tokenizer / 预训练权重，真正从零训练。

词表 = 256 个 UTF-8 字节 + PAD/BOS/EOS，天然支持中英文混合 prompt。
"""

import torch
import torch.nn as nn

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258
VOCAB_SIZE = 259


def encode_text(text: str, max_len: int) -> list[int]:
    ids = [BOS_ID] + list(text.encode("utf-8"))[: max_len - 2] + [EOS_ID]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


class ByteTextEncoder(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 4, heads: int = 4, max_len: int = 128):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(VOCAB_SIZE, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, depth)
        self.final = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] -> [B, L, dim]；padding 位置由 mask 屏蔽。"""
        mask = ids == PAD_ID
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.tok(ids) + self.pos(pos)
        return self.final(self.enc(h, src_key_padding_mask=mask))
