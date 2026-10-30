# nanyang-Helios1.2

从零搭建的统一**图片 / 视频生成模型**（早期开源版 `v1.2.0a0`）。不依赖任何预训练权重：文本编码器、VAE、Diffusion Transformer 全部自研自训。

- 统一 DiT：图片 = 单帧视频（T=1），同一模型处理两种模态
- 字节级文本编码器：UTF-8 tokenizer（词表 259），中英文 prompt 原生支持
- Rectified Flow + Euler 采样 + CFG
- 两阶段从零训练：先 VAE 重建，再冻结 VAE 训扩散
- CPU 可复现全流程，GPU 开箱即用

## 安装

```bash
git clone https://github.com/Yang1235744/nanyang-Helios1.2.git
cd nanyang-Helios1.2
pip install -r requirements.txt
```

## 数据

```
data/images/   # 图片（jpg/png/webp...），同名 .txt 为 caption（可选）
data/videos/   # 视频（mp4/gif/...），同名 .txt 为 caption（可选）
```

## 训练

```bash
# 阶段一：VAE 重建
python -m helios.train --config configs/helios_tiny.yaml --stage vae
# 阶段二：冻结 VAE，训练 DiT + 文本编码器
python -m helios.train --config configs/helios_tiny.yaml --stage dit
```

## 生成

```bash
# 图片
python -m helios.sample --ckpt runs/helios_tiny/latest_dit.pt --prompt "一只雪地里的红狐"
# 视频（GIF）
python -m helios.sample --ckpt runs/helios_tiny/latest_dit.pt --prompt "海浪拍打礁石" --type video --out out.gif
```

## 项目结构

```
helios/
  models/      TinyVAE · ByteTextEncoder · HeliosDiT
  diffusion/   rectified flow（flow_loss + euler_sample）
  data/        图片/视频目录数据集
  train.py     两阶段训练入口
  sample.py    采样生成入口
configs/       helios_tiny.yaml（tiny 配置）
scripts/       smoke_test.py（端到端冒烟测试）
docs/          官网（GitHub Pages）
```

## 冒烟测试

```bash
python scripts/smoke_test.py
```

## License

MIT
