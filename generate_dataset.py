"""生成教学用 synthetic clean/noisy 配对数据，并写入 metadata.csv。"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.noise import apply_noise


def find_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont | None:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_pattern(size: int, index: int) -> Image.Image:
    rng = random.Random(index * 1009 + 17)
    image = Image.new("L", (size, size), color=rng.randint(0, 12))
    draw = ImageDraw.Draw(image)
    font = find_font(rng.randint(size // 5, size // 2))
    kinds = ["digit", "letter", "shape", "stripe", "ring", "cross", "frame"]
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(kinds)
        cx, cy = rng.randint(35, size - 35), rng.randint(35, size - 35)
        extent = rng.randint(size // 10, size // 3)
        color = rng.randint(100, 255)
        if kind == "digit":
            draw.text((cx - extent // 2, cy - extent // 2), str(rng.randint(0, 9)), fill=color, font=font)
        elif kind == "letter":
            draw.text((cx - extent // 2, cy - extent // 2), rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), fill=color, font=font)
        elif kind == "shape":
            box = (cx - extent, cy - extent, cx + extent, cy + extent)
            if rng.random() < 0.5:
                draw.rectangle(box, outline=color, width=max(1, size // 64))
            else:
                draw.ellipse(box, outline=color, width=max(1, size // 64))
        elif kind == "stripe":
            gap = rng.randint(8, 24)
            for x in range(cx - extent, cx + extent, gap):
                draw.line((x, cy - extent, x + extent // 2, cy + extent), fill=color, width=max(1, size // 96))
        elif kind == "ring":
            for ring in range(2, 5):
                r = extent * ring // 4
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=max(1, size // 80))
        elif kind == "cross":
            width = max(2, extent // 4)
            draw.rectangle((cx - width, cy - extent, cx + width, cy + extent), fill=color)
            draw.rectangle((cx - extent, cy - width, cx + extent, cy + width), fill=color)
        else:
            cell = max(4, extent // 5)
            for row in range(5):
                for col in range(5):
                    if rng.random() > 0.5:
                        x0, y0 = cx - extent + col * cell, cy - extent + row * cell
                        draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=color)

    # 叠加一个简化的通信数据帧，帮助未来测试阈值化和 BER。
    if rng.random() < 0.55:
        left, top = rng.randint(8, 35), rng.randint(8, 35)
        cell = max(3, size // 64)
        for row in range(8):
            for col in range(16):
                bit = rng.randint(0, 1)
                if bit:
                    draw.rectangle((left + col * cell, top + row * cell, left + (col + 1) * cell - 1, top + (row + 1) * cell - 1), fill=rng.randint(160, 255))

    if rng.random() < 0.35:
        image = image.rotate(rng.uniform(-25, 25), resample=Image.Resampling.BILINEAR, fillcolor=0)
    return image


def save_uint8(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.squeeze().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 synthetic 全息重建图去噪数据")
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="清空并重建 clean/generated/train/val/test")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_root = root / "data"
    if args.force:
        for name in ["clean", "generated", "train", "val", "test"]:
            shutil.rmtree(data_root / name, ignore_errors=True)
    for path in [data_root / "clean", data_root / "generated"] + [data_root / s / k for s in ["train", "val", "test"] for k in ["clean", "noisy"]]:
        path.mkdir(parents=True, exist_ok=True)
    # 重新生成时清理旧 split，避免 --num-images 变小时遗留旧样本。
    for split in ["train", "val", "test"]:
        for kind in ["clean", "noisy"]:
            for old_file in (data_root / split / kind).glob("*.png"):
                old_file.unlink()

    existing = sorted((data_root / "clean").glob("*.png"))
    if len(existing) < args.num_images:
        for index in range(len(existing), args.num_images):
            draw_pattern(args.image_size, index).save(data_root / "clean" / f"synthetic_{index:05d}.png")
    clean_files = sorted((data_root / "clean").glob("*.png"))[: args.num_images]
    rng = random.Random(args.seed)
    rng.shuffle(clean_files)
    n_train = math.floor(len(clean_files) * 0.70)
    n_val = math.floor(len(clean_files) * 0.15)
    split_files = {"train": clean_files[:n_train], "val": clean_files[n_train:n_train + n_val], "test": clean_files[n_train + n_val:]}
    metadata_path = data_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["filename", "clean_filename", "split", "noise_type", "gaussian_sigma", "poisson_peak", "speckle_sigma", "intensity_factor", "blur_sigma", "background_sigma"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, paths in split_files.items():
            for clean_path in paths:
                destination_clean = data_root / split / "clean" / clean_path.name
                destination_noisy = data_root / split / "noisy" / clean_path.name
                shutil.copy2(clean_path, destination_clean)
                clean = torch.from_numpy(np.asarray(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0)[None]
                noisy, params = apply_noise(clean, "mixed")
                save_uint8(noisy, destination_noisy)
                row = {"filename": clean_path.name, "clean_filename": clean_path.name, "split": split, **params}
                writer.writerow(row)
    print(f"Synthetic 数据已生成: {len(clean_files)} 张 {args.image_size}x{args.image_size} 灰度图")
    print(f"train={len(split_files['train'])}, val={len(split_files['val'])}, test={len(split_files['test'])}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
