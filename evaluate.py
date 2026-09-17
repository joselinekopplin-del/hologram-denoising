"""在 test set 上评估，并保存逐图 CSV 与 Clean/Noisy/Denoised 对比图。"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from src.dataset import PairedImageDataset
from src.metrics import mse, psnr, ssim
from src.model import LightweightUNet


def save_comparison(clean, noisy, denoised, row, filename: Path) -> None:
    """优先用 matplotlib；依赖缺失时用 Pillow 保证评估仍可完成。"""
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, image, title in zip(axes, [clean, noisy, denoised], ["Clean", f"Noisy\nPSNR {row['psnr_noisy']:.2f} dB | SSIM {row['ssim_noisy']:.3f}", f"Denoised\nPSNR {row['psnr_denoised']:.2f} dB | SSIM {row['ssim_denoised']:.3f}"]):
            ax.imshow(image.squeeze().numpy(), cmap="gray", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
        fig.suptitle(row["filename"]); fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)
        return
    images = [(clean, "Clean"), (noisy, f"Noisy PSNR {row['psnr_noisy']:.2f} SSIM {row['ssim_noisy']:.3f}"), (denoised, f"Denoised PSNR {row['psnr_denoised']:.2f} SSIM {row['ssim_denoised']:.3f}")]
    panel_width, panel_height = clean.shape[-1], clean.shape[-2]
    canvas = Image.new("L", (panel_width * 3, panel_height + 42), color=255)
    draw = ImageDraw.Draw(canvas)
    for index, (image, title) in enumerate(images):
        array = (image.squeeze().numpy().clip(0, 1) * 255).astype("uint8")
        panel = Image.fromarray(array, mode="L")
        canvas.paste(panel, (index * panel_width, 42))
        draw.text((index * panel_width + 4, 4), title, fill=0, font=ImageFont.load_default())
    canvas.save(filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--num-comparisons", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightUNet().to(device)
    checkpoint = torch.load(root / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    dataset = PairedImageDataset(root, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows, examples = [], []
    with torch.no_grad():
        for noisy, clean, filename in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy).cpu().squeeze(0)
            noisy_cpu, clean_cpu = noisy.cpu().squeeze(0), clean.cpu().squeeze(0)
            row = {"filename": filename[0], "mse_noisy": mse(noisy_cpu, clean_cpu), "psnr_noisy": psnr(noisy_cpu, clean_cpu), "ssim_noisy": ssim(noisy_cpu, clean_cpu), "mse_denoised": mse(denoised, clean_cpu), "psnr_denoised": psnr(denoised, clean_cpu), "ssim_denoised": ssim(denoised, clean_cpu)}
            rows.append(row)
            examples.append((filename[0], clean_cpu, noisy_cpu, denoised, row))
    metrics_dir, comparisons_dir = root / "outputs" / "metrics", root / "outputs" / "comparisons"
    metrics_dir.mkdir(parents=True, exist_ok=True); comparisons_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "test_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    avg = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0] if key != "filename"}
    print("Before denoising:"); print(f"PSNR = {avg['psnr_noisy']:.2f} dB"); print(f"SSIM = {avg['ssim_noisy']:.3f}")
    print("After denoising:"); print(f"PSNR = {avg['psnr_denoised']:.2f} dB"); print(f"SSIM = {avg['ssim_denoised']:.3f}")
    print("Improvement:"); print(f"ΔPSNR = {avg['psnr_denoised'] - avg['psnr_noisy']:.2f} dB"); print(f"ΔSSIM = {avg['ssim_denoised'] - avg['ssim_noisy']:.3f}")
    random.seed(args.seed); selected = random.sample(examples, min(args.num_comparisons, len(examples)))
    for index, (filename, clean, noisy, denoised, row) in enumerate(selected, 1):
        save_comparison(clean, noisy, denoised, row, comparisons_dir / f"comparison_{index:03d}.png")
    print(f"CSV: {metrics_dir / 'test_results.csv'}")
    print(f"comparisons: {comparisons_dir}")


if __name__ == "__main__":
    main()
