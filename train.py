"""训练轻量 U-Net。默认 MSE；可用 --epochs、--batch-size、--resume 修改。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from src.dataset import PairedImageDataset
from src.metrics import batch_metrics
from src.model import LightweightUNet, count_parameters


def save_fallback_curve(epochs, values, ylabel, path):
    """matplotlib 不可用时的极简曲线回退，正常环境仍使用 matplotlib。"""
    width, height = 720, 440
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 30, width - 25, height - 55
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    points = []
    for epoch, value in zip(epochs, values):
        x = left if len(epochs) == 1 else left + (epoch - epochs[0]) * (right - left) / (epochs[-1] - epochs[0])
        y = bottom - (value - lo) * (bottom - top) / (hi - lo)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill="#2060c0", width=3)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#2060c0")
    draw.text((left, 8), ylabel, fill="black")
    image.save(path)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_psnr, total_ssim, count = 0.0, 0.0, 0.0, 0
    for batch_index, (noisy, clean, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(noisy)
            loss = criterion(output, clean)
        if training:
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        m, p, s = batch_metrics(output.detach(), clean)
        batch_size = noisy.size(0)
        total_loss += loss.item() * batch_size
        total_psnr += p * batch_size
        total_ssim += s * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("本 epoch 没有处理任何 batch，请检查数据或 max_batches")
    return total_loss / count, total_psnr / count, total_ssim / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None, help="smoke test 用")
    parser.add_argument("--max-val-batches", type=int, default=None, help="smoke test 用")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "outputs"
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "curves").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    train_set, val_set = PairedImageDataset(root, "train"), PairedImageDataset(root, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = LightweightUNet().to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoints = output / "checkpoints"
    best_path, last_path = checkpoints / "best_model.pt", checkpoints / "last_model.pt"
    start_epoch, best_ssim, history = 0, -float("inf"), []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_ssim = checkpoint.get("best_ssim", best_ssim)
        history = checkpoint.get("history", [])
        print(f"已从 epoch {start_epoch} 继续训练")

    for epoch in range(start_epoch, args.epochs):
        begin = time.perf_counter()
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, args.max_train_batches)
        with torch.no_grad():
            val_loss, val_psnr, val_ssim = run_epoch(model, val_loader, criterion, device, max_batches=args.max_val_batches)
        scheduler.step(val_ssim)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - begin
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim, "lr": lr, "seconds": elapsed}
        history.append(record)
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | train loss {train_loss:.6f} | val loss {val_loss:.6f} | val PSNR {val_psnr:.2f} dB | val SSIM {val_ssim:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_ssim": best_ssim, "history": history}
        torch.save(state, last_path)
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            state["best_ssim"] = best_ssim
            torch.save(state, best_path)

    with (output / "curves" / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (output / "curves" / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    try:
        import matplotlib.pyplot as plt
        epochs = [r["epoch"] for r in history]
        for key, ylabel, filename in [("train_loss", "Train Loss", "train_loss.png"), ("val_loss", "Validation Loss", "val_loss.png"), ("val_psnr", "Validation PSNR (dB)", "val_psnr.png"), ("val_ssim", "Validation SSIM", "val_ssim.png")]:
            plt.figure(figsize=(7, 4))
            plt.plot(epochs, [r[key] for r in history], linewidth=2)
            plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(output / "curves" / filename, dpi=140); plt.close()
    except ImportError:
        print("未找到 matplotlib，使用 Pillow 生成简易训练曲线；安装 requirements.txt 后会优先使用 matplotlib")
        epochs = [r["epoch"] for r in history]
        for key, ylabel, filename in [("train_loss", "Train Loss", "train_loss.png"), ("val_loss", "Validation Loss", "val_loss.png"), ("val_psnr", "Validation PSNR (dB)", "val_psnr.png"), ("val_ssim", "Validation SSIM", "val_ssim.png")]:
            save_fallback_curve(epochs, [r[key] for r in history], ylabel, output / "curves" / filename)
    if device.type == "cuda":
        print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
