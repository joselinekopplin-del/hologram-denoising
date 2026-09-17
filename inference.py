"""对单张真实或仿真灰度图去噪，并保持原始尺寸。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from src.model import LightweightUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_denoised.png")
    image = Image.open(input_path).convert("L")
    original_size = image.size
    x = pil_to_tensor(image).float()[None] / 255.0
    height, width = x.shape[-2:]
    pad_h, pad_w = (16 - height % 16) % 16, (16 - width % 16) % 16
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightUNet().to(device)
    checkpoint = torch.load(root / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    with torch.no_grad():
        y = model(x.to(device)).cpu()[..., :height, :width].squeeze()
    array = (y.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array, mode="L").resize(original_size).save(output_path)
    print(f"input: {input_path} ({original_size[0]}x{original_size[1]})")
    print(f"output: {output_path} ({original_size[0]}x{original_size[1]})")


if __name__ == "__main__":
    main()

