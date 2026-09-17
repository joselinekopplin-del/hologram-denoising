"""读取已经生成好的 clean/noisy 成对 PNG 数据。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class PairedImageDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.clean_dir = self.root / "data" / split / "clean"
        self.noisy_dir = self.root / "data" / split / "noisy"
        self.files: List[Path] = sorted(self.clean_dir.glob("*.png"))
        self.files = [p for p in self.files if (self.noisy_dir / p.name).exists()]
        if not self.files:
            raise FileNotFoundError(f"{split} 没有成对 PNG，请先运行 python generate_dataset.py")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        clean_path = self.files[index]
        noisy_path = self.noisy_dir / clean_path.name
        clean = pil_to_tensor(Image.open(clean_path).convert("L")).float() / 255.0
        noisy = pil_to_tensor(Image.open(noisy_path).convert("L")).float() / 255.0
        return noisy, clean, clean_path.name

