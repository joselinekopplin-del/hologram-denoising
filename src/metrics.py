"""图像质量指标和未来 BER 接口。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity
except ImportError:  # 只在环境缺少 scikit-image 时触发
    structural_similarity = None


def _numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().squeeze().numpy()
    return np.asarray(image, dtype=np.float32).squeeze()


def mse(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    p, t = _numpy_image(pred), _numpy_image(target)
    return float(np.mean((p - t) ** 2))


def psnr(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, data_range: float = 1.0) -> float:
    value = mse(pred, target)
    if value <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range**2) / value))


def ssim(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, data_range: float = 1.0) -> float:
    if structural_similarity is None:
        raise ImportError("SSIM 需要 scikit-image，请运行 pip install scikit-image")
    p, t = _numpy_image(pred), _numpy_image(target)
    return float(structural_similarity(t, p, data_range=data_range))


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    """返回 batch 的平均 MSE、PSNR、SSIM。"""
    values = [(mse(p, t), psnr(p, t), ssim(p, t)) for p, t in zip(pred, target)]
    return tuple(float(np.mean([row[i] for row in values])) for i in range(3))


def calculate_ber(pred_bits: Iterable[int], target_bits: Iterable[int]) -> float:
    """预留的 BER 接口：输入已经阈值化后的 0/1 bit 序列。"""
    pred = np.asarray(list(pred_bits), dtype=np.uint8)
    target = np.asarray(list(target_bits), dtype=np.uint8)
    if pred.shape != target.shape or pred.size == 0:
        raise ValueError("pred_bits 和 target_bits 必须形状相同且非空")
    return float(np.mean(pred != target))

