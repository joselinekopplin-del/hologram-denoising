"""用于模拟光学通信/全息实验中常见噪声的函数。"""

from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """生成二维高斯卷积核。sigma=0 时不应调用本函数。"""
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d[None, None]


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """对 [1,H,W] 图像做轻微高斯模糊。"""
    if sigma <= 1e-6:
        return image
    kernel = _gaussian_kernel(sigma, image.device)
    pad = kernel.shape[-1] // 2
    return F.conv2d(image[None], kernel, padding=pad).squeeze(0)


def apply_noise(image: torch.Tensor, noise_type: str = "mixed") -> Tuple[torch.Tensor, Dict[str, float]]:
    """给归一化的 [1,H,W] clean 图像加噪。

    返回 noisy 图像和实际使用的参数，所有输出都会裁剪到 [0,1]。
    """
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("image 必须是 [1,H,W] 的单通道张量")
    clean = image.float().clamp(0.0, 1.0)
    noisy = clean.clone()
    params: Dict[str, float] = {
        "noise_type": noise_type,
        "gaussian_sigma": "",
        "poisson_peak": "",
        "speckle_sigma": "",
        "intensity_factor": "",
        "blur_sigma": "",
        "background_sigma": "",
    }

    if noise_type == "mixed":
        choices = ["gaussian", "poisson", "speckle", "intensity", "blur", "background"]
        selected = random.sample(choices, k=random.randint(2, 4))
    else:
        selected = [noise_type]

    for kind in selected:
        if kind == "gaussian":
            sigma = random.uniform(0.005, 0.10)
            noisy = noisy + torch.randn_like(noisy) * sigma
            params["gaussian_sigma"] = round(sigma, 6)
        elif kind == "poisson":
            peak = random.uniform(80.0, 1500.0)
            noisy = torch.poisson(noisy.clamp(0, 1) * peak) / peak
            params["poisson_peak"] = round(peak, 4)
        elif kind == "speckle":
            sigma = random.uniform(0.02, 0.20)
            noisy = noisy + noisy * torch.randn_like(noisy) * sigma
            params["speckle_sigma"] = round(sigma, 6)
        elif kind == "intensity":
            factor = random.uniform(0.7, 1.3)
            noisy = factor * noisy
            params["intensity_factor"] = round(factor, 6)
        elif kind == "blur":
            sigma = random.uniform(0.35, 1.50)
            noisy = gaussian_blur(noisy, sigma)
            params["blur_sigma"] = round(sigma, 6)
        elif kind == "background":
            sigma = random.uniform(0.005, 0.04)
            background = torch.randn_like(noisy) * sigma
            noisy = noisy + background
            params["background_sigma"] = round(sigma, 6)
        else:
            raise ValueError(f"未知噪声类型: {kind}")

    return noisy.clamp(0.0, 1.0), params


if __name__ == "__main__":
    x = torch.zeros(1, 256, 256)
    x[:, 80:180, 80:180] = 1.0
    y, p = apply_noise(x, "mixed")
    print("output range:", float(y.min()), float(y.max()))
    print("parameters:", p)

