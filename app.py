"""轻量 U-Net 去噪可视化桌面软件。

运行：python app.py
依赖 Tkinter（通常随 Windows Python 安装）和 requirements.txt 中的 Pillow/PyTorch。
"""

from __future__ import annotations

import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision.transforms.functional import pil_to_tensor

from src.metrics import psnr, ssim
from src.model import LightweightUNet, count_parameters


class DenoisingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("全息通信重建图 · 轻量 U-Net 去噪实验台")
        self.root.geometry("1240x780")
        self.root.minsize(1000, 680)
        self.project_root = Path(__file__).resolve().parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LightweightUNet | None = None
        self.model_path: Path | None = None
        self.input_path: Path | None = None
        self.reference_path: Path | None = None
        self.input_image: Image.Image | None = None
        self.output_image: Image.Image | None = None
        self.reference_image: Image.Image | None = None
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self.status_var = tk.StringVar(value=f"设备：{self.device}")
        self.metrics_var = tk.StringVar(value="尚未执行推理")
        self.checkpoint_var = tk.StringVar(value=str(self.project_root / "outputs" / "checkpoints" / "best_model.pt"))
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="打开图片", command=self.open_image).pack(side="left", padx=4)
        ttk.Button(toolbar, text="随机测试样本", command=self.open_random_sample).pack(side="left", padx=4)
        ttk.Button(toolbar, text="选择模型", command=self.choose_checkpoint).pack(side="left", padx=4)
        ttk.Button(toolbar, text="开始去噪", command=self.start_inference).pack(side="left", padx=4)
        ttk.Button(toolbar, text="保存去噪图", command=self.save_output).pack(side="left", padx=4)
        ttk.Label(toolbar, text="Checkpoint:").pack(side="left", padx=(18, 4))
        ttk.Entry(toolbar, textvariable=self.checkpoint_var, width=58).pack(side="left", fill="x", expand=True)

        info = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.status_var, foreground="#1f5f99").pack(anchor="w")
        ttk.Label(info, textvariable=self.metrics_var, foreground="#7a3e00").pack(anchor="w", pady=(4, 0))

        self.image_frame = ttk.Frame(self.root, padding=10)
        self.image_frame.pack(fill="both", expand=True)
        self.panels: list[tuple[ttk.Label, ttk.Label]] = []
        for title in ["Noisy / 输入图", "Denoised / 去噪结果", "Clean / 参考图"]:
            panel = ttk.Frame(self.image_frame, relief="groove", borderwidth=1)
            panel.pack(side="left", fill="both", expand=True, padx=6)
            title_label = ttk.Label(panel, text=title, anchor="center", font=("Segoe UI", 12, "bold"))
            title_label.pack(fill="x", pady=6)
            image_label = ttk.Label(panel, text="等待图片", anchor="center")
            image_label.pack(fill="both", expand=True, padx=8, pady=8)
            self.panels.append((title_label, image_label))

        footer = ttk.Frame(self.root, padding=10)
        footer.pack(fill="x")
        ttk.Label(footer, text="说明：输入图应为单通道灰度图；程序会自动 padding 到适合网络下采样的尺寸，并裁回原始尺寸。", foreground="#666666").pack(anchor="w")

    def choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(title="选择模型 checkpoint", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.checkpoint_var.set(path)
            self.model = None

    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="打开输入图像", filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if path:
            self.load_input(Path(path))

    def open_random_sample(self) -> None:
        folder = self.project_root / "data" / "test" / "noisy"
        files = sorted(folder.glob("*.png"))
        if not files:
            messagebox.showwarning("没有测试样本", "请先运行 generate_dataset.py 生成数据。")
            return
        self.load_input(random.choice(files))

    def load_input(self, path: Path) -> None:
        try:
            self.input_path = path
            self.input_image = Image.open(path).convert("L")
            self.output_image = None
            self.reference_path = self._find_reference(path)
            self.reference_image = Image.open(self.reference_path).convert("L") if self.reference_path else None
            self.show_images()
            reference_text = f"；参考图：{self.reference_path.name}" if self.reference_path else "；未找到对应参考 clean 图"
            self.status_var.set(f"已加载：{path.name}，尺寸 {self.input_image.width}×{self.input_image.height}{reference_text}")
            self.metrics_var.set("点击“开始去噪”执行模型推理")
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _find_reference(self, path: Path) -> Path | None:
        # 识别项目 test/noisy 下的样本，自动找到同名 test/clean 文件。
        if path.parent.name == "noisy" and path.parent.parent.name in {"train", "val", "test"}:
            candidate = path.parent.parent / "clean" / path.name
            if candidate.exists():
                return candidate
        return None

    def _load_model(self) -> None:
        checkpoint_path = Path(self.checkpoint_var.get()).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")
        self.status_var.set(f"正在加载模型：{checkpoint_path.name} …")
        self.root.update_idletasks()
        model = LightweightUNet().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
        model.eval()
        self.model = model
        self.model_path = checkpoint_path

    def start_inference(self) -> None:
        if self.input_image is None or self.input_path is None:
            messagebox.showinfo("请先打开图片", "先打开一张实验图或随机测试样本。")
            return
        self.status_var.set("正在推理，请稍候 …")
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self) -> None:
        try:
            if self.model is None:
                self._load_model()
            assert self.model is not None and self.input_image is not None
            original_h, original_w = self.input_image.height, self.input_image.width
            tensor = pil_to_tensor(self.input_image).float()[None] / 255.0
            height, width = tensor.shape[-2:]
            pad_h, pad_w = (16 - height % 16) % 16, (16 - width % 16) % 16
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            with torch.no_grad():
                output = self.model(tensor.to(self.device)).cpu()[..., :height, :width].squeeze().clamp(0, 1)
            array = (output.numpy() * 255).round().astype(np.uint8)
            result = Image.fromarray(array, mode="L").resize((original_w, original_h))
            self.root.after(0, lambda: self._inference_done(result))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("推理失败", str(exc)))
            self.root.after(0, lambda: self.status_var.set("推理失败"))

    def _inference_done(self, result: Image.Image) -> None:
        self.output_image = result
        self.show_images()
        if self.input_image is not None and self.reference_image is not None:
            noisy = np.asarray(self.input_image, dtype=np.float32) / 255.0
            clean = np.asarray(self.reference_image.resize(self.input_image.size), dtype=np.float32) / 255.0
            denoised = np.asarray(result, dtype=np.float32) / 255.0
            self.metrics_var.set(
                f"Noisy  PSNR {psnr(noisy, clean):.2f} dB | SSIM {ssim(noisy, clean):.3f}    "
                f"Denoised  PSNR {psnr(denoised, clean):.2f} dB | SSIM {ssim(denoised, clean):.3f}"
            )
        else:
            self.metrics_var.set("没有 clean 参考图，无法计算 PSNR/SSIM；可以保存去噪结果。")
        self.status_var.set(f"推理完成：{self.output_image.width}×{self.output_image.height}，设备 {self.device}")

    def show_images(self) -> None:
        images = [self.input_image, self.output_image, self.reference_image]
        self.photo_refs.clear()
        for (_, label), image in zip(self.panels, images):
            if image is None:
                label.configure(image="", text="暂无图片")
                continue
            preview = image.copy()
            preview.thumbnail((350, 560), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            self.photo_refs.append(photo)
            label.configure(image=photo, text="")

    def save_output(self) -> None:
        if self.output_image is None:
            messagebox.showinfo("暂无结果", "请先执行去噪。")
            return
        default_name = (self.input_path.stem + "_denoised.png") if self.input_path else "denoised.png"
        path = filedialog.asksaveasfilename(title="保存去噪图", initialfile=default_name, defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self.output_image.save(path)
            self.status_var.set(f"已保存：{path}")


def main() -> None:
    root = tk.Tk()
    DenoisingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

