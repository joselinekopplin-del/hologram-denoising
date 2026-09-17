# 全息通信重建图：轻量 U-Net 去噪

网页版本位于 `web/`，可通过 GitHub Pages 发布。网页使用 `web/model.onnx` 在浏览器中执行轻量 U-Net 推理，不需要上传图片到服务器。

这是一个面向物理专业本科生的、可以逐步运行的教学/科研起点项目。它学习的是：

```text
带噪重建图 I_noisy → 轻量 U-Net → 去噪重建图 I_denoised
                                      ↑
                         训练标签 I_clean
```

当前第一阶段使用的是 **Synthetic / Simulation Dataset（仿真数据集）**。仿真结果不能直接描述成真实 CCD/CMOS 实验结果；将来加入真实实验数据后，应单独划分和评估。

## 从零开始运行

在 Windows PowerShell 中，在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_cuda.py
python generate_dataset.py
python train.py
python evaluate.py
python inference.py --input data/test/noisy/synthetic_00000.png
```

如果 PowerShell 不允许激活脚本，可以先执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

RTX 5060 的 PyTorch CUDA wheel 通常会自带所需 CUDA runtime，不要因为系统中没有单独安装 CUDA Toolkit 就盲目改环境。若 `check_cuda.py` 显示 `CUDA available: False`，请到 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)按 Windows、Pip、Python、CUDA 版本选择器生成命令，然后重新安装对应版本；这比手写一个不匹配的 CUDA 版本更安全。

第一次只想验证代码，可以用较短的 smoke test：

```powershell
python train.py --epochs 2 --batch-size 4 --max-train-batches 10 --max-val-batches 4
python evaluate.py --num-comparisons 3
```

## 目录说明

```text
hologram_denoising/
├─ data/
│  ├─ clean/                 # 原始 clean 图，按原图身份保存
│  ├─ generated/             # 为后续扩展预留
│  ├─ train/{clean,noisy}/
│  ├─ val/{clean,noisy}/
│  ├─ test/{clean,noisy}/
│  └─ metadata.csv           # 每张 noisy 图实际使用的噪声参数
├─ outputs/
│  ├─ checkpoints/           # best_model.pt、last_model.pt
│  ├─ comparisons/           # Clean/Noisy/Denoised 对比图
│  ├─ curves/                # 训练曲线和 history.csv
│  └─ metrics/               # test_results.csv
├─ src/
│  ├─ model.py               # Lightweight U-Net
│  ├─ dataset.py             # 成对图像 Dataset
│  ├─ noise.py               # 噪声模型
│  └─ metrics.py             # MSE、PSNR、SSIM、BER 接口
├─ generate_dataset.py
├─ train.py
├─ evaluate.py
├─ inference.py
├─ app.py                   # Windows Tkinter 可视化软件
└─ check_cuda.py
```

## 数据生成与数据泄漏控制

`generate_dataset.py` 默认生成 10,000 张 256×256 灰度 clean 图，内容包括数字、英文字母、几何图形、条纹、圆环、十字、随机组合和简化二值通信帧。字体会自动尝试 Windows 常见字体；字体不可用时会回退到 PIL 的基本字体，不会让程序因字体报错退出。

划分是在 clean 原图层面完成的：70% train、15% validation、15% test。每个 clean 原图只属于一个 split，然后才对它生成对应 noisy 图，因此同一原图不会同时出现在 train 和 test 中。

重新生成完整数据集：

```powershell
python generate_dataset.py --force --seed 123
```

也可以指定规模，例如 `python generate_dataset.py --num-images 20000`。增加数据量会增加磁盘占用和数据生成时间，但通常能让模型见到更多图案组合和噪声参数。

`metadata.csv` 记录了文件名、split、噪声类型、Gaussian sigma、Poisson peak、speckle sigma、强度因子、模糊 sigma 和背景噪声 sigma。混合噪声中未使用的字段为空。

## 噪声模型

`src/noise.py` 实现了 Gaussian、Poisson、Speckle、Intensity Fluctuation、Gaussian Blur、Background Noise 和 Mixed Noise。强度在最后裁剪到 `[0,1]`，保存 PNG 时映射为 `[0,255]`。Mixed Noise 每张图随机组合 2–4 种噪声，更接近“多种实验误差同时存在”的教学模拟，但它仍不等同于真实光学系统的完整物理模型。

## 模型为什么能去噪

- **CNN**：卷积核在局部邻域内提取边缘、纹理和亮度变化，适合图像这种有空间结构的数据。
- **U-Net**：左侧 Encoder 逐步下采样，提取从局部纹理到全局形状的特征；右侧 Decoder 上采样，恢复图像尺寸。
- **Skip Connection**：把 Encoder 的细节直接传给 Decoder，防止上采样后边缘和细线被丢掉。
- **Loss**：模型输出和 clean 标签之间的误差。默认使用 MSE，简单、稳定、无需额外复杂依赖。
- **MSE**：逐像素计算 `(预测值 - 标签值)^2` 再取平均；误差越小越好。
- **PSNR**：从 MSE 转换而来的 dB 指标，通常越大越好，适合衡量像素重建质量。
- **SSIM**：从亮度、对比度和结构相似性评价图像，范围通常接近 `[-1,1]`，越接近 1 越相似。
- **Epoch**：完整看过一次训练集。
- **Batch Size**：一次送入 GPU 的图片数量。8GB 显存可以从 8 开始，显存不足时降到 4 或 2。
- **Learning Rate**：每次根据误差更新参数时迈多大一步。太大可能不稳定，太小会学习很慢。
- **Backpropagation**：把输出误差沿网络反向传回，计算每个参数应该怎样改变。
- **AdamW**：根据历史梯度自适应调节参数更新，并加入权重衰减，通常比手写普通梯度下降更容易稳定训练。

训练时使用 AdamW、初始学习率 `1e-3`、ReduceLROnPlateau 和 CUDA mixed precision。默认模型通道为 32/64/128/256，参数量约 1–5M 范围内，实际数量可由 `python src/model.py` 打印。

## 训练、断点和评估

标准训练默认 50 个 epoch：

```powershell
python train.py --epochs 100 --batch-size 8
```

保存的 `best_model.pt` 以 validation SSIM 最好为准；`last_model.pt` 保存最近一轮的模型、优化器、学习率调度器、AMP scaler 和 history。训练中断后：

```powershell
python train.py --resume
```

评估会在 test set 上打印去噪前后的 PSNR、SSIM 以及提升量，并保存：

- `outputs/metrics/test_results.csv`：每个测试样本一行；
- `outputs/comparisons/comparison_001.png` 等：至少 10 张 Clean | Noisy | Denoised 对比图；
- `outputs/curves/`：train loss、validation loss、validation PSNR、validation SSIM 曲线。

## 真实 CCD/CMOS 数据如何替换

1. `clean` 应是相同场景、相同空间位置、尽可能高信噪比的理想/参考重建图；不能把一张不同场景图误当成标签。
2. `noisy` 必须和 `clean` 像素级空间对应：裁剪、旋转、缩放、配准后坐标一致。
3. 尽量保持曝光、增益、波长、光路和重建参数一致；如果真实实验中它们会变化，应把这些变化纳入数据集并记录 metadata。
4. 训练前将两者转换为灰度、归一化到 `[0,1]`，同时保留原始位深和曝光信息，便于科研复现。
5. 先按实验样本/场景划分 train、val、test，再生成或收集同一场景的 noisy 观测。不要把同一场景的不同噪声版本拆到不同 split，否则会产生数据泄漏。
6. 真实实验结果单独保存和单独汇报，不能与仿真数据混合后直接声称为实验结果。

单张图片推理：

```powershell
python inference.py --input my_experiment.png
```

输出默认是 `my_experiment_denoised.png`。模型是全卷积网络，程序会在输入尺寸的右侧和下侧做反射 padding，使尺寸适合下采样，再裁回原始尺寸，因此输出保留原始宽高，不会强制缩放成 256×256。输入会转为灰度；如果实验图包含重要彩色/多通道信息，需要后续扩展模型的输入通道。

扩充数据后建议重新训练模型：

```powershell
python train.py --epochs 50 --batch-size 8
```

## 可视化软件

项目还提供一个 Windows 桌面可视化软件，不需要编写 GUI 代码即可查看效果：

```powershell
python app.py
```

`app.py` 使用 Windows Python 通常自带的 Tkinter，不需要安装 Web 框架。若运行时提示 `tkinter` 或 `init.tcl` 缺失，请重新安装 python.org 的 Windows Python，并在安装器中保留 Tcl/Tk 组件；某些第三方内置 Python 运行时可能只包含 PyTorch，不包含 Tkinter 的 Tcl/Tk 文件。

软件支持：

- 打开真实实验图片或仿真 noisy 图片；
- 随机选择 test set 样本；
- 选择 `best_model.pt` 或其他 checkpoint；
- 显示 Noisy、Denoised、Clean 三栏图像；
- 有 clean 参考图时自动计算 PSNR 和 SSIM；
- 保存原始尺寸的去噪 PNG；
- 自动使用 CUDA（如果可用），否则使用 CPU。

如果从 `data/test/noisy/` 打开图片，软件会自动寻找同名的 `data/test/clean/` 作为参考图。打开真实 CCD/CMOS 图片时没有 clean 参考图也可以推理，但不能计算有监督 PSNR/SSIM。

## BER 预留接口

`src.metrics.calculate_ber(pred_bits, target_bits)` 接收两个相同长度的 0/1 序列，定义为：

```text
BER = 错误 bit 数 / 总 bit 数
```

未来若重建图代表二进制数据帧，可以执行 `denoised image → threshold → binary image`，再和 ground truth 比较。此项目暂不假设通信帧的同步、纠错码或复杂解码规则，因此没有伪装成完整通信 BER 系统。
