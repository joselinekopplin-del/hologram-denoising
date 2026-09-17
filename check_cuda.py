"""检查 PyTorch、CUDA 和显卡是否可用。"""

import torch

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    x = torch.randn(256, 256, device="cuda")
    print("CUDA test:", float((x @ x.T).mean()))
else:
    print("GPU name: unavailable")
    print("GPU memory: unavailable")

