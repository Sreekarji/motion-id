# Cell 0: Install extra deps + GPU check
import subprocess, os, torch

# Do NOT reinstall PyTorch - Kaggle P100 already has correct cu128 version
subprocess.run([
    "pip", "install", "-q",
    "numpy", "scipy", "pandas", "scikit-learn", "tqdm", "pytorch_metric_learning"
], check=True)

assert torch.cuda.is_available(), "Enable GPU in notebook settings!"
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("Setup complete.")