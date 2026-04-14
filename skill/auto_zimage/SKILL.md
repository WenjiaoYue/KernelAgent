---
name: auto_zimage
description: Run Z-Image (Tongyi-MAI/Z-Image-Turbo) text-to-image on XPU/CUDA. Handles venv, deps, error recovery.
metadata:
  openclaw:
    emoji: "🎨"
    homepage: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
    skillKey: auto-zimage
    requires:
      bins: []
      env: []
      config: []
---

# Auto Z-Image Inference Skill

Use this skill to run Z-Image text-to-image generation.
Handles venv creation, dependency installation, error recovery, and image saving.

---

## Input Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `prompt` | Text prompt for image generation | Yes | - |
| `output_path` | Where to save the PNG | Yes | - |
| `output_dir` | Parent dir for venv + logs | Yes | - |
| `device` | `xpu`, `xpu:0`, `cuda`, `cuda:0` | No | `xpu:0` |
| `num_inference_steps` | Denoising steps | No | `9` |
| `seed` | Random seed | No | `42` |

---

## Step 1: Check System Environment

**CRITICAL: Always check existing torch BEFORE creating venv.**

```bash
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
try:
    print('XPU:', torch.xpu.is_available())
    print('XPU count:', torch.xpu.device_count())
except AttributeError:
    print('XPU: not supported in this torch build')
" 2>/dev/null || echo "torch: NOT installed"

pip3 list 2>/dev/null | grep -iE "diffusers|torch|triton|flash|intel"
```

**Decision table:**

| System torch | Target device | Venv strategy |
|---|---|---|
| Installed, matches target | xpu or cuda | `--system-site-packages` (reuse torch) |
| Installed CUDA, target is xpu | xpu | plain venv + install XPU torch |
| NOT installed | any | plain venv + install matching torch |

---

## Step 2: Create Isolated Virtual Environment

```bash
mkdir -p {output_dir}/logs

# System torch matches target -> reuse it
python3 -m venv {output_dir}/venv --system-site-packages
# System torch absent or wrong backend -> plain venv (uncomment):
# python3 -m venv {output_dir}/venv

. {output_dir}/venv/bin/activate
pip install -U pip setuptools wheel
```

**NEVER reinstall torch/flash_attn — driver-coupled.**

---

## Step 3: Install diffusers and Dependencies

```bash
. {output_dir}/venv/bin/activate
pip install git+https://github.com/huggingface/diffusers transformers accelerate pillow

# Verify
python -c "import diffusers; print('diffusers:', diffusers.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
```

If XPU torch missing (system torch absent or wrong backend):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

---

## Step 4: Write and Run Inference Script

Save to `{output_dir}/zimage_run.py` with this content, then run it:

```python
import os, sys, logging

# Set device env BEFORE any torch import
device = "{device}"
if device.startswith("xpu"):
    os.environ["ZE_AFFINITY_MASK"] = device.split(":")[-1] if ":" in device else "0"
elif device.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"

import torch
from diffusers import ZImagePipeline

log_path = "{output_dir}/logs/zimage.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

dev_base = device.split(":")[0]
if dev_base == "xpu":
    assert torch.xpu.is_available(), "XPU not available!"
    logger.info(f"XPU devices: {torch.xpu.device_count()}")
elif dev_base == "cuda":
    assert torch.cuda.is_available(), "CUDA not available!"
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

logger.info(f"Loading ZImagePipeline on {device}...")
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to(device)
logger.info("Model loaded.")

prompt = "{prompt}"
generator = torch.Generator(dev_base).manual_seed({seed})
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps={num_inference_steps},
    guidance_scale=0.0,
    generator=generator,
).images[0]

output_path = "{output_path}"
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
image.save(output_path)
logger.info(f"Saved -> {output_path}")
print(f"SUCCESS: {output_path}")
```

Run:
```bash
. {output_dir}/venv/bin/activate
python {output_dir}/zimage_run.py 2>&1 | tee {output_dir}/logs/zimage.log
```

---

## Step 5: Error Handling and Recovery

| Error | Fix |
|-------|-----|
| `No module named 'diffusers'` | `pip install git+https://github.com/huggingface/diffusers` |
| `No module named 'accelerate'` | `pip install accelerate` |
| `XPU not available` | Check driver or install XPU torch |
| `CUDA out of memory` | Use `torch_dtype=torch.float16` or free GPU |
| `bfloat16 not supported` | Switch to `torch_dtype=torch.float16` |
| `generator device mismatch` | Use `torch.Generator("xpu")` for XPU |
| `ZImagePipeline not found` | `pip install git+https://github.com/huggingface/diffusers` |

**Always retry after fixing. Do not stop until the image is saved.**

```bash
tail -50 {output_dir}/logs/zimage.log
# fix script, then:
. {output_dir}/venv/bin/activate
python {output_dir}/zimage_run.py 2>&1 | tee -a {output_dir}/logs/zimage.log
```

---

## Step 6: Verify Output

```bash
ls -lh {output_path}
echo "Done: $(date)" >> {output_dir}/logs/zimage.log
```

---

## Quick Reference

| Need | Solution |
|------|----------|
| XPU device select | `ZE_AFFINITY_MASK=0` before torch import |
| CUDA device select | `CUDA_VISIBLE_DEVICES=0` before torch import |
| Fix bfloat16 | `torch_dtype=torch.float16` |
| Reuse system torch | `--system-site-packages` |
| Generator device error | `torch.Generator("xpu")` or `torch.Generator("cuda")` |

### NEVER MODIFY (driver-coupled)
- torch, flash_attn, pytorch-triton

### FREE TO INSTALL
- diffusers, accelerate, pillow, transformers
