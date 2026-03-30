---
name: auto_zimage
description: Run Z-Image (Tongyi-MAI/Z-Image-Turbo) text-to-image on XPU/CUDA. Uses system Python environment directly, no venv or pip install needed.
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

Use this skill to run Z-Image text-to-image generation using the system Python environment.
No venv creation or pip install required — all dependencies are pre-installed.

---

## Input Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `prompt` | Text prompt for image generation | Yes | - |
| `output_path` | Where to save the PNG | Yes | - |
| `output_dir` | Parent dir for script + logs | Yes | - |
| `device` | `xpu`, `xpu:0`, `cuda`, `cuda:0` | No | `xpu:0` |
| `num_inference_steps` | Denoising steps | No | `9` |
| `seed` | Random seed | No | `42` |

---

## Step 1: Verify System Environment

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
from diffusers import ZImagePipeline
print('diffusers: OK, ZImagePipeline available')
"
```

If any import fails, stop and report — do NOT attempt to install packages.

---

## Step 2: Write and Run Inference Script

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
mkdir -p {output_dir}/logs
python3 {output_dir}/zimage_run.py 2>&1 | tee {output_dir}/logs/zimage.log
```

---

## Step 3: Error Handling and Recovery

| Error | Fix |
|-------|-----|
| `XPU not available` | Check driver: `xpu-smi` or `clinfo` |
| `CUDA out of memory` | Use `torch_dtype=torch.float16` or free GPU |
| `bfloat16 not supported` | Switch to `torch_dtype=torch.float16` |
| `generator device mismatch` | Use `torch.Generator("xpu")` for XPU |
| `AssertionError: XPU not available` | Wrong device, check `ZE_AFFINITY_MASK` |

**Always retry after fixing the script. Do NOT install packages.**

```bash
tail -50 {output_dir}/logs/zimage.log
# fix script, then:
python3 {output_dir}/zimage_run.py 2>&1 | tee -a {output_dir}/logs/zimage.log
```

---

## Step 4: Verify Output

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
| Generator device error | `torch.Generator("xpu")` or `torch.Generator("cuda")` |

### NEVER DO
- Create venv
- pip install anything
