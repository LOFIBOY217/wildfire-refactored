"""
4-GPU DataParallel wrapper for train_s2s_hotspot_cwfis_v2.py
=============================================================
This script does NOT modify the original training script.
It reads the source at runtime, applies 3 minimal DataParallel patches,
and executes the patched code.

Patches applied:
  1. After model.to(device): wrap with nn.DataParallel if >1 GPU available
  2. Checkpoint save: use model.module.state_dict() for DataParallel models
  3. Post-training load: use model.module.load_state_dict() for DataParallel models

Usage:
    python -m src.training.train_s2s_hotspot_cwfis_v2_4gpu \\
        --config configs/paths_trillium.yaml \\
        --num_workers 12 --batch_size 512 --epochs 10 --load_to_ram
"""

import os
import sys
import torch

src_path = os.path.join(os.path.dirname(__file__), "train_s2s_hotspot_cwfis_v2.py")

with open(src_path, "r") as f:
    code = f.read()

# Patch 1: wrap model with DataParallel after .to(device)
_orig_to_device = "    ).to(device)\n\n    n_params = sum"
_patched_to_device = (
    "    ).to(device)\n"
    "    if torch.cuda.device_count() > 1:\n"
    "        print(f'  [DataParallel] Using {torch.cuda.device_count()} GPUs')\n"
    "        model = torch.nn.DataParallel(model)\n"
    "\n    n_params = sum"
)
assert _orig_to_device in code, "Patch 1 anchor not found — original script may have changed"
code = code.replace(_orig_to_device, _patched_to_device, 1)

# Patch 2: checkpoint save — use model.module.state_dict() for DataParallel
_orig_save = '                "model_state":    model.state_dict(),'
_patched_save = (
    '                "model_state":    model.module.state_dict()'
    " if isinstance(model, torch.nn.DataParallel) else model.state_dict(),"
)
assert _orig_save in code, "Patch 2 anchor not found — original script may have changed"
code = code.replace(_orig_save, _patched_save, 1)

# Patch 3: post-training load before forecast TIF generation
_orig_load = (
    "    model.load_state_dict(ckpt[\"model_state\"])\n"
    "    model.eval()\n"
    "\n"
    "    out_profile = profile.copy()"
)
_patched_load = (
    "    (model.module if isinstance(model, torch.nn.DataParallel) else model)"
    ".load_state_dict(ckpt[\"model_state\"])\n"
    "    model.eval()\n"
    "\n"
    "    out_profile = profile.copy()"
)
assert _orig_load in code, "Patch 3 anchor not found — original script may have changed"
code = code.replace(_orig_load, _patched_load, 1)

print("[4gpu wrapper] All 3 DataParallel patches applied successfully.")
print(f"[4gpu wrapper] Visible GPUs: {torch.cuda.device_count()}")

exec(compile(code, src_path, "exec"), {"__name__": "__main__", "__file__": src_path})
