"""Compress LUVIA CNN weight files from float32 to float16 in place.

Halves the size of the bundled .pt files (7 x ~19MB -> ~10MB each). Straw's
load_model upcasts to float32 at load time so inference is unchanged.

Run from anywhere with the luvia env active:
    python packaging/compress_weights.py
"""

import os
import pathlib

import torch

WEIGHTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "luvia" / "data" / "weights"

print(f"==> Compressing weights in {WEIGHTS_DIR}")
for path in sorted(WEIGHTS_DIR.glob("*.pt")):
    before = path.stat().st_size
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    halved = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    torch.save(halved, str(path))
    after = path.stat().st_size
    print(f"  {path.name}: {before/1024**2:.1f} MB -> {after/1024**2:.1f} MB")
