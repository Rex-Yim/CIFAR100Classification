#!/usr/bin/env python3
"""Export a small grid of CIFAR-100 *test* images as PNGs for the web demo.

Run from repo root (requires data/cifar-100-python):
  python3 scripts/export_demo_sample_images.py

Writes to demo/static/samples/*.png and does not overwrite unrelated files.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import LocalCIFAR100


def _fine_labels(data_dir: Path) -> list[str]:
    meta = data_dir / "cifar-100-python" / "meta"
    with meta.open("rb") as f:
        return list(pickle.load(f, encoding="latin1")["fine_label_names"])


def main() -> None:
    data_dir = ROOT / "data"
    test_root = data_dir / "cifar-100-python" / "test"
    if not test_root.is_file():
        print(f"Missing CIFAR-100 test data: {test_root}", file=sys.stderr)
        print("Download/extract CIFAR-100 under data/ first.", file=sys.stderr)
        sys.exit(1)

    labels = _fine_labels(data_dir)
    ds = LocalCIFAR100(root=data_dir, train=False, transform=None)
    n = 36
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)

    out = ROOT / "demo" / "static" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    for i, idx in enumerate(indices):
        img_pil, y = ds[int(idx)]
        name = f"{i:02d}_{labels[y]}.png"
        img_pil.save(out / name, format="PNG")

    print(f"Wrote {n} images to {out}")


if __name__ == "__main__":
    main()
