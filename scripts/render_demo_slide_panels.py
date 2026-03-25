#!/usr/bin/env python3
"""Build two stacked prediction panels (sample image + top-5 bars) for Beamer slides.

Picks one sample with the *highest* top-1 confidence (works well) and one with the
*lowest* top-1 confidence (model uncertain / wrong) among ``demo/static/samples/*.png``.

Requires a checkpoint (see demo/app.py). Run from repo root:
  python3 scripts/render_demo_slide_panels.py

Writes: report/figures/demo_two_panels.png (also used by slides via that path)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
    }
)

# Match demo/static/style.css
BG = "#0f1419"
SURFACE = "#1a222d"
TEXT = "#e8eef4"
MUTED = "#8b9aab"
ACCENT = "#3db8a6"
WARN = "#e07070"


def _pick_best_worst_samples() -> tuple[tuple, tuple]:
    """Return (best_tuple, worst_tuple) each (top1_prob, path, pred_label)."""
    from demo.app import SAMPLES_DIR, _preprocess, get_model

    paths = sorted(SAMPLES_DIR.glob("*.png"))
    if len(paths) < 2:
        raise SystemExit("Need at least two sample PNGs in demo/static/samples/")

    model, device, labels = get_model()
    scored: list[tuple[float, Path, str]] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        x = _preprocess(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        top1 = float(probs.max().item())
        pred = labels[int(probs.argmax().item())]
        scored.append((top1, path, pred))

    best = max(scored, key=lambda t: t[0])
    worst = min(scored, key=lambda t: t[0])
    if best[1] == worst[1]:
        scored.sort(key=lambda t: t[0])
        worst = scored[0]
        best = scored[-1]
    return best, worst


def main() -> None:
    from demo.app import _preprocess, get_model

    best, worst = _pick_best_worst_samples()
    model, device, labels = get_model()

    rows: list[tuple[Path, str, str, str]] = [
        (best[1], "A", "Strong prediction", f"top-1 {best[0]*100:.1f}%"),
        (worst[1], "B", "Hard case (uncertain)", f"top-1 {worst[0]*100:.1f}%"),
    ]

    # Wide canvas + generous column gap so bar-chart y-labels never overlap the 32×32 crops
    fig = plt.figure(figsize=(8.2, 5.4), dpi=200, facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 2.55], hspace=0.42, wspace=0.52)

    for row, (path, tag, row_title, conf_line) in enumerate(rows):
        img = Image.open(path).convert("RGB")
        x = _preprocess(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        values, indices = torch.topk(probs, k=5)

        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.set_facecolor(SURFACE)
        ax_img.imshow(np.array(img), interpolation="nearest")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_color("#2d3a4a")
            spine.set_linewidth(1.0)
        ax_img.set_title(f"{tag}: {row_title}", color=TEXT, fontsize=8, weight="semibold", pad=6)
        ax_img.text(
            0.5,
            -0.12,
            path.name,
            transform=ax_img.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=ACCENT,
            family="monospace",
        )
        ax_img.text(
            0.5,
            -0.22,
            conf_line,
            transform=ax_img.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color=WARN if row == 1 else MUTED,
        )

        ax_bar = fig.add_subplot(gs[row, 1])
        ax_bar.set_facecolor(SURFACE)
        bar_names = [labels[i.item()] for i in indices]
        vals = [values[i].item() * 100 for i in range(5)]
        y_pos = np.arange(5)
        colors = [ACCENT if i == 0 else "#2a8f82" for i in range(5)]
        ax_bar.barh(y_pos, vals, color=colors, height=0.65, edgecolor="#2d3a4a", linewidth=0.5)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(bar_names, color=TEXT, fontsize=7, ha="right")
        ax_bar.invert_yaxis()
        ax_bar.yaxis.set_tick_params(pad=10, labelleft=True)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel("Probability (%)", color=MUTED, fontsize=7)
        ax_bar.tick_params(axis="x", colors=MUTED, labelsize=7)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax_bar.spines[s].set_color("#2d3a4a")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.25, color="#5a6a7a")

    out = ROOT / "report" / "figures" / "demo_two_panels.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.14, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  Strong: {best[1].name} (top-1 {best[0]*100:.2f}% → {best[2]})")
    print(f"  Weak:   {worst[1].name} (top-1 {worst[0]*100:.2f}% → {worst[2]})")


if __name__ == "__main__":
    main()
