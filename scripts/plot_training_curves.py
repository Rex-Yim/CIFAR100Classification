#!/usr/bin/env python3
"""Parse Colab-style training logs (JSON metrics per epoch) and save PDF+PNG figures.

Default input: artifacts/colab_training_log_4epoch_plus_30_to_65pct.txt
Optional Stage C: artifacts/stage_c_training_history.json (JSON array)

Output: report/figures/training_curves_A_B_C.{pdf,png} when Stage C is present,
        else training_curves_A_B.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Colorblind-friendly (Wong / Okabe–Ito) + neutrals — common in ML papers
_COLORS = {
    "primary": "#0072B2",  # blue — test / main curve
    "train": "#D55E00",  # vermillion — training (under mixup: interpret with care)
    "val": "#009E73",  # bluish green — validation
    "fill": "#0072B2",
    "best": "#CC3311",
    "grid": "#DDDDDD",
    "muted": "#555555",
    "spine": "#333333",
}

# Shared accuracy scale (0–100%) for cross-stage comparison
SHARED_ACCURACY_YLIM: tuple[float, float] = (0.0, 100.0)


def extract_json_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    i = 0
    out: list[dict] = []
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, j = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and "epoch" in obj and "eval_accuracy" in obj:
                    out.append(obj)
                i = j
            except json.JSONDecodeError:
                i += 1
        else:
            i += 1
    return out


def split_stages(records: list[dict]) -> tuple[list[dict], list[dict]]:
    stage_a: list[dict] = []
    stage_b: list[dict] = []
    for r in records:
        sp = r.get("eval_split", "")
        if sp == "test":
            stage_a.append(r)
        elif sp == "val":
            stage_b.append(r)
    return stage_a, stage_b


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_COLORS["spine"])
    ax.spines["bottom"].set_color(_COLORS["spine"])
    ax.tick_params(axis="both", colors=_COLORS["spine"], width=0.8, length=4)
    ax.grid(True, linestyle="-", linewidth=0.6, color=_COLORS["grid"], alpha=0.85, zorder=0)
    ax.set_axisbelow(True)


def _panel_label(ax, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=_COLORS["spine"],
        va="top",
        ha="left",
        zorder=5,
    )


def _annotate_best(
    ax,
    x: float,
    y: float,
    text: str,
    xytext: tuple[float, float],
) -> None:
    ax.scatter(
        [x],
        [y],
        s=55,
        color=_COLORS["best"],
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=8,
        color=_COLORS["spine"],
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.9,
            "color": "#888888",
            "shrinkA": 4,
            "shrinkB": 4,
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "linewidth": 0.8,
        },
    )


def plot_train_val_panel(
    ax,
    records: list[dict],
    *,
    title: str,
    footnote: str,
    panel_letter: str,
    annotate_fmt: str = "Best val: {best:.2f}%\n(epoch {ep})",
    ylim: tuple[float, float] = SHARED_ACCURACY_YLIM,
) -> None:
    eb = [r["epoch"] for r in records]
    train_acc = [100.0 * r["train_accuracy"] for r in records]
    val_acc = [100.0 * r["eval_accuracy"] for r in records]
    best_idx = max(range(len(val_acc)), key=val_acc.__getitem__)

    ax.plot(
        eb,
        train_acc,
        "--",
        color=_COLORS["train"],
        linewidth=1.85,
        alpha=0.92,
        label="Train",
        zorder=2,
    )
    ax.plot(
        eb,
        val_acc,
        "-",
        color=_COLORS["val"],
        linewidth=2.1,
        alpha=0.98,
        label="Validation",
        zorder=2,
    )
    _annotate_best(
        ax,
        eb[best_idx],
        val_acc[best_idx],
        annotate_fmt.format(best=val_acc[best_idx], ep=int(eb[best_idx])),
        xytext=(-8, -42),
    )
    ax.set_xlabel("Epoch", fontsize=10, color=_COLORS["spine"])
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=_COLORS["spine"])
    ax.set_title(title, fontsize=10.5, fontweight="semibold", color=_COLORS["spine"], pad=8)
    ax.set_xlim(min(eb), max(eb))
    ax.set_ylim(*ylim)
    leg = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        fontsize=8.5,
        handlelength=2.4,
        borderpad=0.6,
    )
    leg.get_frame().set_linewidth(0.6)
    ax.text(
        0.02,
        0.93,
        footnote,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        color=_COLORS["muted"],
    )
    _panel_label(ax, panel_letter)
    _style_axis(ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "colab_training_log_4epoch_plus_30_to_65pct.txt",
    )
    ap.add_argument(
        "--stage-c-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "stage_c_training_history.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "report" / "figures",
    )
    args = ap.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    records = extract_json_objects(text)
    stage_a, stage_b = split_stages(records)

    stage_c: list[dict] = []
    if args.stage_c_json.is_file():
        raw = json.loads(args.stage_c_json.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            stage_c = [r for r in raw if isinstance(r, dict) and "epoch" in r and "eval_accuracy" in r]
            stage_c.sort(key=lambda r: r["epoch"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: pip install matplotlib"
        ) from e

    # Publication-oriented defaults (vector PDF stays crisp)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Helvetica",
                "Arial",
                "Liberation Sans",
            ],
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8.5,
            "axes.unicode_minus": False,
        }
    )

    ncols = 3 if stage_c else 2
    # ~7.2 in tall × 2.65 in height ≈ two-column width; tweak for 3 panels
    fig_w = 10.25 if stage_c else 6.85
    fig_h = 2.85
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    if ncols == 2:
        axes = [axes[0], axes[1], None]
    else:
        axes = [axes[0], axes[1], axes[2]]
    ax0, ax1, ax2 = axes

    if stage_a:
        ea = [r["epoch"] for r in stage_a]
        va = [100.0 * r["eval_accuracy"] for r in stage_a]
        best_idx = max(range(len(va)), key=va.__getitem__)
        ax0.plot(
            ea,
            va,
            "o-",
            color=_COLORS["primary"],
            linewidth=2.0,
            markersize=5.5,
            markerfacecolor=_COLORS["primary"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            clip_on=False,
            zorder=2,
            label="Test accuracy",
        )
        ax0.fill_between(ea, va, 0.0, color=_COLORS["fill"], alpha=0.12, zorder=1)
        _annotate_best(
            ax0,
            ea[best_idx],
            va[best_idx],
            f"Best: {va[best_idx]:.2f}%",
            xytext=(-6, 12),
        )
        ax0.set_xlabel("Epoch", fontsize=10, color=_COLORS["spine"])
        ax0.set_ylabel("Accuracy (%)", fontsize=10, color=_COLORS["spine"])
        ax0.set_title("Stage A — baseline", fontsize=10.5, fontweight="semibold", color=_COLORS["spine"], pad=8)
        ax0.set_xticks(ea)
        ax0.set_ylim(*SHARED_ACCURACY_YLIM)
        leg = ax0.legend(
            loc="lower right",
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            fontsize=8.5,
        )
        leg.get_frame().set_linewidth(0.6)
        ax0.text(
            0.02,
            0.93,
            "Eval: official test set",
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color=_COLORS["muted"],
        )
        _panel_label(ax0, "a")
        _style_axis(ax0)

    if stage_b:
        plot_train_val_panel(
            ax1,
            stage_b,
            title="Stage B — RandAugment finetune",
            footnote="Train 90% · val 10%",
            panel_letter="b",
        )

    if stage_c and ax2 is not None:
        plot_train_val_panel(
            ax2,
            stage_c,
            title="Stage C — Mixup / CutMix + EMA",
            footnote="Train acc. not clean (mixed batches)",
            panel_letter="c",
        )

    # No figure suptitle: avoids overlap with panel (b); the report caption titles the figure.

    base_name = "training_curves_A_B_C" if stage_c else "training_curves_A_B"
    base = args.out_dir / base_name
    fig.savefig(base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
