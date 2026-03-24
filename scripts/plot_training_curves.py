#!/usr/bin/env python3
"""Parse Colab-style training logs (JSON metrics per epoch) and save PDF+PNG figures.

Default input: artifacts/colab_training_log_4epoch_plus_30_to_65pct.txt
Output: report/figures/training_curves_A_B.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "colab_training_log_4epoch_plus_30_to_65pct.txt",
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

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: pip install matplotlib"
        ) from e

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True)

    if stage_a:
        ea = [r["epoch"] for r in stage_a]
        va = [100.0 * r["eval_accuracy"] for r in stage_a]
        best_idx = max(range(len(va)), key=va.__getitem__)
        ax0.plot(ea, va, "o-", color="#1f77b4", linewidth=2.4, markersize=6.5)
        ax0.fill_between(ea, va, [min(va) - 2] * len(va), color="#1f77b4", alpha=0.08)
        ax0.scatter([ea[best_idx]], [va[best_idx]], s=70, color="#d62728", zorder=3)
        ax0.annotate(
            f"Best: {va[best_idx]:.2f}%",
            xy=(ea[best_idx], va[best_idx]),
            xytext=(-45, 14),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444"},
        )
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Accuracy (%)")
        ax0.set_title("Stage A: baseline")
        ax0.set_xticks(ea)
        ax0.set_ylim(max(0, min(va) - 4), max(va) + 6)
        ax0.text(
            0.03,
            0.95,
            "Eval split: official test",
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#444",
        )

    if stage_b:
        eb = [r["epoch"] for r in stage_b]
        train_acc = [100.0 * r["train_accuracy"] for r in stage_b]
        val_acc = [100.0 * r["eval_accuracy"] for r in stage_b]
        best_idx = max(range(len(val_acc)), key=val_acc.__getitem__)
        ax1.plot(
            eb,
            train_acc,
            "-",
            color="#ff7f0e",
            linewidth=2.0,
            alpha=0.95,
            label="Train accuracy",
        )
        ax1.plot(
            eb,
            val_acc,
            "o-",
            color="#2ca02c",
            linewidth=2.3,
            markersize=4.0,
            label="Validation accuracy",
        )
        ax1.scatter([eb[best_idx]], [val_acc[best_idx]], s=70, color="#d62728", zorder=3)
        ax1.annotate(
            f"Best: {val_acc[best_idx]:.2f}%\nEpoch {eb[best_idx]}",
            xy=(eb[best_idx], val_acc[best_idx]),
            xytext=(-55, -38),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444"},
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Stage B: RandAugment finetune")
        ax1.set_xlim(min(eb), max(eb))
        ax1.set_ylim(min(train_acc) - 2, max(val_acc) + 3)
        ax1.legend(loc="lower right")
        ax1.text(
            0.03,
            0.95,
            "Train split: 90% | Eval split: 10% val",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#444",
        )

    fig.suptitle("CIFAR-100 SE-ResNet training curves", fontsize=14)

    base = args.out_dir / "training_curves_A_B"
    for ext in ("pdf", "png"):
        fig.savefig(base.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
