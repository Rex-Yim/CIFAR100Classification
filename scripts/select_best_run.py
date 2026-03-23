from __future__ import annotations

import json
import sys
from pathlib import Path


def best_record(run_dir: Path) -> dict:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    history = json.loads(history_path.read_text())
    if not history:
        raise ValueError(f"Empty history file: {history_path}")

    best = max(history, key=lambda item: item.get("eval_accuracy", item.get("test_accuracy", 0.0)))
    return {
        "run_dir": str(run_dir),
        "best_epoch": best["epoch"],
        "eval_split": best.get("eval_split", "test"),
        "best_accuracy": best.get("eval_accuracy", best.get("test_accuracy", 0.0)),
        "best_loss": best.get("eval_loss", best.get("test_loss", 0.0)),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: select_best_run.py <run_dir> [<run_dir> ...]", file=sys.stderr)
        return 1

    summaries = [best_record(Path(arg)) for arg in argv[1:]]
    winner = max(summaries, key=lambda item: (item["best_accuracy"], -item["best_loss"]))
    print(json.dumps({"winner": winner, "candidates": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
