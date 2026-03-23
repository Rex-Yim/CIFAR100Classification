from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import build_model_from_checkpoint
from utils import CIFAR100_MEAN, CIFAR100_STD, build_test_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained CIFAR-100 model and evaluate it on the test set."
    )
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictions-file", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(
        checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    pin_memory = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    testset = build_test_dataset(data_dir=args.data_dir, transform=transform)
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    total = 0
    correct = 0
    predictions = []

    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        predictions.extend(preds.cpu().tolist())

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f}")

    if args.predictions_file:
        predictions_path = Path(args.predictions_file)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "prediction"])
            for idx, pred in enumerate(predictions):
                writer.writerow([idx, pred])
        print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
