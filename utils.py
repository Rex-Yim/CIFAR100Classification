from __future__ import annotations

import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def cifar100_is_available(data_dir: str | Path) -> bool:
    root = Path(data_dir)
    required_files = [
        root / "cifar-100-python" / "train",
        root / "cifar-100-python" / "test",
    ]
    return all(path.exists() for path in required_files)


class LocalCIFAR100(Dataset):
    def __init__(self, root: str | Path, train: bool, transform=None) -> None:
        self.root = Path(root) / "cifar-100-python"
        self.transform = transform
        split = "train" if train else "test"
        with (self.root / split).open("rb") as file:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                )
                entry = pickle.load(file, encoding="latin1")

        self.data = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = entry["fine_labels"]
        self.coarse_targets = entry.get("coarse_labels")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def build_transforms(
    aug: str = "basic",
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    random_erasing_p: float = 0.0,
):
    train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if aug == "randaugment":
        train_steps.append(RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))
    elif aug != "basic":
        raise ValueError(f"Unsupported augmentation mode: {aug}")

    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    if random_erasing_p > 0:
        train_steps.append(
            transforms.RandomErasing(
                p=random_erasing_p,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
            )
        )
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, eval_transform


def _pin_memory_enabled() -> bool:
    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def _build_cifar100_datasets(
    data_dir: str | Path,
    train_transform,
    eval_transform,
):
    should_download = not cifar100_is_available(data_dir)
    root = str(data_dir)

    if should_download:
        train_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=train_transform,
        )
        train_eval_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=False,
            transform=eval_transform,
        )
        test_dataset = datasets.CIFAR100(
            root=root,
            train=False,
            download=False,
            transform=eval_transform,
        )
    else:
        train_dataset = LocalCIFAR100(root=data_dir, train=True, transform=train_transform)
        train_eval_dataset = LocalCIFAR100(root=data_dir, train=True, transform=eval_transform)
        test_dataset = LocalCIFAR100(root=data_dir, train=False, transform=eval_transform)
    return train_dataset, train_eval_dataset, test_dataset


def stratified_split_indices(
    targets: list[int] | np.ndarray,
    val_ratio: float,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 for validation splitting")

    targets_array = np.asarray(targets)
    rng = np.random.default_rng(split_seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in np.unique(targets_array):
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)
        val_count = max(1, int(round(len(class_indices) * val_ratio)))
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    aug: str = "basic",
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    random_erasing_p: float = 0.0,
    eval_split: str = "val",
    val_ratio: float = 0.1,
    split_seed: int = 42,
    final_train_full: bool = False,
):
    train_transform, eval_transform = build_transforms(
        aug=aug,
        randaugment_n=randaugment_n,
        randaugment_m=randaugment_m,
        random_erasing_p=random_erasing_p,
    )
    train_dataset, train_eval_dataset, test_dataset = _build_cifar100_datasets(
        data_dir=data_dir,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    pin_memory = _pin_memory_enabled()

    if final_train_full:
        resolved_eval_split = "test"
        train_subset = train_dataset
        eval_subset = test_dataset
    elif eval_split == "val":
        train_indices, val_indices = stratified_split_indices(
            targets=train_dataset.targets,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
        train_subset = Subset(train_dataset, train_indices)
        eval_subset = Subset(train_eval_dataset, val_indices)
        resolved_eval_split = "val"
    elif eval_split == "test":
        train_subset = train_dataset
        eval_subset = test_dataset
        resolved_eval_split = "test"
    else:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    info = {
        "train_size": len(train_subset),
        "eval_size": len(eval_subset),
        "eval_split": resolved_eval_split,
    }
    return train_loader, eval_loader, info


def build_test_dataset(data_dir: str | Path, transform=None):
    if cifar100_is_available(data_dir):
        return LocalCIFAR100(root=data_dir, train=False, transform=transform)
    return datasets.CIFAR100(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )


def build_cifar100_fine_to_coarse(data_dir: str | Path) -> list[int]:
    train_file = Path(data_dir) / "cifar-100-python" / "train"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Missing CIFAR-100 train file at {train_file}. Download the dataset first."
        )

    with train_file.open("rb") as file:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            )
            entry = pickle.load(file, encoding="latin1")

    mapping: dict[int, int] = {}
    for fine_label, coarse_label in zip(entry["fine_labels"], entry["coarse_labels"]):
        existing = mapping.get(fine_label)
        if existing is not None and existing != coarse_label:
            raise ValueError(
                f"Fine label {fine_label} maps to multiple coarse labels: {existing} and {coarse_label}"
            )
        mapping[fine_label] = coarse_label

    if len(mapping) != 100:
        raise ValueError(f"Expected 100 CIFAR-100 fine labels, found {len(mapping)}")

    return [mapping[index] for index in range(100)]


def build_cifar100_coarse_groups(fine_to_coarse: list[int]) -> list[list[int]]:
    num_coarse = max(fine_to_coarse) + 1
    coarse_groups: list[list[int]] = [[] for _ in range(num_coarse)]
    for fine_label, coarse_label in enumerate(fine_to_coarse):
        coarse_groups[coarse_label].append(fine_label)

    for coarse_label, group in enumerate(coarse_groups):
        if not group:
            raise ValueError(f"Coarse label {coarse_label} has no assigned fine labels")
    return coarse_groups


def fine_logits_to_coarse_logits(
    fine_logits: torch.Tensor,
    coarse_groups: list[torch.Tensor],
) -> torch.Tensor:
    coarse_logits = [
        torch.logsumexp(fine_logits[:, fine_indices], dim=1)
        for fine_indices in coarse_groups
    ]
    return torch.stack(coarse_logits, dim=1)


def mixup_data(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, float(lam)


def cutmix_data(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(inputs.size(0), device=inputs.device)
    targets_a = targets
    targets_b = targets[rand_index]

    height, width = inputs.size(2), inputs.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))

    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed_inputs, targets_a, targets_b, float(lam)


def mixup_criterion(
    criterion,
    predictions: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(predictions, targets_a) + (1.0 - lam) * criterion(
        predictions, targets_b
    )
