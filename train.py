from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model import build_model
from utils import (
    accuracy,
    build_cifar100_coarse_groups,
    build_cifar100_fine_to_coarse,
    build_dataloaders,
    cutmix_data,
    fine_logits_to_coarse_logits,
    mixup_criterion,
    mixup_data,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-100 classifier from scratch.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-from",
        type=str,
        default="",
        help="Load only model weights from this checkpoint; fresh optimizer/scheduler/history. "
        "Same architecture as the checkpoint. Use to finetune from e.g. the ~34%% SE-ResNet baseline.",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--eval-split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--final-train-full", action="store_true")

    parser.add_argument("--aug", type=str, choices=["basic", "randaugment"], default="basic")
    parser.add_argument("--randaugment-n", type=int, default=2)
    parser.add_argument("--randaugment-m", type=int, default=9)
    parser.add_argument("--random-erasing-p", type=float, default=0.1)

    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--use-hierarchical-loss", action="store_true")
    parser.add_argument("--coarse-loss-weight", type=float, default=0.25)
    parser.add_argument("--mix-mode", type=str, choices=["none", "mixup", "cutmix", "both"], default="both")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)

    parser.add_argument("--model-name", type=str, choices=["se_resnet", "wrn_hydra"], default="wrn_hydra")
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=10)
    parser.add_argument("--attention", type=str, choices=["none", "eca", "se"], default="eca")
    parser.add_argument(
        "--downsample-mode",
        type=str,
        choices=["stride", "antialias"],
        default="antialias",
    )

    parser.add_argument("--base-width", type=int, default=96)
    parser.add_argument("--blocks-per-stage", type=int, nargs=4, default=[3, 4, 6, 3])
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    mix_mode: str,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
    ema_model: nn.Module | None,
    ema_decay: float,
    fine_to_coarse: torch.Tensor | None,
    coarse_groups: list[torch.Tensor] | None,
    coarse_loss_weight: float,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_coarse_loss = 0.0

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_a, targets_b, lam = targets, targets, 1.0

        optimizer.zero_grad(set_to_none=True)

        if mix_mode != "none" and torch.rand(1).item() < mix_prob:
            if mix_mode == "mixup":
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
            elif mix_mode == "cutmix":
                images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
            else:
                if torch.rand(1).item() < 0.5:
                    images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
                else:
                    images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)

        with torch.autocast(device_type=device.type, enabled=device.type in {"cuda", "mps"}):
            outputs = model(images)
            fine_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            coarse_loss = torch.zeros((), device=device)
            if fine_to_coarse is not None and coarse_groups is not None and coarse_loss_weight > 0:
                coarse_logits = fine_logits_to_coarse_logits(outputs, coarse_groups)
                coarse_targets_a = fine_to_coarse[targets_a]
                coarse_targets_b = fine_to_coarse[targets_b]
                coarse_loss = mixup_criterion(
                    criterion,
                    coarse_logits,
                    coarse_targets_a,
                    coarse_targets_b,
                    lam,
                )
            loss = fine_loss + coarse_loss_weight * coarse_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            update_ema(ema_model, model, ema_decay)

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, targets) * images.size(0)
        running_coarse_loss += coarse_loss.item() * images.size(0)

    total = len(loader.dataset)
    return running_loss / total, running_acc / total, running_coarse_loss / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    fine_to_coarse: torch.Tensor | None,
    coarse_groups: list[torch.Tensor] | None,
    coarse_loss_weight: float,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_coarse_loss = 0.0

    for images, targets in tqdm(loader, desc="Eval ", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=device.type in {"cuda", "mps"}):
            outputs = model(images)
            fine_loss = criterion(outputs, targets)
            coarse_loss = torch.zeros((), device=device)
            if fine_to_coarse is not None and coarse_groups is not None and coarse_loss_weight > 0:
                coarse_logits = fine_logits_to_coarse_logits(outputs, coarse_groups)
                coarse_targets = fine_to_coarse[targets]
                coarse_loss = criterion(coarse_logits, coarse_targets)
            loss = fine_loss + coarse_loss_weight * coarse_loss

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, targets) * images.size(0)
        running_coarse_loss += coarse_loss.item() * images.size(0)

    total = len(loader.dataset)
    return running_loss / total, running_acc / total, running_coarse_loss / total


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def build_scheduler(args: argparse.Namespace, optimizer):
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    cosine_epochs = max(args.epochs - warmup_epochs, 1)
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key, value in model_state.items():
            if not torch.is_floating_point(value):
                ema_state[key].copy_(value)
            else:
                ema_state[key].mul_(decay).add_(value, alpha=1.0 - decay)


def build_model_from_args(args: argparse.Namespace) -> nn.Module:
    if args.model_name == "wrn_hydra":
        return build_model(
            model_name="wrn_hydra",
            num_classes=100,
            depth=args.depth,
            widen_factor=args.widen_factor,
            attention=args.attention,
            downsample_mode=args.downsample_mode,
        )
    return build_model(
        model_name="se_resnet",
        num_classes=100,
        base_width=args.base_width,
        blocks_per_stage=tuple(args.blocks_per_stage),
        max_drop_path_rate=args.drop_path_rate,
        use_se=True,
    )


def resolve_run_dir(args: argparse.Namespace) -> Path:
    base_dir = Path(args.save_dir)
    return base_dir / args.run_name if args.run_name else base_dir


def resolve_resume_path(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if not args.resume:
        return None

    if args.resume == "latest":
        candidate = run_dir / "last.pt"
        return candidate if candidate.exists() else None

    candidate = Path(args.resume)
    if not candidate.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    return candidate


def write_run_args(run_dir: Path, args: argparse.Namespace, resolved_eval_split: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    args_path = run_dir / "args.json"
    payload = vars(args).copy()
    payload["resolved_eval_split"] = resolved_eval_split
    args_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    best_eval_accuracy: float,
    history: list[dict],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_state = scaler.state_dict() if scaler.is_enabled() else None
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler_state,
            "history": history,
            "best_eval_accuracy": best_eval_accuracy,
            "num_classes": 100,
            "model_name": args.model_name,
            "depth": args.depth,
            "widen_factor": args.widen_factor,
            "attention": args.attention,
            "downsample_mode": args.downsample_mode,
            "base_width": args.base_width,
            "blocks_per_stage": tuple(args.blocks_per_stage),
            "drop_path_rate": args.drop_path_rate,
            "use_se": True,
            "args": vars(args),
        },
        save_path,
    )


def maybe_build_ema_model(model: nn.Module, args: argparse.Namespace, device: torch.device):
    if args.ema_decay <= 0:
        return None
    ema_model = copy.deepcopy(model).to(device)
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


def load_resume_state(
    resume_path: Path | None,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    history_path: Path,
    device: torch.device,
) -> tuple[int, float, list[dict]]:
    if resume_path is None:
        return 1, 0.0, []

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    ema_state = checkpoint.get("ema_model_state_dict")
    if ema_model is not None and ema_state is not None:
        ema_model.load_state_dict(ema_state)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None and scaler.is_enabled():
        scaler.load_state_dict(scaler_state)

    history = checkpoint.get("history", [])
    if not history and history_path.exists():
        history = json.loads(history_path.read_text())

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_eval_accuracy = checkpoint.get(
        "best_eval_accuracy",
        checkpoint.get("best_test_accuracy", 0.0),
    )
    return start_epoch, best_eval_accuracy, history


def load_init_from_weights(
    init_path: Path,
    model: nn.Module,
    ema_model: nn.Module | None,
    device: torch.device,
) -> None:
    checkpoint = torch.load(init_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ema_state = checkpoint.get("ema_model_state_dict")
    if ema_model is not None:
        if ema_state is not None:
            ema_model.load_state_dict(ema_state, strict=True)
        else:
            ema_model.load_state_dict(model.state_dict())


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Fixed image size (CIFAR) — benchmark picks faster conv algorithms (often faster on Colab T4/L4).
        torch.backends.cudnn.benchmark = True
    run_dir = resolve_run_dir(args)
    history_path = run_dir / "history.json"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    resolved_eval_split = "test" if args.final_train_full else args.eval_split
    write_run_args(run_dir, args, resolved_eval_split)

    train_loader, eval_loader, data_info = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug=args.aug,
        randaugment_n=args.randaugment_n,
        randaugment_m=args.randaugment_m,
        random_erasing_p=args.random_erasing_p,
        eval_split=args.eval_split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        final_train_full=args.final_train_full,
    )

    model = build_model_from_args(args).to(device)
    ema_model = maybe_build_ema_model(model, args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    fine_to_coarse = None
    coarse_groups = None
    if args.use_hierarchical_loss and args.coarse_loss_weight > 0:
        fine_to_coarse_list = build_cifar100_fine_to_coarse(args.data_dir)
        fine_to_coarse = torch.tensor(fine_to_coarse_list, dtype=torch.long, device=device)
        coarse_groups = [
            torch.tensor(group, dtype=torch.long, device=device)
            for group in build_cifar100_coarse_groups(fine_to_coarse_list)
        ]

    resume_path = resolve_resume_path(args, run_dir)
    init_from = Path(args.init_from) if args.init_from else None
    if init_from is not None and resume_path is not None:
        raise ValueError("Use either --init-from or --resume, not both.")
    if init_from is not None:
        if not init_from.is_file():
            raise FileNotFoundError(f"--init-from checkpoint not found: {init_from}")
        load_init_from_weights(init_from, model, ema_model, device)
        start_epoch, best_eval_accuracy, history = 1, 0.0, []
    elif resume_path is not None:
        start_epoch, best_eval_accuracy, history = load_resume_state(
            resume_path=resume_path,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history_path=history_path,
            device=device,
        )
    else:
        start_epoch, best_eval_accuracy, history = 1, 0.0, []

    print(f"Training split size: {data_info['train_size']}", flush=True)
    print(f"Evaluation split: {data_info['eval_split']} ({data_info['eval_size']} samples)", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    if init_from is not None:
        print(f"Initialized weights from: {init_from}", flush=True)
    elif resume_path is not None:
        print(f"Resuming from: {resume_path}", flush=True)
    if fine_to_coarse is not None:
        print(f"Hierarchical loss enabled with weight {args.coarse_loss_weight:.2f}", flush=True)
    if args.num_threads > 0:
        print(f"CPU threads: {args.num_threads}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_loss, train_acc, train_coarse_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            mix_mode=args.mix_mode,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
            scaler=scaler,
            grad_clip=args.grad_clip,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            fine_to_coarse=fine_to_coarse,
            coarse_groups=coarse_groups,
            coarse_loss_weight=args.coarse_loss_weight,
        )

        eval_model = ema_model if ema_model is not None else model
        eval_loss, eval_acc, eval_coarse_loss = evaluate(
            model=eval_model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            fine_to_coarse=fine_to_coarse,
            coarse_groups=coarse_groups,
            coarse_loss_weight=args.coarse_loss_weight,
        )
        scheduler.step()

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_coarse_loss": train_coarse_loss,
            "eval_split": data_info["eval_split"],
            "eval_loss": eval_loss,
            "eval_accuracy": eval_acc,
            "eval_coarse_loss": eval_coarse_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, indent=2), flush=True)

        save_checkpoint(
            save_path=last_path,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch,
            best_eval_accuracy=max(best_eval_accuracy, eval_acc),
            history=history,
        )

        if eval_acc > best_eval_accuracy:
            best_eval_accuracy = eval_acc
            save_checkpoint(
                save_path=best_path,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                best_eval_accuracy=best_eval_accuracy,
                history=history,
            )
            print(f"New best {data_info['eval_split']} accuracy: {best_eval_accuracy:.4f}", flush=True)

        history_path.write_text(json.dumps(history, indent=2))

    print(
        f"\nTraining finished. Best {data_info['eval_split']} accuracy: {best_eval_accuracy:.4f}",
        flush=True,
    )
    print(f"Best checkpoint saved to: {best_path}", flush=True)


if __name__ == "__main__":
    main()
