from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ECABlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x).squeeze(-1).transpose(-1, -2)
        scale = self.conv(scale)
        scale = scale.transpose(-1, -2).unsqueeze(-1)
        scale = torch.sigmoid(scale)
        return x * scale


class BlurPool2d(nn.Module):
    def __init__(self, channels: int, stride: int = 2) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.channels = channels
        self.stride = stride
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(dtype=x.dtype, device=x.device)
        x = self.pad(x)
        return F.conv2d(x, kernel, stride=self.stride, groups=self.channels)


def build_attention(attention: str, channels: int) -> nn.Module:
    if attention == "none":
        return nn.Identity()
    if attention == "eca":
        return ECABlock(channels)
    if attention == "se":
        return SqueezeExcitation(channels)
    raise ValueError(f"Unsupported attention mode: {attention}")


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        drop_path_rate: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.drop_path_rate = drop_path_rate

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self._drop_path(out)
        out = out + identity
        return F.relu(out, inplace=True)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_rate == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SEResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        base_width: int = 96,
        blocks_per_stage: tuple[int, int, int, int] = (3, 4, 6, 3),
        max_drop_path_rate: float = 0.1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        total_blocks = sum(blocks_per_stage)
        drop_rates = torch.linspace(0, max_drop_path_rate, total_blocks).tolist()
        block_index = 0

        self.layer1, block_index = self._make_stage(
            base_width,
            base_width,
            blocks=blocks_per_stage[0],
            stride=1,
            drop_rates=drop_rates,
            block_index=block_index,
            use_se=use_se,
        )
        self.layer2, block_index = self._make_stage(
            base_width,
            base_width * 2,
            blocks=blocks_per_stage[1],
            stride=2,
            drop_rates=drop_rates,
            block_index=block_index,
            use_se=use_se,
        )
        self.layer3, block_index = self._make_stage(
            base_width * 2,
            base_width * 4,
            blocks=blocks_per_stage[2],
            stride=2,
            drop_rates=drop_rates,
            block_index=block_index,
            use_se=use_se,
        )
        self.layer4, block_index = self._make_stage(
            base_width * 4,
            base_width * 8,
            blocks=blocks_per_stage[3],
            stride=2,
            drop_rates=drop_rates,
            block_index=block_index,
            use_se=use_se,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(base_width * 8, num_classes)

        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        drop_rates: list[float],
        block_index: int,
        use_se: bool,
    ) -> tuple[nn.Sequential, int]:
        layers = [
            SEResNetBlock(
                in_channels,
                out_channels,
                stride=stride,
                drop_path_rate=drop_rates[block_index],
                use_se=use_se,
            )
        ]
        block_index += 1
        for _ in range(1, blocks):
            layers.append(
                SEResNetBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    drop_path_rate=drop_rates[block_index],
                    use_se=use_se,
                )
            )
            block_index += 1
        return nn.Sequential(*layers), block_index

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class WRNHydraBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        attention: str,
        downsample_mode: str,
    ) -> None:
        super().__init__()
        self.use_projection = stride != 1 or in_channels != out_channels
        self.downsample_mode = downsample_mode

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        residual_stride = stride
        self.residual_downsample = nn.Identity()
        self.shortcut_downsample = nn.Identity()
        if stride > 1 and downsample_mode == "antialias":
            residual_stride = 1
            self.residual_downsample = BlurPool2d(in_channels, stride=stride)
            self.shortcut_downsample = BlurPool2d(in_channels, stride=stride)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=residual_stride,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.attention = build_attention(attention, out_channels)

        if self.use_projection:
            shortcut_stride = stride if downsample_mode == "stride" else 1
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=shortcut_stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        shortcut_input = out if self.use_projection else x
        out = self.residual_downsample(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out = self.attention(out)

        shortcut = self.shortcut_downsample(shortcut_input)
        shortcut = self.shortcut(shortcut)
        return out + shortcut


class WRNHydra(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        depth: int = 28,
        widen_factor: int = 10,
        attention: str = "eca",
        downsample_mode: str = "antialias",
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WRN depth must satisfy (depth - 4) % 6 == 0")
        if attention not in {"none", "eca", "se"}:
            raise ValueError(f"Unsupported attention mode: {attention}")
        if downsample_mode not in {"stride", "antialias"}:
            raise ValueError(f"Unsupported downsample mode: {downsample_mode}")

        blocks_per_stage = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.stem = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(
            in_channels=widths[0],
            out_channels=widths[1],
            num_blocks=blocks_per_stage,
            stride=1,
            attention=attention,
            downsample_mode=downsample_mode,
        )
        self.stage2 = self._make_stage(
            in_channels=widths[1],
            out_channels=widths[2],
            num_blocks=blocks_per_stage,
            stride=2,
            attention=attention,
            downsample_mode=downsample_mode,
        )
        self.stage3 = self._make_stage(
            in_channels=widths[2],
            out_channels=widths[3],
            num_blocks=blocks_per_stage,
            stride=2,
            attention=attention,
            downsample_mode=downsample_mode,
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        attention: str,
        downsample_mode: str,
    ) -> nn.Sequential:
        layers = [
            WRNHydraBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                attention=attention,
                downsample_mode=downsample_mode,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                WRNHydraBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    attention=attention,
                    downsample_mode=downsample_mode,
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.relu(self.bn(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(
    model_name: str = "wrn_hydra",
    num_classes: int = 100,
    base_width: int = 96,
    blocks_per_stage: tuple[int, int, int, int] = (3, 4, 6, 3),
    max_drop_path_rate: float = 0.1,
    use_se: bool = True,
    depth: int = 28,
    widen_factor: int = 10,
    attention: str = "eca",
    downsample_mode: str = "antialias",
) -> nn.Module:
    if model_name == "se_resnet":
        return SEResNet(
            num_classes=num_classes,
            base_width=base_width,
            blocks_per_stage=blocks_per_stage,
            max_drop_path_rate=max_drop_path_rate,
            use_se=use_se,
        )
    if model_name == "wrn_hydra":
        return WRNHydra(
            num_classes=num_classes,
            depth=depth,
            widen_factor=widen_factor,
            attention=attention,
            downsample_mode=downsample_mode,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    model_name = checkpoint.get("model_name", "se_resnet")
    if model_name == "wrn_hydra":
        return build_model(
            model_name="wrn_hydra",
            num_classes=checkpoint.get("num_classes", 100),
            depth=checkpoint.get("depth", 28),
            widen_factor=checkpoint.get("widen_factor", 10),
            attention=checkpoint.get("attention", "eca"),
            downsample_mode=checkpoint.get("downsample_mode", "antialias"),
        )

    return build_model(
        model_name="se_resnet",
        num_classes=checkpoint.get("num_classes", 100),
        base_width=checkpoint.get("base_width", 96),
        blocks_per_stage=tuple(checkpoint.get("blocks_per_stage", (3, 4, 6, 3))),
        max_drop_path_rate=checkpoint.get("drop_path_rate", 0.1),
        use_se=checkpoint.get("use_se", True),
    )
