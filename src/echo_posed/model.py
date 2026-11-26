"""The EfficientNet-B0 ensemble and its two SO(3) outputs."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .pose import TARGETS, angle, project_so3


INPUT_SIZE = 224
BUNDLE_FORMAT = "echogsd_checkpoint_bundle"
MODEL_CONFIG = {
    "architecture": "efficientnet_b0",
    "backbone_weights": "imagenet",
    "in_channels": 1,
    "mode": "9d_svd",
    "num_rotation_targets": 2,
    "world_head_so3": True,
    "world_rotation_frame": "probe_local",
}


class PoseRegressor(nn.Module):
    def __init__(self, *, pretrained: bool = False):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights if pretrained else None)
        features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()
        self.head_rotation = nn.Linear(features, 18, bias=False)
        self.register_buffer(
            "mean",
            torch.tensor(weights.transforms().mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(weights.transforms().std).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rgb = (images.float().repeat(1, 3, 1, 1) - self.mean) / self.std
        raw = self.head_rotation(self.backbone(rgb)).float().reshape(-1, 2, 3, 3)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            return torch.stack([project_so3(raw[:, factor]) for factor in (0, 1)], 1)


@dataclass(frozen=True)
class Prediction:
    rotations: torch.Tensor
    uncertainty_rad: torch.Tensor


class Ensemble(nn.Module):
    def __init__(self, models: list[PoseRegressor], adapters: torch.Tensor):
        super().__init__()
        if not models or adapters.shape != (len(models), 2, 3, 3):
            raise ValueError("Expected two alignment matrices per ensemble member")
        identity = torch.eye(3, device=adapters.device, dtype=adapters.dtype).expand_as(
            adapters
        )
        if (
            not torch.allclose(adapters.mT @ adapters, identity, atol=1e-4, rtol=0)
            or (torch.linalg.det(adapters) <= 0).any()
        ):
            raise ValueError("Alignment matrices must be proper rotations")
        self.models = nn.ModuleList(models)
        self.register_buffer("adapters", adapters)

    @classmethod
    def load(cls, checkpoint: str | Path, *, device: str = "cpu"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != BUNDLE_FORMAT
            or payload.get("version") != 2
            or payload.get("input_size") != INPUT_SIZE
            or tuple(payload.get("rotation_target_names", ())) != TARGETS
            or any(
                payload.get("model_config", {}).get(key) != value
                for key, value in MODEL_CONFIG.items()
            )
        ):
            raise ValueError(f"Unsupported Echo-POSED checkpoint: {checkpoint}")
        models = []
        for state in payload["model_state_dicts"]:
            model = PoseRegressor()
            model.load_state_dict(state)
            models.append(model)
        adapters = payload["gauge_alignment"]["alignments"].float()
        return cls(models, adapters).to(device).eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> Prediction:
        if images.ndim != 4 or images.shape[1:] != (1, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(
                f"Expected [B,1,{INPUT_SIZE},{INPUT_SIZE}] scan-converted frames"
            )
        images = images.to(self.adapters.device)
        members = self.adapters[:, None] @ torch.stack(
            [model(images) for model in self.models]
        )
        mean = project_so3(members.mean(dim=0))
        uncertainty = angle(mean.mT @ members).square().mean(dim=(0, 2)).sqrt()
        return Prediction(mean, uncertainty)
