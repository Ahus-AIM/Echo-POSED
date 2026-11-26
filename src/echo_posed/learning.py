"""Train and evaluate dual-rotation ensembles using EchoXFlow patient folds."""

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .cartesian import prepare_cache
from .data import VOLUME_STREAM, eligible, load_volume, recordings, volume_geometry
from .imaging import reslice
from .model import BUNDLE_FORMAT, INPUT_SIZE, MODEL_CONFIG, Ensemble, PoseRegressor
from .pose import TARGETS, angle, axis_angle_rotation, probe_rotation
from .sampling import SampledRecordings, TrainingRecording, oversample_deep


DEVELOPMENT_FOLDS = (0, 1, 2, 3)
TEST_FOLD = 4
LEARNING_RATE = 0.0037
FINAL_LEARNING_RATE = 0.000037
ALIGNMENT_RECORDINGS = 512
ALIGNMENT_VIEWS = 16


def select_records(root, folds, limit):
    """Eligible 3D recordings of the requested folds in catalog order."""
    selected = []
    counts = dict.fromkeys(folds, 0)
    for record in recordings(root, content=VOLUME_STREAM):
        if record.fold not in counts:
            continue
        if limit is not None and (
            counts[record.fold] >= limit or not record.path().is_dir()
        ):
            continue
        if eligible(record):
            selected.append(record)
            counts[record.fold] += 1
    for fold, count in counts.items():
        if not count:
            raise ValueError(
                f"No eligible, extracted 3D B-mode recordings in fold {fold}; check the dataset and splits.csv"
            )
    return selected


def relative_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare rotation differences within a batch, invariant to a shared left gauge.

    Takes [B,K,3,3] rotations and returns the mean angle in radians over all pairs
    of samples and over the K factors.
    """
    count = len(predicted)
    others = ~torch.eye(count, dtype=torch.bool, device=predicted.device)
    losses = []
    for factor in range(predicted.shape[1]):
        gauge = predicted[:, factor] @ target[:, factor].mT
        pairs = (
            gauge.unsqueeze(1).expand(count, count, 3, 3)
            @ gauge.unsqueeze(0).expand(count, count, 3, 3).mT
        )
        trace = pairs.diagonal(dim1=-2, dim2=-1).sum(-1)
        cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(cosine).masked_fill(~others, 0.0)
        losses.append((angles.sum(dim=1) / (count - 1)).mean())
    return torch.stack(losses).mean()


class Muon(torch.optim.Optimizer):
    """Momentum and Newton–Schulz whitening for dense and convolutional weights."""

    def __init__(self, parameters, lr):
        super().__init__(parameters, {"lr": lr})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(grad)
                momentum = state["momentum"]
                momentum.mul_(0.95).add_(grad)
                grad = grad.add(momentum, alpha=0.95)
                update = grad.reshape(len(grad), -1).bfloat16()
                update /= update.norm() + 1e-7
                transpose = grad.shape[0] > grad[0].numel()
                if transpose:
                    update = update.T
                for _ in range(5):
                    product = update @ update.T
                    polynomial = -4.7750 * product + 2.0315 * product @ product
                    update = 3.4445 * update + polynomial @ update
                if transpose:
                    update = update.T
                parameter.add_(update.reshape(grad.shape), alpha=-group["lr"])


def initial_members(seed, pretrained, device):
    """Build one member per development fold, each from its own seed."""
    members = []
    for index in range(len(DEVELOPMENT_FOLDS)):
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.manual_seed(seed + 1009 * index)
            members.append(PoseRegressor(pretrained=pretrained).to(device))
    return members


def learning_rate(epoch, epochs):
    """Cosine decay over the epochs, counted from one."""
    progress = (epoch - 1) / max(1, epochs - 1)
    floor = FINAL_LEARNING_RATE / LEARNING_RATE
    decay = floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * decay


@torch.inference_mode()
def calibrate(models, batches, device):
    """Fit one left rotation per member and factor that maps predictions to targets."""
    targets = []
    predictions = [[] for _ in models]
    for images, target, _ in batches:
        targets.append(target)
        for model, parts in zip(models, predictions, strict=True):
            parts.append(model(images.to(device)).cpu())
    target = torch.cat(targets).to(device)
    alignments = []
    for parts in predictions:
        prediction = torch.cat(parts).to(device)
        for factor in range(len(TARGETS)):
            moment = torch.einsum(
                "bij,bkj->ik", target[:, factor], prediction[:, factor]
            )
            u, _, vh = torch.linalg.svd(moment)
            if torch.det(u @ vh) < 0:
                u[:, -1] *= -1
            alignments.append(u @ vh)
    return torch.stack(alignments).reshape(len(models), len(TARGETS), 3, 3).cpu()


def train(
    root,
    output: Path,
    *,
    device,
    epochs,
    samples,
    limit,
    workers,
    seed,
    pretrained,
    cache: Path,
):
    items = [
        TrainingRecording(record, record.fold, volume_geometry(record))
        for record in select_records(root, DEVELOPMENT_FOLDS, limit)
    ]
    prepare_cache(items, cache, workers)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    models = initial_members(seed, pretrained, device)
    optimizers = []
    for model in models:
        weight, bias = [], []
        for parameter in model.parameters():
            (weight if parameter.ndim >= 2 else bias).append(parameter)
        optimizers.append(
            (
                Muon(weight, lr=LEARNING_RATE),
                torch.optim.AdamW(bias, lr=LEARNING_RATE, weight_decay=0),
            )
        )

    torch.manual_seed(seed)
    options = {}
    if workers:
        torch.multiprocessing.set_sharing_strategy("file_system")
        options = {
            "persistent_workers": True,
            "prefetch_factor": 2,
            "multiprocessing_context": "fork",
        }
    views = SampledRecordings(items, cache, samples // 2, augmentation=True)
    loader = DataLoader(
        Subset(views, oversample_deep(items)),
        batch_size=None,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.device(device).type == "cuda",
        **options,
    )

    for epoch in range(1, epochs + 1):
        for pair in optimizers:
            for optimizer in pair:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(epoch, epochs)
        totals = {split: [] for split in ("train", "validation")}
        for images, target, fold in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device)
            for index, (model, pair) in enumerate(zip(models, optimizers, strict=True)):
                training = index != fold
                model.train(training)
                with torch.set_grad_enabled(training):
                    loss = relative_loss(model(images), target)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite rotation loss in epoch {epoch}, fold {fold}, member {index}"
                    )
                if training:
                    for optimizer in pair:
                        optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for optimizer in pair:
                        optimizer.step()
                totals["train" if training else "validation"].append(loss.item())
        # The draw a validation data loader would make; it changes the next shuffle.
        torch.empty((), dtype=torch.int64).random_()
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    **{
                        key: sum(values) / len(values)
                        for key, values in totals.items()
                        if values
                    },
                }
            ),
            flush=True,
        )

    for model in models:
        model.eval()
    torch.manual_seed(seed + epochs)
    order = list(range(len(items)))
    random.Random(seed + epochs).shuffle(order)
    views = SampledRecordings(items, cache, ALIGNMENT_VIEWS, augmentation=False)
    adapters = calibrate(
        models, (views[index] for index in order[:ALIGNMENT_RECORDINGS]), device
    )
    payload = {
        "format": BUNDLE_FORMAT,
        "version": 2,
        "input_size": INPUT_SIZE,
        "rotation_target_names": list(TARGETS),
        "model_config": MODEL_CONFIG,
        "model_state_dicts": [
            {key: value.cpu() for key, value in model.state_dict().items()}
            for model in models
        ],
        "gauge_alignment": {"alignments": adapters},
        "training_config": {
            "seed": seed,
            "epochs": epochs,
            "samples": samples,
            "pretrained": pretrained,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        checkpoint = Path(directory) / output.name
        torch.save(payload, checkpoint)
        checkpoint.replace(output)


@dataclass(frozen=True)
class ReslicedRecordings(Dataset):
    """Random virtual views of each recording for evaluation."""

    records: list
    samples: int

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        volumes, _, geometry = load_volume(
            record, indices=lambda n: torch.randint(n, (self.samples,)).tolist()
        )

        xyz = (torch.rand(self.samples, 3) * 2 - 1) * torch.deg2rad(
            torch.tensor((10.0, 10.0, 180.0))
        )
        direction = torch.rand(self.samples) * (2 * math.pi)
        magnitude = torch.rand(self.samples) * math.radians(10)
        orbital = torch.stack(
            (direction.cos(), direction.sin(), torch.zeros_like(direction)), dim=-1
        )
        target = torch.stack(
            (axis_angle_rotation(orbital * magnitude[:, None]), probe_rotation(xyz)),
            dim=1,
        )

        images = reslice(
            volumes,
            geometry,
            target,
            size=INPUT_SIZE,
            width=min(geometry.azimuth_width, math.pi / 3),
            depth=min(geometry.depth_end, 0.16),
        )
        return images, target, record.fold


def batches(records, samples, workers, *, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ReslicedRecordings(records, samples),
        batch_size=None,
        num_workers=workers,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(model: Ensemble, root, *, samples, limit, workers, seed):
    records = select_records(root, (TEST_FOLD,), limit)
    torch.manual_seed(seed)
    totals = torch.zeros(4, device=model.adapters.device)
    for images, target, _ in batches(records, samples, workers, seed=seed):
        prediction = model(images)
        target = target.to(prediction.rotations)
        error = angle(prediction.rotations.mT @ target).mean(dim=0)
        totals += torch.cat(
            (
                error,
                relative_loss(prediction.rotations, target).reshape(1),
                prediction.uncertainty_rad.mean().reshape(1),
            )
        )
    values = torch.rad2deg(totals / len(records)).tolist()
    return dict(
        zip(
            (
                "world_local_mae_deg",
                "probe_mae_deg",
                "relative_mae_deg",
                "uncertainty_deg",
            ),
            values,
            strict=True,
        )
    ) | {
        "fold": TEST_FOLD,
        "recordings": len(records),
        "samples": len(records) * samples,
    }
