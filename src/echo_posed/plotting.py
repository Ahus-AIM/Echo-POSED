"""Figures of virtual slices and their predicted rotations."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .imaging import VolumeGeometry, scan_convert, slice_volume


def volume_support(
    geometry: VolumeGeometry,
    poses: torch.Tensor,
    *,
    size: int,
    width: float,
    depth: float,
) -> torch.Tensor:
    """Mark the [T,size,size] sector samples that fall inside the acquired volume."""
    ones = poses.new_ones((len(poses), 2, 2, 2))
    sampled = slice_volume(ones, geometry, poses, size=size, width=width, depth=depth)
    return (sampled > 0.9999).to(sampled)


def transparent_sectors(
    sectors: torch.Tensor,
    support: torch.Tensor,
    *,
    width: float,
    depth: float,
    size: int,
    oversample=4,
) -> torch.Tensor:
    """Scan-convert [T,R,A] sectors to [T,size,size,2] gray and alpha images.

    Alpha is the fraction of each pixel covered by acquired data inside the sector,
    based on geometry alone. Everything else is transparent, while dark tissue and
    blood stay opaque.
    """
    options = {
        "width": width,
        "depth_start": 0,
        "depth_end": depth,
        "size": size * oversample,
    }
    gray = scan_convert(sectors, **options)
    inside = scan_convert(support, **options)
    alpha = F.avg_pool2d(inside, oversample)
    # Average only the covered samples, which avoids a dark rim along the edges.
    gray = F.avg_pool2d(gray * inside, oversample) / alpha.clamp_min(1e-6)
    return torch.stack((gray[:, 0].clamp(0, 1), alpha[:, 0].clamp(0, 1)), dim=-1)


def save_rotation_grid(
    path: str | Path,
    sectors: torch.Tensor,
    rotations: torch.Tensor,
    titles: list[str],
    *,
    geometry: VolumeGeometry,
    poses: torch.Tensor,
    width: float,
    depth: float,
    columns: int = 5,
):
    """Save [T,R,A] sectors beside their [T,3,3] rotations on a transparent background."""
    if not len(sectors) == len(rotations) == len(titles):
        raise ValueError("Expected one rotation and one title per sector")
    support = volume_support(
        geometry, poses.cpu(), size=sectors.shape[-1], width=width, depth=depth
    )
    images = transparent_sectors(
        sectors.cpu(), support, width=width, depth=depth, size=448
    )
    rows = (len(images) + columns - 1) // columns
    gray = "#7d8590"  # readable on light and dark backgrounds
    colorbar_style = ("text.color", "axes.labelcolor", "ytick.color", "ytick.labelcolor")
    with plt.rc_context(dict.fromkeys(colorbar_style, gray)):
        figure = plt.figure(figsize=(16, 1.73 * rows))
        cells = figure.add_gridspec(
            rows,
            columns,
            left=0.005,
            right=0.93,
            bottom=0.01,
            top=0.94,
            wspace=0.08,
            hspace=0.2,
        )
        for cell, image, rotation, title in zip(cells, images, rotations.cpu(), titles):
            left, right = cell.subgridspec(1, 2, wspace=0).subplots()
            left.imshow(image[..., [0, 0, 0, 1]].numpy())
            left.set_title(title, loc="left")
            colors = right.imshow(rotation.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
            left.axis("off")
            right.axis("off")
        bar = figure.add_axes((0.945, 0.2, 0.008, 0.6))
        figure.colorbar(colors, cax=bar, label="Predicted rotation relative to 0°")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, transparent=True)
        plt.close(figure)
