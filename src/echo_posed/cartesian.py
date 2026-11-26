"""Cartesian voxel copies of the training volumes and virtual probe sampling in them.

A quarter of the training batches are sliced from these copies.
"""

import math
from pathlib import Path

import numpy as np
import torch
from echoxflow import open_recording
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .model import INPUT_SIZE
from .pose import render_pose
from .stitching import (
    beat_gating,
    display_stitch_plan,
    load_stitched_3d_brightness_frames,
)


SPACING_MM = 0.78
VOXEL_MM = float(np.float32(SPACING_MM))
CACHED_FRAMES = 8
SECTOR_WIDTH = math.radians(60.0)


def cache_path(cache: Path, record) -> Path:
    return Path(cache) / f"{record.exam_id}__{record.recording_id}.npz"


def voxelize(volume: torch.Tensor, geometry) -> np.ndarray:
    """Resample [T,A,E,R] beamspace frames to uint8 [X,Y,Z,T] voxels."""
    spacing = SPACING_MM * 1.0e-3
    half_x = geometry.depth_end * math.sin(0.5 * geometry.azimuth_width)
    half_y = geometry.depth_end * math.sin(0.5 * geometry.elevation_width)
    nx = max(2, int(math.floor(2.0 * half_x / spacing)) + 1)
    ny = max(2, int(math.floor(2.0 * half_y / spacing)) + 1)
    nz = max(2, int(math.floor(geometry.depth_end / spacing)) + 1)
    x = (torch.arange(nx, dtype=torch.float32) - 0.5 * (nx - 1)) * spacing
    y = (torch.arange(ny, dtype=torch.float32) - 0.5 * (ny - 1)) * spacing
    z = torch.arange(nz, dtype=torch.float32) * spacing
    # Keep the coordinates interleaved: atan2 rounds differently on contiguous tensors.
    points = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
    x, y, z = points.reshape(-1, 3).unbind(dim=-1)
    distance = torch.sqrt(x**2 + y**2 + z**2)
    azimuth = torch.atan2(x, z)
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2))
    grid = torch.stack(
        (
            2 * (azimuth + geometry.azimuth_width / 2) / geometry.azimuth_width - 1,
            2 * (elevation + geometry.elevation_width / 2) / geometry.elevation_width
            - 1,
            2
            * (distance - geometry.depth_start)
            / (geometry.depth_end - geometry.depth_start)
            - 1,
        ),
        dim=-1,
    ).reshape(1, nx, ny, nz, 3)
    frames = [
        F.grid_sample(frame.permute(2, 1, 0)[None, None], grid, align_corners=True)[
            0, 0
        ]
        for frame in volume
    ]
    stacked = torch.stack(frames, dim=-1)
    return (stacked.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()


def write_cache(item, path: Path) -> None:
    """Voxelize evenly spaced frames of one stitched recording."""
    with open_recording(item.record) as store:
        group = store.group
        metadata, beats = beat_gating(group)
        frames = np.asarray(group["data/3d_brightness_mode"][()])
        if beats > 1:
            timestamps = np.asarray(group["timestamps/3d_brightness_mode"][()])
            plan, order = display_stitch_plan(frames, timestamps, metadata, beats)
            total = len(plan.stitched_indices_matrix)
        else:
            total = len(frames)
        positions = np.linspace(0, total - 1, min(total, CACHED_FRAMES))
        rows = sorted({int(round(position)) for position in positions})
        if beats > 1:
            frames = load_stitched_3d_brightness_frames(frames, plan, rows, order)
        else:
            frames = frames[rows]
    volume = torch.as_tensor(frames).permute(0, 2, 1, 3).float() / 255.0
    partial = path.with_name(f"{path.stem}.partial.npz")
    np.savez_compressed(partial, volume=voxelize(volume, item.geometry))
    partial.replace(path)


class _MissingCopies(Dataset):
    def __init__(self, items, cache):
        self.missing = [
            (item, path)
            for item in items
            if not (path := cache_path(cache, item.record)).is_file()
        ]

    def __len__(self):
        return len(self.missing)

    def __getitem__(self, index):
        write_cache(*self.missing[index])
        return index


def prepare_cache(items, cache: Path, workers: int) -> None:
    """Write the Cartesian copies that do not exist yet."""
    Path(cache).mkdir(parents=True, exist_ok=True)
    missing = _MissingCopies(items, cache)
    if not missing:
        return
    print(f"Writing {len(missing)} Cartesian volumes to {cache}", flush=True)
    for _ in DataLoader(missing, batch_size=None, num_workers=workers):
        pass


def cached_volume(cache: Path, record) -> torch.Tensor:
    """Load [T,X,Y,Z] voxels with the median background removed."""
    with np.load(cache_path(cache, record), allow_pickle=False) as data:
        voxels = data["volume"].astype(np.float32)
    volume = torch.from_numpy(voxels).permute(3, 0, 1, 2).contiguous()
    return (volume - volume.median()).clamp_min(0.0)


def slice_cartesian(
    volume: torch.Tensor, target: torch.Tensor, frames: torch.Tensor
) -> torch.Tensor:
    """Render [B,1,224,224] images at [B,2,3,3] poses from [T,X,Y,Z] voxels."""
    extent = tuple((size - 1) * VOXEL_MM for size in volume.shape[1:])
    depth = extent[2]
    rotation, translation = render_pose(target[:, 0], target[:, 1], depth)
    azimuth = torch.linspace(-0.5 * SECTOR_WIDTH, 0.5 * SECTOR_WIDTH, INPUT_SIZE)
    radius = torch.linspace(0.0, depth, INPUT_SIZE)
    azimuth, radius = torch.meshgrid(azimuth, radius, indexing="ij")
    points = torch.stack(
        (
            radius * torch.sin(azimuth),
            torch.zeros_like(radius),
            radius * torch.cos(azimuth),
        ),
        dim=-1,
    ).reshape(1, -1, 3)
    source = torch.matmul(points.expand(len(target), -1, -1) - translation, rotation)
    origin = torch.zeros((1, 1, 3))
    origin[..., 2] = -0.5 * extent[2]
    half = torch.as_tensor(extent, dtype=source.dtype) * 0.5
    grid = ((source + origin) / half).flip(-1)
    grid = grid.reshape(len(target), INPUT_SIZE, INPUT_SIZE, 3)
    sectors = torch.empty(len(target), INPUT_SIZE, INPUT_SIZE)
    for frame in torch.unique(frames).tolist():
        rows = torch.where(frames == frame)[0]
        sectors[rows] = F.grid_sample(
            volume[frame][None, None].expand(len(rows), -1, -1, -1, -1),
            grid[rows].unsqueeze(1),
            align_corners=True,
        )[:, 0, 0]
    return _scan_convert(sectors, depth)


def _scan_convert(sectors: torch.Tensor, depth: float) -> torch.Tensor:
    """Convert [B,angle,depth] sectors and normalize each image to [0, 1]."""
    half = 0.5 * SECTOR_WIDTH
    x = torch.linspace(-depth * math.sin(half), depth * math.sin(half), INPUT_SIZE)
    z = torch.linspace(0.0, depth, INPUT_SIZE)
    z, x = torch.meshgrid(z, x, indexing="ij")
    angle = torch.atan2(x, z)
    radius = torch.sqrt(x.square() + z.square())
    valid = (angle >= -half) & (angle <= half) & (radius >= 0.0) & (radius <= depth)
    grid = torch.stack(
        (2.0 * (angle + half) / SECTOR_WIDTH - 1.0, 2.0 * radius / depth - 1.0),
        dim=-1,
    )
    images = F.grid_sample(
        sectors.transpose(-2, -1).unsqueeze(1),
        grid.expand(len(sectors), -1, -1, -1),
        align_corners=True,
    )
    images = torch.where(valid, images, torch.zeros(()))
    peak = images.amax(dim=(-2, -1), keepdim=True)
    images = images / torch.where(peak > 1.5, 255.0, 1.0)
    lo = images.amin(dim=(-2, -1), keepdim=True)
    hi = images.amax(dim=(-2, -1), keepdim=True)
    return (images - lo) / (hi - lo).clamp_min(1.0e-6)
