"""Scan conversion and virtual probe sampling in native beamspace."""

from dataclasses import dataclass
from math import isfinite, pi, sin

import torch
from torch.nn import functional as F

from .pose import render_pose


@dataclass(frozen=True)
class VolumeGeometry:
    depth_start: float
    depth_end: float
    azimuth_width: float
    elevation_width: float

    def __post_init__(self):
        values = (
            self.depth_start,
            self.depth_end,
            self.azimuth_width,
            self.elevation_width,
        )
        if not all(map(isfinite, values)) or not (
            0 <= self.depth_start < self.depth_end
            and 0 < self.azimuth_width <= pi
            and 0 < self.elevation_width <= pi
        ):
            raise ValueError(f"Invalid spherical beamspace geometry: {self}")


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize each scan-converted frame to [0, 1]."""
    lo = images.amin(dim=(-2, -1), keepdim=True)
    hi = images.amax(dim=(-2, -1), keepdim=True)
    return (images - lo) / (hi - lo).clamp_min(1e-6)


def scan_convert(
    frames: torch.Tensor,
    *,
    width: float,
    depth_start: float,
    depth_end: float,
    size: int,
    depth_cap: float | None = None,
) -> torch.Tensor:
    """Convert [T,R,A] to square [T,1,H,W], preserving physical aspect."""
    used_depth = depth_end if depth_cap is None else min(depth_end, depth_cap)
    if not 0 <= depth_start < used_depth or not 0 < width <= pi or size < 2:
        raise ValueError("Invalid sector geometry, depth cap or output size")
    half_extent = used_depth * sin(width / 2)
    ncols = max(2, round(size * 2 * sin(width / 2)))
    x = torch.linspace(-half_extent, half_extent, ncols, device=frames.device)
    z = torch.linspace(0, used_depth, size, device=frames.device)
    z, x = torch.meshgrid(z, x, indexing="ij")
    az, radius = torch.atan2(x, z), torch.hypot(x, z)
    grid = torch.stack(
        (2 * az / width, 2 * (radius - depth_start) / (depth_end - depth_start) - 1),
        dim=-1,
    )
    sampled = F.grid_sample(
        frames[:, None],
        grid.expand(len(frames), -1, -1, -1),
        align_corners=True,
    )
    valid = (az.abs() <= width / 2) & (radius >= depth_start) & (radius <= used_depth)
    sampled = sampled.masked_fill(~valid, 0)
    if ncols > size:
        left = (ncols - size) // 2
        sampled = sampled[..., left : left + size]
    else:
        left = (size - ncols) // 2
        sampled = F.pad(sampled, (left, size - ncols - left))
    return normalize(sampled)


def slice_volume(
    volumes: torch.Tensor,
    geometry: VolumeGeometry,
    poses: torch.Tensor,
    *,
    size: int,
    width: float,
    depth: float,
) -> torch.Tensor:
    """Sample [B,E,A,R] volumes at [B,2,3,3] world-local/probe poses.

    Returns [B,size,size] beamspace sectors on (depth, azimuth) axes, from the probe
    origin to ``depth`` metres across ``width`` radians.
    """
    if volumes.ndim != 4 or poses.shape != (len(volumes), 2, 3, 3):
        raise ValueError(
            "Expected [B,E,A,R] volumes and one pair of rotations per volume"
        )
    if not 0 < width <= pi or not 0 < depth <= geometry.depth_end or size < 2:
        raise ValueError("Invalid virtual probe width, depth or output size")
    ray_angle = torch.linspace(-width / 2, width / 2, size, device=volumes.device)
    ray_depth = torch.linspace(0, depth, size, device=volumes.device)
    radius, az = torch.meshgrid(ray_depth, ray_angle, indexing="ij")
    points = torch.stack(
        (radius * az.sin(), torch.zeros_like(radius), radius * az.cos()), dim=-1
    )
    rotation, translation = render_pose(poses[:, 0], poses[:, 1], depth)
    source = (points.reshape(-1, 3) - translation) @ rotation
    x, y, z = source.unbind(dim=-1)
    azimuth = torch.atan2(x, z) * (2 / geometry.azimuth_width)
    elevation = torch.atan2(y, torch.hypot(x, z)) * (2 / geometry.elevation_width)
    distance = (source.norm(dim=-1) - geometry.depth_start) * (
        2 / (geometry.depth_end - geometry.depth_start)
    ) - 1
    grid = torch.stack((distance, azimuth, elevation), dim=-1).reshape(
        len(volumes), 1, size, size, 3
    )
    return F.grid_sample(volumes[:, None], grid, align_corners=True)[:, 0, 0]


def reslice(
    volumes: torch.Tensor,
    geometry: VolumeGeometry,
    poses: torch.Tensor,
    *,
    size: int,
    width: float,
    depth: float,
) -> torch.Tensor:
    """Slice [B,E,A,R] volumes and scan-convert the sectors to [B,1,size,size]."""
    sectors = slice_volume(
        volumes, geometry, poses, size=size, width=width, depth=depth
    )
    return scan_convert(sectors, width=width, depth_start=0, depth_end=depth, size=size)
