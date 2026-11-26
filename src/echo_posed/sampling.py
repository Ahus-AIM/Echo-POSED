"""Training batches from virtual probe movements in 3D EchoXFlow recordings."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from echoxflow import open_recording
from echoxflow.croissant import RecordingRecord
from torch.nn import functional as F
from torch.utils.data import Dataset

from .cartesian import cached_volume, slice_cartesian
from .imaging import VolumeGeometry, normalize
from .model import INPUT_SIZE
from .pose import render_pose
from .stitching import (
    beat_gating,
    load_3d_brightness_frames,
    load_stitched_3d_brightness_frames,
    stitch_plan,
)


FRAMES_PER_RECORDING = 16
STITCH_SCORE_FRAMES = 8
PROBE_RANGE_DEG = (10.0, 10.0, 180.0)
TILT_RANGE_DEG = 10.0
PAIRED_FRACTION = 0.5
CARTESIAN_FRACTION = 0.25
DEEP_SECTOR_M = 0.16
DEEP_SECTOR_REPEATS = 3


@dataclass(frozen=True)
class TrainingRecording:
    record: RecordingRecord
    fold: int
    geometry: VolumeGeometry


def oversample_deep(recordings: Sequence[TrainingRecording]) -> list[int]:
    """Recording indices for one epoch, with deep sectors repeated."""
    indices = []
    for index, item in enumerate(recordings):
        depth = item.geometry.depth_end - item.geometry.depth_start
        indices.extend([index] * (DEEP_SECTOR_REPEATS if depth >= DEEP_SECTOR_M else 1))
    return indices


# Poses


def rotation_from_vector(vector: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from rotation vectors, through unit quaternions."""
    angles = torch.linalg.norm(vector, dim=-1, keepdim=True)
    small = angles.abs() < 1e-6
    safe = torch.where(small, torch.ones_like(angles), angles)
    scale = torch.where(
        small, 0.5 - angles * angles / 48.0, torch.sin(0.5 * angles) / safe
    )
    quaternion = torch.cat((torch.cos(0.5 * angles), vector * scale), dim=-1)
    w, x, y, z = F.normalize(quaternion, dim=-1, eps=1e-8).unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rows = (
        (ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)),
        (2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)),
        (2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def share_within_pairs(values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Give every member of a paired group the value of its first member."""
    shared = values.clone()
    for group in torch.unique(groups):
        if group >= 0:
            members = torch.nonzero(groups == group).flatten()
            shared[members] = shared[members[0]].clone()
    return shared


def sample_poses(count: int):
    """Draw [B,2,3,3] world-local/probe rotations and their paired group labels.

    Paired groups of five share a probe rotation and frame while the world-local
    tilt is zero, then positive and negative about x and about y.
    """
    limit = torch.tensor(
        [[math.radians(-value), math.radians(value)] for value in PROBE_RANGE_DEG]
    )
    xyz = limit[:, 0] + torch.rand((count, 3)) * (limit[:, 1] - limit[:, 0])
    vectors = torch.zeros(3, count, 3)
    for axis in range(3):
        vectors[axis, :, axis] = xyz[:, axis]
    x, y, z = rotation_from_vector(vectors)
    probe = y @ x @ z

    maximum = torch.tensor(math.radians(TILT_RANGE_DEG))
    direction = torch.rand(count) * (2.0 * torch.pi)
    magnitude = torch.rand(count) * maximum
    tilt = torch.stack(
        (magnitude * torch.cos(direction), magnitude * torch.sin(direction)), dim=1
    )
    groups = torch.full((count,), -1)
    paired = min(int(round(count * PAIRED_FRACTION / 5)), count // 5)
    for group in range(paired):
        rows = slice(5 * group, 5 * group + 5)
        tilt[rows] = torch.tensor(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))) * maximum
        groups[rows] = group
    probe = share_within_pairs(probe, groups)
    world_local = rotation_from_vector(
        torch.cat((tilt, torch.zeros_like(tilt[:, :1])), dim=1)
    )
    return torch.stack((world_local, probe), dim=1), groups


# Beamspace rendering. Volumes are [T,A,E,R] and sectors are [B,1,angle,depth].


def equalize_angular_resolution(
    volume: torch.Tensor, geometry: VolumeGeometry
) -> torch.Tensor:
    """Randomly reduce the finer angular axis of individual frames."""
    frames, first, second, radial = volume.shape
    first_spacing = geometry.azimuth_width / (first - 1)
    second_spacing = geometry.elevation_width / (second - 1)
    applied = torch.rand(frames) < 0.75
    fraction = torch.linspace(0.25, 1.0, 4)[torch.randint(4, (frames,))]
    fraction = torch.where(applied, fraction, torch.zeros_like(fraction))
    if first_spacing == second_spacing:
        return volume
    reduce_second = second_spacing < first_spacing
    fine, coarse = sorted((first_spacing, second_spacing))
    spacing = fine * torch.pow(
        torch.full_like(fraction, max(coarse / fine, 1.0)), fraction
    )
    width = geometry.elevation_width if reduce_second else geometry.azimuth_width
    full = second if reduce_second else first
    counts = (torch.round(width / spacing).long() + 1).clamp(2, full)
    applied &= counts < full
    if not applied.any():
        return volume
    output = volume.float() / 255.0
    for count in torch.unique(counts[applied]).tolist():
        rows = torch.nonzero(applied & (counts == count)).flatten()
        size = (first, count) if reduce_second else (count, second)
        reduced = F.adaptive_avg_pool3d(output[rows].unsqueeze(1), (*size, radial))
        output[rows] = F.interpolate(
            reduced,
            size=(first, second, radial),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
    return output


def reslice_frames(
    volume: torch.Tensor,
    geometry: VolumeGeometry,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Sample one virtual sector per pose from the frame named by ``slots``."""
    width, depth = geometry.azimuth_width, geometry.depth_end
    azimuth = torch.linspace(-width / 2, width / 2, INPUT_SIZE)
    radius = torch.linspace(0, depth, INPUT_SIZE)
    azimuth, radius = torch.meshgrid(azimuth, radius, indexing="ij")
    points = torch.stack(
        (
            radius * torch.sin(azimuth),
            torch.zeros_like(radius),
            radius * torch.cos(azimuth),
        ),
        dim=-1,
    ).reshape(-1, 3)
    source = torch.matmul(points.unsqueeze(0) - translation, rotation)
    x, y, z = source.unbind(dim=-1)
    distance = torch.sqrt(x**2 + y**2 + z**2)
    az = torch.atan2(x, z)
    el = torch.atan2(y, torch.sqrt(x**2 + z**2))
    grid = torch.stack(
        (
            2 * (az + width / 2) / width - 1,
            2 * (el + geometry.elevation_width / 2) / geometry.elevation_width - 1,
            2
            * (distance - geometry.depth_start)
            / (geometry.depth_end - geometry.depth_start)
            - 1,
        ),
        dim=-1,
    ).reshape(len(slots), INPUT_SIZE, INPUT_SIZE, 3)
    sectors = volume.new_empty((len(slots), 1, INPUT_SIZE, INPUT_SIZE))
    for slot in torch.unique(slots).tolist():
        rows = torch.where(slots == slot)[0]
        sampled = F.grid_sample(
            volume[slot].permute(2, 1, 0)[None, None],
            grid[rows].unsqueeze(0),
            align_corners=True,
        )
        sectors[rows, 0] = sampled[0, 0]
    return sectors


def equalize_lateral_resolution(
    sectors: torch.Tensor,
    rotation: torch.Tensor,
    geometry: VolumeGeometry,
    azimuth_count: int,
    elevation_count: int,
) -> torch.Tensor:
    """Blur each sector so that its lateral resolution does not reveal its roll."""
    count = len(sectors)
    spacing_az = geometry.azimuth_width / (azimuth_count - 1)
    spacing_el = geometry.elevation_width / (elevation_count - 1)
    lateral = rotation[:, 0, :]
    norm = torch.linalg.vector_norm(lateral[:, :2], dim=1).clamp_min(1.0e-6)
    along_az = lateral[:, 0].abs() / norm
    along_el = lateral[:, 1].abs() / norm
    source = (along_az * spacing_az).square() + (along_el * spacing_el).square()
    target = torch.full_like(source, max(spacing_az, spacing_el))
    target = target * (1.0 + 0.25 * torch.rand(count))
    sigma = 0.5 * torch.sqrt((target.square() - source).clamp_min(0.0))
    sigma = sigma / (geometry.azimuth_width / (INPUT_SIZE - 1))
    applied = torch.rand(count) < 1.0
    sigma = torch.where(applied, sigma, torch.zeros_like(sigma))
    output = sectors[:, 0].clone()
    quantized = (sigma * 10.0).round().clamp(max=300.0) / 10.0
    for value in torch.unique(quantized).tolist():
        if value <= 0.2:
            continue
        rows = quantized == value
        reach = max(1, math.ceil(3.0 * value))
        offsets = torch.arange(-reach, reach + 1, dtype=output.dtype)
        kernel = torch.exp(-0.5 * (offsets / value).square())
        kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
        lines = output[rows].permute(0, 2, 1).reshape(-1, 1, INPUT_SIZE)
        blurred = F.conv1d(F.pad(lines, (reach, reach), mode="replicate"), kernel)
        output[rows] = blurred.reshape(int(rows.sum()), -1, INPUT_SIZE).permute(0, 2, 1)
    return output.unsqueeze(1)


def remap_intensity(sectors: torch.Tensor) -> torch.Tensor:
    """Random piecewise-linear mapping between per-sector intensity percentiles."""
    count = len(sectors)
    low, high, low_value, high_value = (
        start + torch.rand(count) * (stop - start)
        for start, stop in ((0.0, 30.0), (95.0, 99.99), (0.01, 0.2), (0.9, 0.99))
    )
    output = sectors[:, 0].clone()
    quantiles = torch.linspace(0.0, 1.0, 256)
    for index in range(count):
        mask = sectors[index, 0] != 0
        values = sectors[index, 0][mask]
        if values.numel() < 16:
            continue
        table = torch.quantile(values, quantiles)

        def percentile(percent):
            position = torch.clamp(percent / 100.0, 0.0, 1.0) * 255.0
            below = int(torch.floor(position).item())
            weight = position - float(below)
            return table[below] * (1.0 - weight) + table[min(below + 1, 255)] * weight

        smallest, largest = table[0], table[-1]
        start = torch.clamp(percentile(low[index]), min=smallest + 1e-6)
        stop = torch.clamp(percentile(high[index]), max=largest - 1e-6)
        if stop <= start:
            stop = start + 1e-6
        mapped = torch.empty_like(values)
        below = values <= start
        between = (values > start) & (values < stop)
        above = values >= stop
        first = torch.clamp(start - smallest, min=1e-6)
        middle = torch.clamp(stop - start, min=1e-6)
        last = torch.clamp(largest - stop, min=1e-6)
        mapped[below] = low_value[index] * (values[below] - smallest) / first
        mapped[between] = (
            low_value[index]
            + (high_value[index] - low_value[index])
            * (values[between] - start)
            / middle
        )
        mapped[above] = (
            high_value[index]
            + (1.0 - high_value[index]) * (values[above] - stop) / last
        )
        output[index][mask] = mapped.clamp(0.0, 1.0)
        output[index][~mask] = 0.0
    return output.unsqueeze(1)


def visible_widths(rotation: torch.Tensor, geometry: VolumeGeometry) -> torch.Tensor:
    """Contiguous angle of each sector plane that lies inside the acquired cone."""
    limit = math.pi / 2.0 - 1.0e-4
    theta = torch.linspace(-limit, limit, 257)
    rays = torch.stack(
        (torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)), dim=-1
    )
    x, y, z = torch.einsum("si,bij->bsj", rays, rotation).unbind(dim=-1)
    azimuth = torch.atan2(x, z)
    elevation = torch.atan2(y, torch.sqrt(x.square() + z.square()))
    valid = (
        (azimuth.abs() <= 0.5 * geometry.azimuth_width)
        & (elevation.abs() <= 0.5 * geometry.elevation_width)
        & (z > 0.0)
    ).tolist()
    step = float(theta[1].item() - theta[0].item())
    widths = torch.zeros(len(rotation))
    for index, inside in enumerate(valid):
        left = right = 128
        if not inside[left]:
            continue
        while left > 0 and inside[left - 1]:
            left -= 1
        while right < 256 and inside[right + 1]:
            right += 1
        widths[index] = min((right - left + 1) * step, math.pi)
    return widths


def reduce_sector(
    sectors: torch.Tensor, rotation: torch.Tensor, geometry: VolumeGeometry
) -> torch.Tensor:
    """Randomly narrow each sector with a soft edge, keeping the canvas size."""
    count = len(sectors)
    visible = visible_widths(rotation, geometry)
    applied = torch.rand(count) < 1.0
    capped = applied & (torch.rand(count) < 0.25)
    upper = torch.where(
        capped,
        torch.minimum(visible, torch.full_like(visible, geometry.azimuth_width)),
        visible,
    )
    lower = torch.minimum(upper, torch.full_like(upper, 0.75))
    kept = lower + torch.rand_like(upper) * (upper - lower)
    kept = torch.where(applied, kept, visible)
    half = torch.where(
        visible > 1.0e-6,
        (kept / visible).clamp(0.0, 1.0),
        torch.ones_like(visible),
    ).view(-1, 1, 1, 1)
    coordinate = torch.linspace(-1.0, 1.0, INPUT_SIZE).abs().view(1, 1, -1, 1)
    return sectors * ((half + 0.5 * 0.02 - coordinate) / 0.02).clamp(0.0, 1.0)


def scan_convert_sectors(
    sectors: torch.Tensor, width: float, depth: float
) -> torch.Tensor:
    """Convert [B,1,angle,depth] sectors to [B,1,224,224] images."""
    start, stop = -width / 2, width / 2

    def extent(low, high):
        low_t, high_t = torch.tensor(low), torch.tensor(high)
        near, far = torch.tensor(0.0), torch.tensor(depth)
        corner = torch.max(
            torch.stack(
                (
                    torch.cos(low_t) * near,
                    torch.cos(high_t) * near,
                    torch.cos(low_t) * far,
                    torch.cos(high_t) * far,
                )
            )
        )
        crosses = torch.sin(low_t) < 0 and torch.sin(high_t) > 0
        return float(far if crosses else corner)

    quarter, half = math.pi / 2, math.pi
    left = -extent(start + quarter, stop + quarter)
    right = extent(start - quarter, stop - quarter)
    top = -extent(start + half, stop + half)
    bottom = extent(start, stop)
    columns = int(round(INPUT_SIZE * abs((right - left) / (bottom - top))))
    x = torch.linspace(left, right, columns)
    z = torch.linspace(top, bottom, INPUT_SIZE)
    z, x = torch.meshgrid(z, x, indexing="ij")
    angle = torch.atan2(x, z)
    radius = torch.sqrt(x**2 + z**2)
    valid = (angle >= start) & (angle <= stop) & (radius >= 0.0) & (radius <= depth)
    last = INPUT_SIZE - 1
    grid = torch.stack(
        (
            2.0 * ((angle - start) / (stop - start) * last / last) - 1.0,
            2.0 * (radius / depth * last / last) - 1.0,
        ),
        dim=-1,
    )
    images = F.grid_sample(
        sectors.transpose(-2, -1),
        grid.expand(len(sectors), -1, -1, -1),
        align_corners=True,
    )
    images = torch.where(valid, images, torch.zeros(()))
    if columns > INPUT_SIZE:
        offset = (columns - INPUT_SIZE) // 2
        return images[..., offset : offset + INPUT_SIZE]
    offset = (INPUT_SIZE - columns) // 2
    return F.pad(images, (offset, INPUT_SIZE - columns - offset))


# Batch augmentation


def augment_batch(images: torch.Tensor, target: torch.Tensor):
    """Append mirrored views, then vary intensity, sharpness and the sector top."""
    mirror = torch.diag(torch.tensor((-1.0, -1.0, 1.0)))
    mirrored = torch.stack(
        (mirror @ target[:, 0] @ mirror, target[:, 1] @ mirror), dim=1
    )
    images = torch.cat((images, images.flip(-1)))
    target = torch.cat((target, mirrored))
    images = (images - images.min()) / (images.max() - images.min())
    count = len(images)
    gamma = torch.empty(count).uniform_(0.9, 1.1)
    images = torch.pow(images, gamma.view(-1, 1, 1, 1))

    blurred = torch.rand(count) < 0.1
    if blurred.any():
        sigma = torch.empty(count).uniform_(0.3, 1.0)
        offsets = torch.arange(-3.0, 4.0)
        kernels = torch.exp(-0.5 * (offsets.unsqueeze(0) / sigma.unsqueeze(1)).square())
        kernels = kernels / kernels.sum(dim=1, keepdim=True)
        grouped = images.reshape(1, count, INPUT_SIZE, INPUT_SIZE)
        smooth = F.conv2d(
            F.pad(grouped, (3, 3, 0, 0), mode="reflect"),
            kernels.reshape(count, 1, 1, 7),
            groups=count,
        )
        smooth = F.conv2d(
            F.pad(smooth, (0, 0, 3, 3), mode="reflect"),
            kernels.reshape(count, 1, 7, 1),
            groups=count,
        ).reshape_as(images)
        images = torch.where(blurred.view(-1, 1, 1, 1), smooth, images)

    # Cut a triangle from each of the two top corners.
    last = float(INPUT_SIZE - 1)
    top = torch.empty(count).uniform_(0.0, last).view(-1, 1, 1)
    side = torch.empty(count).uniform_(1.0, last * 0.5).view(-1, 1, 1)
    row = torch.arange(INPUT_SIZE, dtype=torch.float32).view(1, -1, 1)
    column = torch.arange(INPUT_SIZE, dtype=torch.float32).view(1, 1, -1)
    along = (row / side).clamp(0.0, 1.0)
    corner = (row <= side) & (
        (column <= top * (1.0 - along)) | (column >= top + (last - top) * along)
    )
    return images.masked_fill(corner.unsqueeze(1), 0.0), target


def frame_rows(total: int, count: int, shuffle: bool) -> list[int]:
    """Choose ``count`` frames, repeating the recording when it is shorter."""
    if not shuffle:
        return [index % total for index in range(count)]
    rows = []
    while len(rows) < count:
        rows.extend(torch.randperm(total)[: count - len(rows)].tolist())
    return rows


class SampledRecordings(Dataset):
    """One batch of virtual views per recording.

    With ``augmentation`` the views are drawn for training and returned with their
    mirrored copies. Without it they are drawn for fitting the alignment matrices:
    from the first frame, normalized per view.
    """

    def __init__(
        self,
        recordings: Sequence[TrainingRecording],
        cache,
        views: int,
        *,
        augmentation: bool,
    ):
        self.recordings = list(recordings)
        self.cache = cache
        self.views = views
        self.augmentation = augmentation
        self._plans = {}

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int):
        item = self.recordings[index]
        if torch.rand(()) < CARTESIAN_FRACTION:
            images, target = self._cartesian_views(item)
        else:
            images, target = self._beamspace_views(item)
        if self.augmentation:
            images, target = augment_batch(images, target)
        else:
            images = normalize(images)
        return images, target, item.fold

    def _cartesian_views(self, item):
        volume = cached_volume(self.cache, item.record)
        target, groups = sample_poses(self.views)
        count = self.views if self.augmentation else 1
        frames = torch.randint(0, len(volume), (count,)).expand(self.views)
        frames = share_within_pairs(frames, groups)
        return slice_cartesian(volume, target, frames), target

    def _load_frames(self, item) -> torch.Tensor:
        count = min(FRAMES_PER_RECORDING, self.views) if self.augmentation else 1
        with open_recording(item.record) as store:
            group = store.group
            metadata, beats = beat_gating(group)
            volumes = group["data/3d_brightness_mode"]
            if beats == 1:
                rows = frame_rows(volumes.shape[0], count, self.augmentation)
                frames = load_3d_brightness_frames(volumes, rows)
            else:
                key = item.record.recording_id
                if key not in self._plans:
                    timestamps = np.asarray(group["timestamps/3d_brightness_mode"][:])
                    self._plans[key] = stitch_plan(
                        volumes,
                        timestamps,
                        metadata,
                        beats,
                        score_frames=STITCH_SCORE_FRAMES,
                    )
                plan, order = self._plans[key]
                rows = frame_rows(
                    len(plan.stitched_indices_matrix), count, self.augmentation
                )
                frames = load_stitched_3d_brightness_frames(volumes, plan, rows, order)
        return torch.as_tensor(frames).permute(0, 2, 1, 3).contiguous()

    def _beamspace_views(self, item):
        geometry = item.geometry
        volume = self._load_frames(item)
        if self.augmentation:
            volume = equalize_angular_resolution(volume, geometry)
        target, groups = sample_poses(self.views)
        rotation, translation = render_pose(
            target[:, 0], target[:, 1], geometry.depth_end
        )
        slots = share_within_pairs(torch.arange(self.views), groups) % len(volume)
        if volume.dtype == torch.uint8:
            volume = volume.float() / 255.0
        sectors = reslice_frames(volume, geometry, rotation, translation, slots)
        sectors = equalize_lateral_resolution(
            sectors, rotation, geometry, volume.shape[1], volume.shape[2]
        )
        if self.augmentation:
            sectors = remap_intensity(sectors)
            sectors = reduce_sector(sectors, rotation, geometry)
        images = scan_convert_sectors(
            sectors, geometry.azimuth_width, geometry.depth_end
        )
        return images, target
