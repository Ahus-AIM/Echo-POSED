"""Read stock EchoXFlow recordings and patient folds."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from echoxflow import load_croissant, open_recording
from echoxflow.config import data_root
from echoxflow.croissant import RecordingRecord
from echoxflow.scan.beat_stitching import prepare_3d_brightness_for_display
from echoxflow.streams import BrightnessModeStream

from .imaging import VolumeGeometry, scan_convert
from .model import INPUT_SIZE


VOLUME_STREAM = "3d_brightness_mode"
CINE_STREAM = "2d_brightness_mode"
EXCLUDED_RECORDINGS = frozenset(
    (
        "recording_8895061950720fd2",
        "recording_992181021a2133b3",
        "recording_d0c7ab72a730fad7",
        "recording_e23060e7e44947b6",
    )
)


def recordings(
    root: Path | None, *, content: str, fold: int | None = None
) -> list[RecordingRecord]:
    catalog = load_croissant(root=data_root(root))
    return list(
        catalog.filter(
            content_types=(content,),
            array_paths=(f"data/{content}",),
            folds=(() if fold is None else (fold,)),
        )
    )


def recording(root: Path | None, identifier: str) -> RecordingRecord | Path:
    path = Path(identifier)
    if path.suffix == ".zarr":
        return path
    catalog = load_croissant(root=data_root(root))
    match = catalog.by_recording_id(identifier)
    if match is None:
        raise ValueError(f"Unknown EchoXFlow recording: {identifier}")
    return match


def _volume_geometry(metadata) -> VolumeGeometry:
    geometry = (metadata.raw or {}).get("render_metadata")
    if not geometry:
        raise ValueError("Missing 3D B-mode sector geometry")
    return VolumeGeometry(
        float(geometry.get("DepthStart", 0)),
        float(geometry["DepthEnd"]),
        float(geometry["Width"]),
        float(geometry.get("ElevationWidth", geometry["Width"])),
    )


def load_reference_volume(path: str | Path):
    """Load a single exported 3D frame as ([E,A,R] values in [0, 1], geometry)."""
    with np.load(path, allow_pickle=False) as sample:
        volume = torch.from_numpy(sample["volume"]).float() / 255
        geometry = VolumeGeometry(
            float(sample["depth_start_m"]),
            float(sample["depth_end_m"]),
            float(sample["azimuth_width_rad"]),
            float(sample["elevation_width_rad"]),
        )
    return volume, geometry


def volume_geometry(record: RecordingRecord) -> VolumeGeometry:
    with open_recording(record) as store:
        header = store.load_stream_header(VOLUME_STREAM)
    return _volume_geometry(header.stream.metadata)


def eligible(record: RecordingRecord) -> bool:
    """Apply the exclusion list and the geometry and frame-count thresholds."""
    if record.recording_id in EXCLUDED_RECORDINGS:
        return False
    count = record.frame_count(VOLUME_STREAM)
    if count is not None and count < 20:
        return False
    with open_recording(record) as store:
        header = store.load_stream_header(VOLUME_STREAM)
    geometry = _volume_geometry(header.stream.metadata)
    return (
        header.frame_count >= 20
        and geometry.depth_start <= 0.01
        and geometry.depth_end >= 0.10
        and min(geometry.azimuth_width, geometry.elevation_width) >= 1
    )


def load_volume(
    source: RecordingRecord | Path, *, indices: list[int] | Callable[[int], list[int]]
):
    """Read selected frames, using the stitched frame count for callable selectors."""
    with open_recording(source) as store:
        header = store.load_stream_header(VOLUME_STREAM)
        geometry = _volume_geometry(header.stream.metadata)
        times = header.timestamps
        if times is None:
            raise ValueError(f"Missing 3D frame timestamps for {source}")
        metadata = header.stream.metadata
        if (metadata.stitch_beat_count or 1) > 1:
            stream = store.load_stream(VOLUME_STREAM)
            stitched = prepare_3d_brightness_for_display(
                stream.data, stream.timestamps, stream.metadata.raw
            )
            if not stitched.was_stitched or stitched.timestamps is None:
                raise ValueError(
                    f"EchoXFlow could not stitch the gated 3D recording {store.path}"
                )
            frames = stitched.volumes
            times = stitched.timestamps
        else:
            frames = store.group[header.stream.data_path]

        rows = indices(len(times)) if callable(indices) else indices
        if not rows or min(rows) < 0 or max(rows) >= len(times):
            raise ValueError(
                f"Expected frame indices in [0, {len(times) - 1}] for {store.path}"
            )
        unique, inverse = np.unique(rows, return_inverse=True)
        if isinstance(frames, np.ndarray):
            selected = frames[unique]
        else:
            selected = frames.oindex[unique, :, :, :]
        selected = torch.from_numpy(selected[inverse]).float() / 255
    return selected, times[rows], geometry


def cine_batches(
    source: RecordingRecord | Path,
    stream_name: str,
    *,
    batch_size: int,
    depth_cap: float,
    frame: int | None = None,
):
    """Yield scan-converted frames and native timestamps in bounded batches."""
    with open_recording(source) as store:
        header = store.load_stream_header(stream_name)
        if (
            not isinstance(header.stream, BrightnessModeStream)
            or len(header.shape) != 3
        ):
            raise ValueError(f"Expected a 2D B-mode sector for {stream_name!r}")
        geometry = header.stream.metadata.geometry
        if geometry is None or header.timestamps is None:
            raise ValueError(f"Missing sector geometry or timestamps for {store.path}")
        if frame is not None and not 0 <= frame < header.frame_count:
            raise ValueError(
                f"Expected frame index in [0, {header.frame_count - 1}] for {store.path}"
            )
        first, stop = (0, header.frame_count) if frame is None else (frame, frame + 1)
        for start in range(first, stop, batch_size):
            stream = store.load_stream_slice(
                stream_name, start, min(start + batch_size, stop)
            )
            frames = torch.from_numpy(stream.data).float() / 255
            images = scan_convert(
                frames,
                width=geometry.angle_span_rad,
                depth_start=geometry.depth_start_m,
                depth_end=geometry.depth_end_m,
                size=INPUT_SIZE,
                depth_cap=depth_cap,
            )
            yield images, header.timestamps[start : start + len(images)]
