"""ECG-gated stitching of multi-beat 3D recordings, as used for training."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def beat_gating(group) -> tuple[dict[str, Any], int]:
    """Return the store attributes with QRS times, and the number of gated beats."""
    attrs = dict(group.attrs)
    stream = attrs.get("recording_contract", {}).get("modalities", {})
    stream = stream.get("3d_brightness_mode", {})
    if (path := stream.get("frame_locations")) is None:
        return attrs, 1
    count = int(np.unique(np.asarray(group[path][:])).size)
    if count > 1:
        events = np.asarray(group[stream["source_qrs_events"]][:])
        attrs["qrs_trigger_times"] = events
    return attrs, count


def _sorted_finite(values) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    return values[np.isfinite(values)]


def stitch_plan(volumes, timestamps, metadata, beat_count, *, score_frames=None):
    """Plan one stitched cycle of [T,E,A,R] volumes and choose the beat order."""
    plan = build_stitch_index_plan(
        volumes,
        timestamps,
        metadata["qrs_trigger_times"],
        beat_count,
        metadata=metadata,
    )
    return plan, choose_stitch_beat_order(volumes, plan, max_score_frames=score_frames)


def display_stitch_plan(volumes, timestamps, metadata, beat_count):
    """Plan on the best gating clock and score the beat order on every frame."""
    qrs = _sorted_finite(metadata["qrs_trigger_times"])
    relative, relative_qrs = _relative_display_axis(
        timestamps, qrs, metadata, beat_count
    )
    axis, axis_qrs = _best_stitching_axis(
        timestamps, relative, qrs, relative_qrs, beat_count
    )
    plan = build_stitch_index_plan(volumes, axis, axis_qrs, beat_count)
    return plan, choose_stitch_beat_order(volumes, plan)


def build_stitch_index_plan(
    volumes,
    timestamps,
    qrs_trigger_times,
    beat_count,
    *,
    metadata=None,
):
    shape = tuple(int(dim) for dim in volumes.shape)
    raw_qrs = _sorted_finite(qrs_trigger_times)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    relative_timestamps, relative_qrs = _relative_display_axis(
        timestamps, raw_qrs, metadata, beat_count
    )
    if timestamps.size == shape[0] and relative_qrs.size >= 2:
        stitch_timestamps, stitch_qrs = _best_stitching_axis(
            timestamps, relative_timestamps, raw_qrs, relative_qrs, beat_count
        )
        beats, start = _stitching_beat_frame_indices(
            stitch_timestamps, stitch_qrs, beat_count
        )
        if beats:
            beats = refine_stitching_beat_boundaries(volumes, beats)
            beats = trim_stitching_outer_boundaries(volumes, beats)
            plan = _build_interpolated_stitch_index_plan(
                stitch_timestamps=stitch_timestamps,
                stitch_qrs=stitch_qrs,
                qrs_start_index=start,
                beat_frame_indices=beats,
                beat_count=beat_count,
            )
            if plan is not None:
                return plan
    raise ValueError(
        f"Could not stitch a {beat_count}-beat recording from {raw_qrs.size} QRS times"
    )


@dataclass(frozen=True)
class StitchIndexPlan:
    stitch_beat_count: int
    stitched_indices_matrix: tuple[tuple[int, ...], ...]
    interpolation_indices_matrix: tuple[tuple[tuple[int, int], ...], ...]
    interpolation_weights_matrix: tuple[tuple[float, ...], ...]


def refine_stitching_beat_boundaries(
    volumes: Any,
    beat_frame_indices: list[np.ndarray],
    *,
    search_radius: int = 3,
    min_peak_ratio: float = 2.0,
    min_candidate_over_default_ratio: float = 2.0,
) -> list[np.ndarray]:
    """Move QRS-derived beat boundaries to clear acquisition slab changes."""
    if len(beat_frame_indices) <= 1:
        return beat_frame_indices

    intervals = [
        np.asarray(indices, dtype=np.int64).copy() for indices in beat_frame_indices
    ]
    radius = max(0, int(search_radius))
    frame_count = int(volumes.shape[0])
    frame_cache: dict[int, np.ndarray] = {}

    def _frame(index: int) -> np.ndarray:
        cached = frame_cache.get(index)
        if cached is None:
            frame = np.asarray(volumes[index], dtype=np.float32)
            strides = tuple(
                max(1, int(np.ceil(size / target)))
                for size, target in zip(frame.shape, (32, 48, 64))
            )
            cached = frame[:: strides[0], :: strides[1], :: strides[2]]
            frame_cache[index] = cached
        return cached

    for boundary_index in range(len(intervals) - 1):
        right_start = int(intervals[boundary_index + 1][0])
        default_pair_start = right_start - 1
        candidates: list[tuple[float, int, int]] = []
        for offset in range(-radius, radius + 1):
            pair_start = default_pair_start + offset
            if pair_start < 0 or pair_start + 1 >= frame_count:
                continue
            candidates.append((0.0, offset, pair_start))

        if not candidates:
            continue

        window_frames = [
            _frame(pair_start + side)
            for _, _, pair_start in candidates
            for side in (0, 1)
        ]
        window_scale = max(
            float(
                np.percentile(
                    np.abs(
                        np.concatenate([frame.reshape(-1) for frame in window_frames])
                    ),
                    95.0,
                )
            ),
            1.0,
        )
        candidates = [
            (
                _normalized_frame_difference(
                    _frame(pair_start), _frame(pair_start + 1), scale=window_scale
                ),
                offset,
                pair_start,
            )
            for _, offset, pair_start in candidates
        ]

        values = np.asarray([item[0] for item in candidates], dtype=np.float64)
        local_median = max(float(np.median(values)), 1e-8)
        best_score, best_offset, best_pair_start = max(
            candidates, key=lambda item: item[0]
        )
        default_score = next(
            (item[0] for item in candidates if item[1] == 0), float("inf")
        )
        peak_ratio = float(best_score / local_median)
        candidate_over_default = float(best_score / max(default_score, 1e-8))

        should_move = (
            best_offset != 0
            and peak_ratio >= float(min_peak_ratio)
            and candidate_over_default >= float(min_candidate_over_default_ratio)
        )
        if should_move:
            corrected_right_start = int(best_pair_start + 1)
            left_start = int(intervals[boundary_index][0])
            right_stop = int(intervals[boundary_index + 1][-1]) + 1
            if left_start < corrected_right_start < right_stop:
                intervals[boundary_index] = np.arange(
                    left_start, corrected_right_start, dtype=np.int64
                )
                intervals[boundary_index + 1] = np.arange(
                    corrected_right_start, right_stop, dtype=np.int64
                )

    return intervals


def _normalized_frame_difference(
    left: np.ndarray, right: np.ndarray, *, scale: float | None = None
) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0
    left_values = np.asarray(left[finite], dtype=np.float32)
    right_values = np.asarray(right[finite], dtype=np.float32)
    if scale is None:
        scale_values = np.abs(np.concatenate((left_values, right_values)))
        scale = max(float(np.percentile(scale_values, 95.0)), 1.0)
    return float(np.mean(np.abs(left_values - right_values)) / float(scale))


def trim_stitching_outer_boundaries(
    volumes: Any,
    beat_frame_indices: list[np.ndarray],
    *,
    search_radius: int = 3,
    min_peak_ratio: float = 2.0,
    min_candidate_over_default_ratio: float = 2.0,
) -> list[np.ndarray]:
    """Trim clear acquisition changes just inside the first or last QRS edge."""
    if not beat_frame_indices:
        return beat_frame_indices

    intervals = [
        np.asarray(indices, dtype=np.int64).copy() for indices in beat_frame_indices
    ]
    radius = max(0, int(search_radius))
    frame_count = int(volumes.shape[0])
    frame_shape = tuple(int(value) for value in volumes.shape[1:])
    strides = tuple(
        max(1, int(np.ceil(size / target)))
        for size, target in zip(frame_shape, (32, 48, 64))
    )
    frame_cache: dict[int, np.ndarray] = {}

    def _frame(index: int) -> np.ndarray:
        cached = frame_cache.get(index)
        if cached is None:
            frame = np.asarray(volumes[index], dtype=np.float32)
            cached = frame[:: strides[0], :: strides[1], :: strides[2]]
            frame_cache[index] = cached
        return cached

    def _edge_offset(default_pair_start: int, *, inward: str) -> int:
        candidates: list[tuple[int, int]] = []
        for offset in range(-radius, radius + 1):
            pair_start = default_pair_start + offset
            if pair_start < 0 or pair_start + 1 >= frame_count:
                continue
            candidates.append((offset, pair_start))
        if not candidates:
            return 0

        unique_indices = sorted(
            {pair_start + side for _, pair_start in candidates for side in (0, 1)}
        )
        window_scale = max(
            float(
                np.percentile(
                    np.abs(
                        np.concatenate(
                            [_frame(index).reshape(-1) for index in unique_indices]
                        )
                    ),
                    95.0,
                )
            ),
            1.0,
        )
        scored = [
            (
                _normalized_frame_difference(
                    _frame(pair_start), _frame(pair_start + 1), scale=window_scale
                ),
                offset,
                pair_start,
            )
            for offset, pair_start in candidates
        ]
        values = np.asarray([item[0] for item in scored], dtype=np.float64)
        local_median = max(float(np.median(values)), 1e-8)
        if inward == "start":
            eligible = [item for item in scored if item[2] + 1 > int(intervals[0][0])]
        else:
            eligible = [
                item for item in scored if item[2] + 1 < int(intervals[-1][-1]) + 1
            ]
        if not eligible:
            return 0
        best_score, best_offset, _ = max(eligible, key=lambda item: item[0])
        default_score = next((item[0] for item in scored if item[1] == 0), None)
        peak_ratio = float(best_score / local_median)
        candidate_over_default = (
            float("inf")
            if default_score is None
            else float(best_score / max(default_score, 1e-8))
        )
        should_trim = peak_ratio >= float(
            min_peak_ratio
        ) and candidate_over_default >= float(min_candidate_over_default_ratio)
        return int(best_offset) if should_trim else 0

    original_start = int(intervals[0][0])
    original_stop = int(intervals[-1][-1]) + 1
    start_offset = _edge_offset(original_start - 1, inward="start")
    stop_offset = _edge_offset(original_stop - 1, inward="stop")

    corrected_start = original_start + max(0, start_offset)
    corrected_stop = original_stop + min(0, stop_offset)
    if corrected_start > original_start and corrected_start < int(intervals[0][-1]):
        intervals[0] = np.arange(
            corrected_start, int(intervals[0][-1]) + 1, dtype=np.int64
        )
    if corrected_stop < original_stop and corrected_stop > int(intervals[-1][0]) + 1:
        intervals[-1] = np.arange(int(intervals[-1][0]), corrected_stop, dtype=np.int64)

    return intervals


def _build_interpolated_stitch_index_plan(
    *,
    stitch_timestamps: np.ndarray,
    stitch_qrs: np.ndarray,
    qrs_start_index: int,
    beat_frame_indices: list[np.ndarray],
    beat_count: int,
) -> StitchIndexPlan | None:
    counts = np.asarray(
        [len(indices) for indices in beat_frame_indices], dtype=np.int64
    )
    if counts.size != beat_count or np.any(counts < 2):
        return None
    target_count = max(2, int(np.rint(np.median(counts))))
    target_phases = np.arange(target_count, dtype=np.float64) / float(target_count)

    phase_series: list[tuple[np.ndarray, np.ndarray]] = []
    for beat_idx, indices in enumerate(beat_frame_indices):
        qrs_index = qrs_start_index + beat_idx
        start, stop = stitch_qrs[qrs_index], stitch_qrs[qrs_index + 1]
        duration = stop - start
        if not np.isfinite(duration) or duration <= 0:
            return None
        phases = np.mod((stitch_timestamps[indices] - start) / duration, 1.0)
        order = np.argsort(phases, kind="stable")
        phase_series.append((phases[order], np.asarray(indices, dtype=np.int64)[order]))

    nearest_rows: list[tuple[int, ...]] = []
    interpolation_rows: list[tuple[tuple[int, int], ...]] = []
    weight_rows: list[tuple[float, ...]] = []
    for phase in target_phases:
        nearest_row: list[int] = []
        interpolation_row: list[tuple[int, int]] = []
        weight_row: list[float] = []
        for phases, indices in phase_series:
            lower_index, upper_index, weight = _periodic_phase_bracket(
                phases, indices, float(phase)
            )
            # Interpolation blends across the cycle wrap, but the frame used to
            # compare beat boundaries must stay within its beat.
            nearest_row.append(
                _nearest_phase_index_in_beat(phases, indices, float(phase))
            )
            interpolation_row.append((lower_index, upper_index))
            weight_row.append(weight)
        nearest_rows.append(tuple(nearest_row))
        interpolation_rows.append(tuple(interpolation_row))
        weight_rows.append(tuple(weight_row))

    return StitchIndexPlan(
        stitch_beat_count=beat_count,
        stitched_indices_matrix=tuple(nearest_rows),
        interpolation_indices_matrix=tuple(interpolation_rows),
        interpolation_weights_matrix=tuple(weight_rows),
    )


def _nearest_phase_index_in_beat(
    phases: np.ndarray,
    indices: np.ndarray,
    target_phase: float,
) -> int:
    position = int(np.searchsorted(phases, target_phase, side="right"))
    if position == 0:
        return int(indices[0])
    if position == phases.size:
        return int(indices[-1])
    lower_distance = target_phase - float(phases[position - 1])
    upper_distance = float(phases[position]) - target_phase
    return (
        int(indices[position - 1])
        if lower_distance <= upper_distance
        else int(indices[position])
    )


def _periodic_phase_bracket(
    phases: np.ndarray,
    indices: np.ndarray,
    target_phase: float,
) -> tuple[int, int, float]:
    position = int(np.searchsorted(phases, target_phase, side="right"))
    if position == 0:
        lower_phase, lower_index = float(phases[-1] - 1.0), int(indices[-1])
        upper_phase, upper_index = float(phases[0]), int(indices[0])
    elif position == phases.size:
        lower_phase, lower_index = float(phases[-1]), int(indices[-1])
        upper_phase, upper_index = float(phases[0] + 1.0), int(indices[0])
    else:
        lower_phase, lower_index = (
            float(phases[position - 1]),
            int(indices[position - 1]),
        )
        upper_phase, upper_index = float(phases[position]), int(indices[position])
    width = upper_phase - lower_phase
    if width <= 1e-12:
        return lower_index, upper_index, 0.0
    weight = float(np.clip((target_phase - lower_phase) / width, 0.0, 1.0))
    return lower_index, upper_index, weight


def choose_stitch_beat_order(
    volumes: Any,
    plan: StitchIndexPlan,
    *,
    max_score_frames: int | None = None,
) -> tuple[int, ...]:
    candidates = _candidate_beat_orders(plan.stitch_beat_count)
    default = tuple(reversed(range(plan.stitch_beat_count)))
    best_order = default if default in candidates else candidates[0]
    row_indices = _sample_indices_for_count(
        len(plan.stitched_indices_matrix), max_score_frames
    )
    boundaries = _load_boundary_planes(
        volumes, plan.stitched_indices_matrix, row_indices, plan.stitch_beat_count
    )
    edge_scores = {}
    best_score = _boundary_plane_score(boundaries, best_order, edge_scores)
    for order in candidates:
        if order == best_order:
            continue
        score = _boundary_plane_score(boundaries, order, edge_scores)
        if np.isfinite(score) and (not np.isfinite(best_score) or score < best_score):
            best_score = score
            best_order = order
    return best_order


def load_3d_brightness_frames(volumes: Any, frame_indices: list[int]) -> np.ndarray:
    """Read zarr frames; zarr groups them by chunk and decodes concurrently."""
    return np.asarray(volumes.oindex[list(frame_indices), :, :, :])


def load_stitched_3d_brightness_frames(
    volumes: Any,
    plan: StitchIndexPlan,
    frame_indices: list[int] | tuple[int, ...] | range,
    order: tuple[int, ...],
) -> np.ndarray:
    rows = [int(index) for index in frame_indices]
    if not rows:
        raise ValueError("At least one stitched frame index is required.")

    frame_cache: dict[int, np.ndarray] = {}
    if hasattr(volumes, "oindex"):
        required_frames = set()
        for row in rows:
            for beat_idx in order:
                lower, upper = plan.interpolation_indices_matrix[row][beat_idx]
                required_frames.add(lower)
                if plan.interpolation_weights_matrix[row][beat_idx] > 1e-7:
                    required_frames.add(upper)
        source_indices = sorted(required_frames)
        decoded = load_3d_brightness_frames(volumes, source_indices)
        frame_cache.update(zip(source_indices, decoded, strict=True))

    def _frame(index: int) -> np.ndarray:
        cached = frame_cache.get(index)
        if cached is None:
            cached = np.asarray(volumes[index])
            frame_cache[index] = cached
        return cached

    def _beat_frame(row: int, beat_idx: int) -> np.ndarray:
        lower_index, upper_index = plan.interpolation_indices_matrix[row][beat_idx]
        weight = float(plan.interpolation_weights_matrix[row][beat_idx])
        lower = _frame(int(lower_index))
        if lower_index == upper_index or weight <= 1e-7:
            return lower
        upper = _frame(int(upper_index))
        if weight >= 1.0 - 1e-7:
            return upper
        blended = (
            np.asarray(lower, dtype=np.float32) * (1.0 - weight)
            + np.asarray(upper, dtype=np.float32) * weight
        )
        # Preserve the source intensity contract. The 3D loader normalizes
        # uint8 input to [0, 1] based on dtype; returning raw-range float32
        # here bypasses that normalization and saturates stitched sectors.
        source_dtype = np.asarray(lower).dtype
        if np.issubdtype(source_dtype, np.integer):
            limits = np.iinfo(source_dtype)
            return np.clip(np.rint(blended), limits.min, limits.max).astype(
                source_dtype
            )
        return blended.astype(source_dtype, copy=False)

    return np.stack(
        [
            np.concatenate(
                [_beat_frame(row, beat_idx) for beat_idx in order],
                axis=0,
            )
            for row in rows
        ],
        axis=0,
    )


def _candidate_beat_orders(beat_count: int) -> tuple[tuple[int, ...], ...]:
    n = max(1, int(beat_count))
    if n <= 4:
        return _permutations(tuple(range(n)))
    return _unique_orders(
        (*_antisymmetric_beat_orders(n), *_cyclic_and_anticyclic_beat_orders(n))
    )


def _cyclic_and_anticyclic_beat_orders(beat_count: int) -> tuple[tuple[int, ...], ...]:
    n = max(1, int(beat_count))
    forward = tuple(range(n))
    backward = tuple(reversed(forward))
    orders = []
    for base in (forward, backward):
        for shift in range(n):
            orders.append(base[shift:] + base[:shift])
    return _unique_orders(tuple(orders))


def _unique_orders(orders: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for order in orders:
        if order not in seen:
            seen.add(order)
            out.append(order)
    return tuple(out)


def _antisymmetric_beat_orders(beat_count: int) -> tuple[tuple[int, ...], ...]:
    n = int(beat_count)
    if n <= 1:
        return ((0,),)

    pairs = [(index, n - 1 - index) for index in range(n // 2)]
    half_orders = _permutations(tuple(range(len(pairs))))
    orientations = tuple(range(1 << len(pairs)))
    orders: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for half_order in half_orders:
        for mask in orientations:
            left: list[int] = []
            right: list[int] = []
            for out_pos, pair_index in enumerate(half_order):
                low, high = pairs[pair_index]
                if (mask >> out_pos) & 1:
                    left.append(high)
                    right.append(low)
                else:
                    left.append(low)
                    right.append(high)
            middle = [n // 2] if n % 2 else []
            order = tuple(left + middle + right[::-1])
            if order not in seen:
                seen.add(order)
                orders.append(order)
    return tuple(orders)


def _permutations(values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not values:
        return ((),)
    out: list[tuple[int, ...]] = []
    for index, value in enumerate(values):
        remaining = values[:index] + values[index + 1 :]
        out.extend((value, *tail) for tail in _permutations(remaining))
    return tuple(out)


def _sample_indices_for_count(count: int, max_samples: int | None) -> tuple[int, ...]:
    if count < 1:
        return ()
    if max_samples is None or int(max_samples) >= count:
        return tuple(range(count))
    samples = max(1, int(max_samples))
    return tuple(
        sorted({int(round(value)) for value in np.linspace(0, count - 1, num=samples)})
    )


def _load_boundary_planes(
    volumes: Any,
    stitched_indices_matrix: tuple[tuple[int, ...], ...],
    row_indices: tuple[int, ...],
    beat_count: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    boundaries: list[list[tuple[np.ndarray, np.ndarray]]] = []
    plane_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if hasattr(volumes, "oindex"):
        source_indices = sorted(
            {index for row in row_indices for index in stitched_indices_matrix[row]}
        )
        if source_indices:
            planes = volumes.oindex[source_indices, [0, volumes.shape[1] - 1], :, :]
            plane_cache.update(
                (
                    index,
                    (
                        np.asarray(first, dtype=np.float64),
                        np.asarray(last, dtype=np.float64),
                    ),
                )
                for index, (first, last) in zip(source_indices, planes, strict=True)
            )
    for row_idx in row_indices:
        indices = stitched_indices_matrix[row_idx]
        row_boundaries: list[tuple[np.ndarray, np.ndarray]] = []
        for beat_idx in range(beat_count):
            raw_idx = int(indices[beat_idx])
            cached = plane_cache.get(raw_idx)
            if cached is None:
                frame = np.asarray(volumes[raw_idx])
                cached = (
                    np.asarray(frame[0], dtype=np.float64),
                    np.asarray(frame[-1], dtype=np.float64),
                )
                plane_cache[raw_idx] = cached
            row_boundaries.append(cached)
        boundaries.append(row_boundaries)
    return boundaries


def _boundary_plane_score(
    boundaries: list[list[tuple[np.ndarray, np.ndarray]]],
    beat_order: tuple[int, ...],
    edge_scores: dict[tuple[int, int, int], float | None] | None = None,
) -> float:
    if edge_scores is None:
        edge_scores = {}
    scores: list[float] = []
    for row_index, row_boundaries in enumerate(boundaries):
        for left_beat, right_beat in zip(beat_order[:-1], beat_order[1:], strict=True):
            key = row_index, left_beat, right_beat
            if key not in edge_scores:
                left = row_boundaries[left_beat][1]
                right = row_boundaries[right_beat][0]
                finite = np.isfinite(left) & np.isfinite(right)
                if np.any(finite):
                    left = left[finite]
                    right = right[finite]
                    values = np.abs(np.concatenate((left, right)))
                    scale = max(float(np.nanpercentile(values, 95.0)), 1.0)
                    edge_scores[key] = float(np.nanmean(np.abs(left - right)) / scale)
                else:
                    edge_scores[key] = None
            score = edge_scores[key]
            if score is not None:
                scores.append(score)
    return float(np.mean(scores)) if scores else float("inf")


def _stitching_beat_frame_indices(
    timestamps: np.ndarray,
    qrs: np.ndarray,
    beat_count: int,
) -> tuple[list[np.ndarray], int]:
    for start_index in range(0, max(0, qrs.size - beat_count)):
        intervals: list[np.ndarray] = []
        for offset in range(beat_count):
            qrs_index = start_index + offset
            start, stop = qrs[qrs_index], qrs[qrs_index + 1]
            indices = np.where((timestamps >= start) & (timestamps < stop))[0]
            if indices.size == 0:
                break
            intervals.append(indices)
        if len(intervals) == beat_count:
            return intervals, start_index
    return [], 0


def _relative_display_axis(
    source_timestamps: np.ndarray | None,
    raw_qrs: np.ndarray,
    metadata: Mapping[str, Any] | None,
    beat_count: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    if source_timestamps is None:
        return None, _relative_qrs_candidates(raw_qrs, metadata)[0]
    source = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
    if source.size == 0:
        return source, _relative_qrs_candidates(raw_qrs, metadata)[0]
    candidates: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
    for rel_ts in _relative_timestamp_candidates(source, raw_qrs, metadata):
        for rel_qrs in _relative_qrs_candidates(raw_qrs, metadata):
            candidates.append(
                (_stitch_axis_score(rel_ts, rel_qrs, beat_count), rel_ts, rel_qrs)
            )
    if not candidates:
        return source, raw_qrs
    _, rel_ts, rel_qrs = max(candidates, key=_relative_axis_candidate_score)
    return rel_ts, rel_qrs


def _relative_axis_candidate_score(
    item: tuple[tuple[int, int], np.ndarray, np.ndarray],
) -> tuple[int, int, float]:
    score, rel_ts, _ = item
    finite = rel_ts[np.isfinite(rel_ts)]
    origin_distance = abs(float(finite[0])) if finite.size else float("inf")
    return int(score[0]), int(score[1]), -origin_distance


def _relative_timestamp_candidates(
    source_timestamps: np.ndarray,
    raw_qrs: np.ndarray,
    metadata: Mapping[str, Any] | None,
) -> tuple[np.ndarray, ...]:
    candidates = [np.asarray(source_timestamps, dtype=np.float64).reshape(-1)]
    metadata_origin = _volume_time_origin_s(metadata)
    if metadata_origin is not None:
        candidates.append(candidates[0] - float(metadata_origin))
    if raw_qrs.size > 0:
        candidates.append(candidates[0] - float(raw_qrs[0]))
    return tuple(_unique_timeline_candidates(candidates))


def _relative_qrs_candidates(
    raw_qrs: np.ndarray, metadata: Mapping[str, Any] | None
) -> tuple[np.ndarray, ...]:
    qrs = np.asarray(raw_qrs, dtype=np.float64).reshape(-1)
    candidates = [qrs]
    metadata_origin = _volume_time_origin_s(metadata)
    if metadata_origin is not None:
        candidates.append(qrs - float(metadata_origin))
    if qrs.size > 0:
        candidates.append(qrs - float(qrs[0]))
    return tuple(_unique_timeline_candidates(candidates))


def _unique_timeline_candidates(candidates: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for candidate in candidates:
        arr = np.asarray(candidate, dtype=np.float64).reshape(-1)
        if not any(
            arr.shape == existing.shape and np.allclose(arr, existing, equal_nan=True)
            for existing in unique
        ):
            unique.append(arr)
    return unique


def _best_stitching_axis(
    source_timestamps: np.ndarray,
    relative_timestamps: np.ndarray,
    source_qrs: np.ndarray,
    relative_qrs: np.ndarray,
    beat_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_score = _stitch_axis_score(source_timestamps, source_qrs, beat_count)
    relative_score = _stitch_axis_score(relative_timestamps, relative_qrs, beat_count)
    if relative_score > source_score:
        return relative_timestamps, relative_qrs
    return source_timestamps, source_qrs


def _stitch_axis_score(
    timestamps: np.ndarray, qrs: np.ndarray, beat_count: int
) -> tuple[int, int]:
    intervals, _ = _stitching_beat_frame_indices(timestamps, qrs, beat_count)
    if intervals:
        return len(intervals), min(len(indices) for indices in intervals)
    return _beat_bucket_score(timestamps, qrs, beat_count)


def _beat_bucket_score(
    timestamps: np.ndarray, qrs: np.ndarray, beat_count: int
) -> tuple[int, int]:
    buffers = _beat_buffers(timestamps, qrs, beat_count)
    nonempty = sum(1 for buffer in buffers if buffer)
    min_len = min((len(buffer) for buffer in buffers), default=0)
    return nonempty, min_len


def _beat_buffers(
    timestamps: np.ndarray, qrs: np.ndarray, beat_count: int
) -> list[list[int]]:
    beat_buffers: list[list[int]] = [list() for _ in range(beat_count)]
    for qrs_index, start in enumerate(qrs):
        stop = (
            float(qrs[qrs_index + 1])
            if qrs_index + 1 < qrs.size
            else _final_stop(timestamps, float(start))
        )
        frame_indices = np.where((timestamps >= float(start)) & (timestamps < stop))[0]
        beat_buffers[qrs_index % beat_count].extend(
            int(index) for index in frame_indices
        )
    return beat_buffers


def _volume_time_origin_s(metadata: Mapping[str, Any] | None) -> float | None:
    reference = None
    for source in _metadata_sources(metadata):
        raw_reference = source.get("time_reference")
        if isinstance(raw_reference, Mapping):
            reference = raw_reference
            break
        raw_reference = source.get("volume_time_reference")
        if isinstance(raw_reference, Mapping):
            reference = raw_reference
            break
    if not isinstance(reference, Mapping):
        return None
    raw = reference.get("volume_time_origin_s")
    if raw is None:
        raw = reference.get("origin_s")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _metadata_sources(
    metadata: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    if metadata is None:
        return ()
    sources: list[Mapping[str, Any]] = [metadata]
    recording_manifest = metadata.get("recording_manifest")
    if isinstance(recording_manifest, Mapping):
        sources.append(recording_manifest)
        manifest_metadata = recording_manifest.get("metadata")
        if isinstance(manifest_metadata, Mapping):
            sources.append(manifest_metadata)
        ecg_metadata = recording_manifest.get("ecg")
        if isinstance(ecg_metadata, Mapping):
            sources.append(ecg_metadata)
    public_metadata = metadata.get("metadata")
    if isinstance(public_metadata, Mapping):
        sources.append(public_metadata)
    return tuple(sources)


def _final_stop(timestamps: np.ndarray, start: float) -> float:
    if timestamps.size == 0:
        return start
    return max(float(timestamps[-1]), start)
