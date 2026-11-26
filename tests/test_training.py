import numpy as np
import torch

from echo_posed import learning
from echo_posed.imaging import VolumeGeometry
from echo_posed.pose import angle
from echo_posed.sampling import (
    TrainingRecording,
    augment_batch,
    frame_rows,
    oversample_deep,
    sample_poses,
)
from echo_posed.stitching import stitch_plan


def test_paired_poses_share_the_probe_and_step_the_tilt():
    torch.manual_seed(0)
    target, groups = sample_poses(128)
    world_local, probe = target[:, 0], target[:, 1]
    assert groups.tolist()[:10] == [0] * 5 + [1] * 5 and (groups >= 0).sum() == 65
    for group in range(13):
        members = groups == group
        assert (probe[members] == probe[members][0]).all()
        tilts = torch.rad2deg(angle(world_local[members]))
        torch.testing.assert_close(tilts, torch.tensor([0.0, 10, 10, 10, 10]))
    assert torch.rad2deg(angle(world_local[groups < 0])).max() <= 10 + 1e-3


def test_mirrored_targets_follow_the_mirrored_images():
    torch.manual_seed(1)
    target, _ = sample_poses(6)
    images, mirrored = augment_batch(torch.rand(6, 1, 224, 224), target)
    assert images.shape == (12, 1, 224, 224) and mirrored.shape == (12, 2, 3, 3)
    assert 0 <= images.min() and images.max() <= 1
    half_turn = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    torch.testing.assert_close(mirrored[6:, 1], target[:, 1] @ half_turn)
    torch.testing.assert_close(mirrored[6:, 0], half_turn @ target[:, 0] @ half_turn)


def test_short_recordings_repeat_frames_and_deep_sectors_repeat_recordings():
    torch.manual_seed(2)
    rows = frame_rows(3, 7, True)
    assert sorted(rows[:3]) == sorted(rows[3:6]) == [0, 1, 2] and len(rows) == 7
    assert frame_rows(3, 5, False) == [0, 1, 2, 0, 1]
    shallow, deep = (
        TrainingRecording(None, 0, VolumeGeometry(0, end, 1, 1)) for end in (0.12, 0.17)
    )
    assert oversample_deep([shallow, deep, shallow]) == [0, 1, 1, 1, 2]


def test_learning_rate_decays_from_first_to_last_epoch():
    rates = [learning.learning_rate(epoch, 20) for epoch in range(1, 21)]
    np.testing.assert_allclose([rates[0], rates[-1]], [0.0037, 0.000037])
    assert all(first > second for first, second in zip(rates, rates[1:]))


def test_gated_beats_are_stitched_side_by_side():
    # Two beats of four frames; each beat holds one slab of the elevation range.
    frames = np.zeros((9, 2, 3, 4), dtype=np.uint8)
    frames[:4] = 10
    frames[4:] = 200
    timestamps = np.arange(9) * 0.25
    metadata = {"qrs_trigger_times": [0.0, 1.0, 2.0]}
    plan, order = stitch_plan(frames, timestamps, metadata, 2)
    assert sorted(order) == [0, 1] and len(plan.stitched_indices_matrix) == 4
