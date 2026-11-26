import math

import torch

from echo_posed.guidance import correction
from echo_posed.imaging import VolumeGeometry, reslice, scan_convert
from echo_posed.learning import relative_loss
from echo_posed.pose import (
    angle,
    axis_rotation,
    limit_rotation,
    local_rotation,
    project_so3,
)


def test_projection_and_relative_loss():
    target = axis_rotation(torch.tensor([0.1, 0.4, -0.1]), 2)
    gauge = axis_rotation(torch.tensor(0.7), 0)
    # The arccosine is clamped just below one, which leaves about 0.5 mrad.
    assert relative_loss((gauge @ target)[:, None], target[:, None]).item() < 1e-3
    distorted = axis_rotation(torch.tensor([0.2, 0.0, 0.0]), 0) @ target
    assert relative_loss(distorted[:, None], target[:, None]).item() > 0.05
    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    assert torch.allclose(torch.linalg.det(project_so3(reflection)), torch.tensor(1.0))


def test_correction_reaches_target_and_caps_half_turn():
    current_tilt = torch.tensor([0.10, -0.12])
    desired_tilt = torch.tensor([-0.03, 0.04])
    current_probe = axis_rotation(torch.tensor(0.1), 0)
    desired_probe = current_probe @ axis_rotation(torch.tensor(math.pi), 2)
    current = torch.stack((local_rotation(current_tilt), current_probe))
    target = torch.stack((local_rotation(desired_tilt), desired_probe))
    tilt, probe = correction(current, target)
    torch.testing.assert_close(current_tilt + tilt, desired_tilt)
    torch.testing.assert_close(current_probe @ probe, desired_probe)
    torch.testing.assert_close(angle(limit_rotation(probe, 0.1)), torch.tensor(0.1))


def test_native_scan_conversion_preserves_left_to_right_and_depth_cap():
    frames = torch.linspace(0, 1, 31).expand(1, 25, -1)
    left_to_right = scan_convert(
        frames, width=1.2, depth_start=0, depth_end=0.2, depth_cap=0.16, size=64
    )
    assert left_to_right.shape == (1, 1, 64, 64)
    assert left_to_right[0, 0, 34, 24] < left_to_right[0, 0, 34, 38]


def test_identity_reslice_matches_native_azimuth_plane():
    source = torch.rand(1, 3, 33, 33)
    sector = VolumeGeometry(0, 0.12, 1, 1)
    pose = torch.eye(3).expand(1, 2, 3, 3)
    resliced = reslice(source, sector, pose, size=33, width=1, depth=0.12)
    direct = scan_convert(
        source[:, 1].mT, width=1, depth_start=0, depth_end=0.12, size=33
    )
    torch.testing.assert_close(resliced, direct, atol=1e-5, rtol=0)
