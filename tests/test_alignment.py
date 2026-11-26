"""Members learn their own frames, so alignment matrices can be far from identity."""

import math

import pytest
import torch
from torch import nn

from echo_posed import guidance, learning
from echo_posed.imaging import VolumeGeometry
from echo_posed.model import (
    BUNDLE_FORMAT,
    INPUT_SIZE,
    MODEL_CONFIG,
    Ensemble,
    PoseRegressor,
)
from echo_posed.pose import (
    TARGETS,
    angle,
    axis_angle_rotation,
    local_rotation,
    probe_rotation,
)

MEMBERS = 4


def large_alignments():
    """Distinct rotations of 72 to 154 degrees for every member and factor."""
    generator = torch.Generator().manual_seed(0)
    axes = torch.nn.functional.normalize(
        torch.randn(MEMBERS, 2, 3, generator=generator), dim=-1
    )
    degrees = torch.tensor(
        [[154.0, 104.0], [150.0, 113.0], [138.0, 120.0], [72.0, 93.0]]
    )
    return axis_angle_rotation(axes * torch.deg2rad(degrees)[..., None])


def poses(count):
    generator = torch.Generator().manual_seed(1)
    unit = torch.rand(count, 5, generator=generator) * 2 - 1
    world = local_rotation(unit[:, :2] * math.radians(10))
    probe = probe_rotation(unit[:, 2:] * torch.deg2rad(torch.tensor([10, 10, 180.0])))
    return torch.stack((world, probe), dim=1)


def encode(pose):
    """Carry [B,2,3,3] poses in the first pixels of a frame."""
    images = torch.zeros(len(pose), 1, INPUT_SIZE, INPUT_SIZE)
    images[:, 0, 0, :18] = pose.flatten(1)
    return images


class FrameMember(nn.Module):
    """Predict the encoded pose in a member-specific frame."""

    def __init__(self, alignment):
        super().__init__()
        self.register_buffer("alignment", alignment)

    def forward(self, images):
        pose = images[:, 0, 0, :18].reshape(-1, 2, 3, 3)
        return self.alignment.mT @ pose


def frame_ensemble():
    alignments = large_alignments()
    return Ensemble([FrameMember(item) for item in alignments], alignments)


def test_ensemble_undoes_large_alignments_on_both_factors():
    truth = poses(6)
    prediction = frame_ensemble()(encode(truth))
    torch.testing.assert_close(prediction.rotations, truth, atol=1e-5, rtol=0)
    assert prediction.uncertainty_rad.max() < 1e-3
    # Treating the alignments as identity would be far off in both factors.
    raw = torch.stack([member(encode(truth)) for member in frame_ensemble().models])
    assert torch.rad2deg(angle(truth.mT @ raw)).min() > 60


def test_checkpoint_alignments_are_applied_on_load(tmp_path):
    torch.manual_seed(2)
    members = [PoseRegressor().eval() for _ in range(MEMBERS)]
    alignments = large_alignments()
    path = tmp_path / "bundle.pth"
    torch.save(
        {
            "format": BUNDLE_FORMAT,
            "version": 2,
            "input_size": INPUT_SIZE,
            "rotation_target_names": list(TARGETS),
            "model_config": MODEL_CONFIG,
            "model_state_dicts": [member.state_dict() for member in members],
            "gauge_alignment": {"alignments": alignments},
        },
        path,
    )
    images = torch.rand(2, 1, INPUT_SIZE, INPUT_SIZE)
    loaded = Ensemble.load(path)
    with torch.inference_mode():
        for index, member in enumerate(members):
            torch.testing.assert_close(
                loaded.adapters[index] @ loaded.models[index](images),
                alignments[index] @ member(images),
            )
    torch.testing.assert_close(loaded.adapters, alignments)


def test_calibration_recovers_large_alignments():
    truth = poses(32)
    alignments = large_alignments()
    fitted = learning.calibrate(
        [FrameMember(item) for item in alignments],
        [(encode(truth), truth, 0)],
        "cpu",
    )
    torch.testing.assert_close(fitted, alignments, atol=1e-5, rtol=0)


def test_evaluation_reports_no_error_for_perfect_members(monkeypatch):
    truth = poses(8)
    monkeypatch.setattr(learning, "select_records", lambda *args: [None])
    monkeypatch.setattr(
        learning, "batches", lambda *args, **kwargs: [(encode(truth), truth, 4)]
    )
    metrics = learning.evaluate(
        frame_ensemble(), None, samples=8, limit=None, workers=0, seed=0
    )
    for name in (
        "world_local_mae_deg",
        "probe_mae_deg",
        "relative_mae_deg",
        "uncertainty_deg",
    ):
        assert metrics[name] == pytest.approx(0, abs=0.05)


def test_guidance_reaches_the_target_with_large_alignments(monkeypatch):
    monkeypatch.setattr(guidance, "reslice", lambda volume, geometry, pose, **_: encode(pose))
    initial = torch.stack(
        (
            local_rotation(torch.deg2rad(torch.tensor([8.0, -6.0]))),
            probe_rotation(torch.deg2rad(torch.tensor([5.0, -4.0, 25.0]))),
        )
    )
    target = torch.stack(
        (
            local_rotation(torch.deg2rad(torch.tensor([-2.0, 4.0]))),
            probe_rotation(torch.deg2rad(torch.tensor([-3.0, 2.0, -20.0]))),
        )
    )
    trace, _, _ = guidance.simulate(
        frame_ensemble(),
        torch.zeros(1, 2, 2, 2),
        VolumeGeometry(0, 0.12, 1, 1),
        initial,
        target,
        depth=0.1,
        width=1,
        steps=40,
    )
    assert max(trace[0]["true_factor_errors_deg"]) > 10
    assert max(trace[-1]["true_factor_errors_deg"]) < 1.5
    assert trace[-1]["predicted_error_deg"] <= 1


def test_correction_between_frames_ignores_a_shared_probe_frame():
    current, target = poses(2)
    frame = large_alignments()[0, 1]
    shifted = [torch.stack((pose[0], frame @ pose[1])) for pose in (current, target)]
    torch.testing.assert_close(
        guidance.correction(*shifted)[1], guidance.correction(current, target)[1]
    )
