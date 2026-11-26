import math

import numpy as np
import pytest
import torch

pytest.importorskip("skimage")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

from benchmarks.pasdeloup import (  # noqa: E402
    CVCNetController,
    EchoPosedController,
    probe_pose,
    project_two_dof,
    run_trial,
)
from echo_posed.pose import probe_rotation  # noqa: E402


def two_dof(alpha, beta):
    return Rotation.from_euler("ZX", (alpha, beta), degrees=True)


def test_projection_matches_a_dense_search():
    rng = np.random.default_rng(0)
    for target in Rotation.random(20, random_state=rng):
        alpha, beta, distance = project_two_dof(target.as_matrix())
        reached = two_dof(alpha, beta).inv() * target
        np.testing.assert_allclose(np.rad2deg(reached.magnitude()), distance, atol=1e-6)
        assert abs(beta) <= 20
        grid = np.stack(
            np.meshgrid(np.linspace(-180, 180, 721), np.linspace(-20, 20, 81)), -1
        ).reshape(-1, 2)
        searched = Rotation.from_euler("ZX", grid, degrees=True).inv() * target
        assert distance <= np.rad2deg(searched.magnitude()).min() + 1e-6


def test_echo_posed_controller_reaches_a_reachable_pose():
    initial = probe_rotation(torch.tensor((0.1, -0.2, 0.3)))
    goal = probe_pose(initial, 40.0, -12.0)
    # The observation is the pose itself, so the predictor is exact.
    predict = lambda poses: poses.numpy()  # noqa: E731
    controller = EchoPosedController(predict, [goal[None].numpy()])
    result = run_trial(controller, lambda pose: pose[None], initial)
    assert result["converged"] and result["updates"] <= 100
    assert all(abs(row["step_deg"]) <= 3 for row in result["trace"])
    final = torch.tensor(result["final_probe"])
    error = torch.arccos(((final.T @ goal).trace() - 1).clamp(-2, 2) / 2)
    assert error < math.radians(10)


def test_controllers_stop_when_they_cannot_improve():
    flat = np.zeros((1, 23))
    flat[0, 0] = flat[0, 12] = 1
    goal = np.zeros((1, 23))
    goal[0, 2] = goal[0, 22] = 1
    controller = CVCNetController(lambda images: flat, [goal])
    result = run_trial(controller, lambda pose: pose[None], torch.eye(3))
    assert not result["converged"] and result["updates"] == 100
    betas = [row["beta_deg"] for row in result["trace"]]
    assert max(map(abs, betas)) <= 20
