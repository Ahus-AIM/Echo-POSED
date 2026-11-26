"""Relative pose corrections and a same-volume guidance simulation."""

import math

import torch
from PIL import Image

from .imaging import reslice
from .model import INPUT_SIZE, Ensemble
from .pose import angle, limit_rotation, local_angles, local_rotation


def correction(current: torch.Tensor, target: torch.Tensor):
    """Return orbital angle offsets and a right-multiplied probe correction."""
    offset = local_angles(target[..., 0, :, :]) - local_angles(current[..., 0, :, :])
    orbital = torch.atan2(offset.sin(), offset.cos())
    probe = current[..., 1, :, :].mT @ target[..., 1, :, :]
    return orbital, probe


def save_image(image: torch.Tensor, path):
    pixels = (image.squeeze().detach().cpu().clamp(0, 1) * 255).round().byte().numpy()
    Image.fromarray(pixels).save(path)


@torch.inference_mode()
def simulate(
    model: Ensemble,
    volume: torch.Tensor,
    geometry,
    initial: torch.Tensor,
    target: torch.Tensor,
    *,
    depth: float,
    width: float,
    steps: int,
    max_step_deg: float = 2.5,
    tolerance_deg: float = 1.0,
):
    device = model.adapters.device
    volume = volume.to(device)
    state = initial.to(device)
    target = target.to(device)

    def sample(pose):
        return reslice(
            volume, geometry, pose[None], size=INPUT_SIZE, depth=depth, width=width
        )

    target_image = sample(target)
    reference = model(target_image).rotations[0]
    trace = []
    for step in range(steps + 1):
        image = sample(state)
        prediction = model(image)
        orbital, probe = correction(prediction.rotations[0], reference)
        orbital_error = orbital.norm()
        probe_error = angle(probe)
        error_deg = torch.rad2deg(torch.maximum(orbital_error, probe_error)).item()
        trace.append(
            {
                "step": step,
                "predicted_error_deg": error_deg,
                "uncertainty_deg": torch.rad2deg(prediction.uncertainty_rad[0]).item(),
                "pose": state.tolist(),
                "true_factor_errors_deg": torch.rad2deg(
                    angle(state.mT @ target)
                ).tolist(),
            }
        )
        if error_deg <= tolerance_deg or step == steps:
            break
        limit = math.radians(max_step_deg)
        orbital_delta = orbital * (limit / orbital_error.clamp_min(limit))
        probe = limit_rotation(probe, limit)
        state = torch.stack(
            (local_rotation(local_angles(state[0]) + orbital_delta), state[1] @ probe)
        )
    return trace, image, target_image
