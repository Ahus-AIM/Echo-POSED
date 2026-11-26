"""Rotation algebra for the world-local and probe factors."""

import torch


TARGETS = ("world_local", "probe")


def project_so3(matrices: torch.Tensor) -> torch.Tensor:
    """Project a batch of matrices to proper rotations."""
    u, _, vh = torch.linalg.svd(matrices)
    sign = torch.linalg.det(u @ vh).sign()
    correction = torch.stack(
        (torch.ones_like(sign), torch.ones_like(sign), sign), dim=-1
    )
    return (u * correction.unsqueeze(-2)) @ vh


def angle(rotation: torch.Tensor) -> torch.Tensor:
    """Geodesic angle in radians, including near identity and half turns."""
    skew = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = skew.norm(dim=-1) / 2
    cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2
    return torch.atan2(sine, cosine)


def axis_rotation(radians: torch.Tensor, axis: int) -> torch.Tensor:
    """Build a right-handed rotation about x, y or z."""
    out = (
        torch.eye(3, device=radians.device, dtype=radians.dtype)
        .expand(*radians.shape, 3, 3)
        .clone()
    )
    i, j = (axis + 1) % 3, (axis + 2) % 3
    out[..., i, i] = out[..., j, j] = radians.cos()
    out[..., j, i] = radians.sin()
    out[..., i, j] = -out[..., j, i]
    return out


def local_angles(world_local: torch.Tensor) -> torch.Tensor:
    """Read orbital out-of-plane x and in-plane y angles."""
    return torch.stack(
        (
            torch.atan2(-world_local[..., 1, 2], world_local[..., 1, 1]),
            torch.atan2(-world_local[..., 2, 0], world_local[..., 0, 0]),
        ),
        dim=-1,
    )


def local_rotation(angles: torch.Tensor) -> torch.Tensor:
    return axis_rotation(angles[..., 1], 1) @ axis_rotation(angles[..., 0], 0)


def probe_rotation(xyz: torch.Tensor) -> torch.Tensor:
    return (
        axis_rotation(xyz[..., 2], 2)
        @ axis_rotation(xyz[..., 1], 1)
        @ axis_rotation(xyz[..., 0], 0)
    )


def render_pose(world_local: torch.Tensor, probe: torch.Tensor, depth_m: float):
    """Return the row-vector renderer rotation and translation in metres."""
    center = probe.new_tensor((0, 0, depth_m / 2)).unsqueeze(-2)
    world = probe @ world_local @ probe.mT
    return world_local @ probe.mT, (center - center @ world.mT) @ probe


def axis_angle_rotation(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation from a vector whose norm is the angle."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *x.shape, 3, 3
    )
    radians = vector.norm(dim=-1)[..., None, None]
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return (
        eye
        + torch.sinc(radians / torch.pi) * skew
        + 0.5 * torch.sinc(radians / (2 * torch.pi)).square() * (skew @ skew)
    )


def limit_rotation(rotation: torch.Tensor, cap: float) -> torch.Tensor:
    """Take a step of at most cap radians along the shortest rotation."""
    if angle(rotation) <= cap:
        return rotation
    _, vectors = torch.linalg.eigh((rotation + rotation.mT) / 2)
    axis = vectors[:, -1]
    skew = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    direction = torch.where(axis @ skew < 0, -axis, axis)
    return axis_angle_rotation(direction * cap)
