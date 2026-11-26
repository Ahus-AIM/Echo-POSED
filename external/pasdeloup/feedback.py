"""cvcNet inference and guidance coordinates for the matched simulator.

Display angles are transcribed from the authors' real-time application (commit
6a8875fb98810dfc8e3fdd14a180f6e8783eb93e of
dfpasdel/retrospective_apical_views_standardization). That application shows cues
to an operator. Applying them to a simulated probe is an adaptation made here.
Licensed under PolyForm Noncommercial 1.0.0, see LICENSE_CODE.txt.
"""

from pathlib import Path

import numpy as np
import torch
from skimage.transform import resize

from .cvcnet import load_checkpoint


TILT_MODELS = {"A4C": "tilt_a4c", "A2C": "tilt_a2c", "A3C": "tilt_alax"}
DISPLAY_ANGLES = np.array(
    [0.0, 322.5, 285.0, 261.0, 226.0, 199.0, 180.0, 142.0, 105.0, 81.0, 46.0, 19.0]
)


def wrap_degrees(value):
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def native_angle(probabilities):
    """Interpolate the display angle of the circular mean of 12 rotation classes."""
    p = np.asarray(probabilities)
    angles = np.arange(12) * 2 * np.pi / 12
    category = (
        np.arctan2(p @ np.sin(angles), p @ np.cos(angles)) % (2 * np.pi) * 12
    ) / (2 * np.pi)
    left = np.floor(category).astype(int) % 12
    fraction = category - np.floor(category)
    ends = np.deg2rad(DISPLAY_ANGLES[[left, (left + 1) % 12]])
    x = (1 - fraction) * np.cos(ends[0]) + fraction * np.cos(ends[1])
    y = (1 - fraction) * np.sin(ends[0]) + fraction * np.sin(ends[1])
    return np.rad2deg(np.arctan2(y, x))


def tilt_category(probabilities):
    """Expected category of the 11 view-specific tilt classes."""
    return probabilities @ np.arange(11)


class CVCNetGuidance:
    """Rotation and view-specific tilt probabilities for scan-converted frames."""

    def __init__(self, device="cpu"):
        root = Path(__file__).parent
        self.device = torch.device(device)
        self.models = {
            name: load_checkpoint(root / f"cvcnet_{name}.pt", device=self.device)
            for name in ("rotation", *TILT_MODELS.values())
        }

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor, view: str) -> np.ndarray:
        """Map [T,1,H,W] images with the sector tip at the top to [T,23]."""
        frames = images[:, 0].cpu().numpy()
        prepared = np.stack(
            [resize(frame.T[..., None], (128, 128, 1)) for frame in frames]
        ).astype(np.float32)
        x = torch.from_numpy(prepared.transpose(0, 3, 1, 2).copy()).to(self.device)
        outputs = (self.models["rotation"](x), self.models[TILT_MODELS[view]](x))
        return torch.cat(outputs, dim=1).cpu().numpy()
