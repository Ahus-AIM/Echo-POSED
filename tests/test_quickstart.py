from pathlib import Path

import numpy as np
import torch
from PIL import Image

from echo_posed.data import load_reference_volume
from echo_posed.imaging import VolumeGeometry, reslice, scan_convert, slice_volume
from echo_posed.plotting import save_rotation_grid, transparent_sectors, volume_support

REFERENCE = Path(__file__).parents[1] / "examples" / "reference_volume.npz"


def test_reference_volume_holds_one_frame_and_its_geometry():
    with np.load(REFERENCE, allow_pickle=False) as sample:
        assert sorted(sample.files) == [
            "azimuth_width_rad",
            "depth_end_m",
            "depth_start_m",
            "elevation_width_rad",
            "volume",
        ]
        assert sample["volume"].dtype == np.uint8
    volume, geometry = load_reference_volume(REFERENCE)
    assert volume.shape == (60, 110, 501)
    assert 0 <= volume.min() < volume.max() <= 1
    assert geometry == VolumeGeometry(0, 0.15, 1.047198, 1.047198)


def test_slicing_then_scan_conversion_matches_reslice():
    volumes = torch.rand(2, 9, 11, 13)
    geometry = VolumeGeometry(0, 0.12, 1, 1)
    poses = torch.eye(3).expand(2, 2, 3, 3)
    options = {"size": 16, "width": 0.9, "depth": 0.1}
    sectors = slice_volume(volumes, geometry, poses, **options)
    assert sectors.shape == (2, 16, 16)
    images = scan_convert(sectors, width=0.9, depth_start=0, depth_end=0.1, size=16)
    torch.testing.assert_close(images, reslice(volumes, geometry, poses, **options))


def test_transparency_follows_geometry_not_intensity():
    geometry = VolumeGeometry(0, 0.12, 1, 1)
    poses = torch.eye(3).repeat(1, 2, 1, 1)
    options = {"width": 0.8, "depth": 0.1}
    support = volume_support(geometry, poses, size=32, **options)
    assert support.bool().all()
    # A black frame stays opaque inside the sector and transparent outside it.
    image = transparent_sectors(torch.zeros(1, 32, 32), support, size=64, **options)[0]
    alpha = image[..., 1]
    assert alpha[32, 32] > 0.999 and alpha[0, 0] == 0 and alpha[-1, 0] == 0
    assert ((alpha > 0) & (alpha < 1)).any()
    # A white frame keeps its intensity in partly covered edge pixels.
    white = transparent_sectors(torch.ones(1, 32, 32), support, size=64, **options)[0]
    assert torch.allclose(white[..., 0][white[..., 1] > 0], torch.tensor(1.0))


def test_samples_beyond_the_acquired_volume_are_unsupported():
    geometry = VolumeGeometry(0, 0.12, 0.6, 1)
    poses = torch.eye(3).repeat(1, 2, 1, 1)
    support = volume_support(geometry, poses, size=33, width=1.0, depth=0.1)[0]
    assert support[20, 16] == 1 and support[20, 0] == 0 and support[20, -1] == 0


def test_rotation_grid_is_saved_with_a_transparent_background(tmp_path):
    geometry = VolumeGeometry(0, 0.12, 1, 1)
    poses = torch.eye(3).repeat(2, 2, 1, 1)
    output = tmp_path / "nested" / "grid.png"
    save_rotation_grid(
        output,
        torch.rand(2, 16, 16),
        poses[:, 1],
        ["a", "b"],
        geometry=geometry,
        poses=poses,
        width=0.9,
        depth=0.1,
        columns=2,
    )
    with Image.open(output) as figure:
        assert figure.mode == "RGBA" and figure.getpixel((0, 0))[3] == 0
