"""PyTorch cvcNet, adapted from dfpasdel/retrospective_apical_views_standardization."""

import argparse
import hashlib
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM = "https://github.com/dfpasdel/retrospective_apical_views_standardization"
REVISION = "f8ecf60529d56c1df9ec96db3bf7c683ed33fa7a"
MODEL_LICENSE = "CC-BY-NC-4.0"
WEIGHT_SHA256 = {
    "rotation": "5d138fabe6bcba4419d9893338213c3e2592532627baf10947d7404bdb73f756",
    "tilt_a2c": "248c45392bfdde40706d162eef98a867818f3f620e62bb5f88fcf1c76f8c8287",
    "tilt_a4c": "d07b280b1a7cf5dbdfa2f25cea3e788cbc7c1335f46340a136e921996e638613",
    "tilt_alax": "bd7a17d61e1b8c5b2c78ed8bab28a2540d62f43a88f54370cdc78dbdecca3d51",
}
ROTATION_LABELS = (
    "a4c", "a42c", "a2c", "ans", "alax", "ane",
    "a4cf", "a42cf", "a2cf", "ansf", "alaxf", "anef",
)


class SpatialPReLU(nn.Module):
    """Keras PReLU without shared axes learns one slope per channel and pixel."""

    def __init__(self, channels, size):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels, size, size))

    def forward(self, x):
        return x.relu() - self.alpha * (-x).relu()


def conv_block(in_channels, channels, kernel, size):
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel, padding=kernel // 2),
        nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
        SpatialPReLU(channels, size),
    )


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, filters, size, *, reduced=False):
        super().__init__()
        if reduced:
            self.branches = nn.ModuleList([
                conv_block(in_channels, filters, 1, size),
                conv_block(in_channels, filters, 3, size),
            ])
        else:
            three_quarters = 3 * filters // 4
            half = filters // 2
            self.branches = nn.ModuleList([
                conv_block(in_channels, three_quarters, 1, size),
                nn.Sequential(
                    conv_block(in_channels, three_quarters, 1, size),
                    conv_block(three_quarters, filters, 3, size),
                ),
                nn.Sequential(
                    conv_block(in_channels, half, 1, size),
                    conv_block(half, three_quarters, 5, size),
                ),
            ])

    def forward(self, x):
        return torch.cat([x, *(branch(x) for branch in self.branches)], dim=1)


class CVCNet(nn.Module):
    """Accept float NCHW inputs of shape (N, 1, 128, 128), scaled to [0, 1]."""

    def __init__(self, variant):
        super().__init__()
        if variant not in WEIGHT_SHA256:
            raise ValueError(f"Unknown cvcNet variant {variant!r}")
        self.variant = variant
        self.labels = ROTATION_LABELS if variant == "rotation" else (
            "5-", "4-", "3-", "2-", "1-", variant.removeprefix("tilt_"),
            "1+", "2+", "3+", "4+", "5+",
        )

        layers = [
            conv_block(1, 16, 3, 128),
            nn.MaxPool2d(2),
            conv_block(16, 32, 3, 64),
            nn.MaxPool2d(2),
        ]
        in_channels, size = 32, 32
        for filters in (64, 128, 256):
            layers.extend([
                InceptionBlock(in_channels, filters, size),
                conv_block(in_channels + 5 * filters // 2, filters, 1, size),
                nn.MaxPool2d(2),
                nn.Dropout(0.35),
            ])
            in_channels, size = filters, size // 2

        for _ in range(2):
            layers.extend([
                InceptionBlock(in_channels, 512, size, reduced=True),
                nn.Dropout(0.35),
            ])
            in_channels += 1024
        layers.extend([
            nn.Conv2d(in_channels, len(self.labels), 1),
            SpatialPReLU(len(self.labels), size),
        ])
        self.network = nn.Sequential(*layers)

        kernel = torch.exp(-0.5 * torch.arange(-1, 2, dtype=torch.float32).square())
        kernel /= kernel.sum()
        self.register_buffer("gaussian", torch.outer(kernel, kernel)[None, None], persistent=False)

    def blur(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.gaussian)

    def logits(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, 128, 128):
            raise ValueError(f"Expected (N, 1, 128, 128), got {tuple(x.shape)}")
        return self.network(self.blur(x)).mean(dim=(2, 3))

    def forward(self, x):
        return self.logits(x).softmax(dim=1)


def load_checkpoint(path, *, device="cpu"):
    """Load a converted state dict in evaluation mode on the requested device."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "cvcnet-torch-v1":
        raise ValueError(f"Unsupported cvcNet checkpoint format in {path}")
    model = CVCNet(checkpoint["variant"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def convert_weights(source, variant):
    """Load and verify a published Keras HDF5 file using h5py."""
    import h5py

    with Path(source).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != WEIGHT_SHA256[variant]:
        raise ValueError(f"{source}: SHA-256 does not match the published {variant} weights")

    model = CVCNet(variant)
    with h5py.File(source) as h5:
        weights = h5["cvcnet"]
        layer_names = list(weights)
        used = set()

        def copy(layer, name, parameter, axes=None):
            key = f"{layer}/{name}:0"
            value = torch.from_numpy(weights[key][...])
            if axes is not None:
                value = value.permute(axes)
            if parameter.shape != value.shape:
                raise ValueError(f"{key}: expected {tuple(parameter.shape)}, got {tuple(value.shape)}")
            parameter.copy_(value)
            used.add(key)

        # Keras saves branches in topological order. Name suffixes retain
        # construction order even when the first suffix differs by checkpoint.
        families = (
            ("conv2d", nn.Conv2d),
            ("batch_normalization", nn.BatchNorm2d),
            ("p_re_lu", SpatialPReLU),
        )
        for prefix, module_type in families:
            names = sorted(
                (name for name in layer_names if name.startswith(prefix + "_")),
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )
            modules = (module for module in model.modules() if isinstance(module, module_type))
            for layer, module in zip(names, modules, strict=True):
                if module_type is nn.Conv2d:
                    copy(layer, "kernel", module.weight, (3, 2, 0, 1))
                    copy(layer, "bias", module.bias)
                elif module_type is nn.BatchNorm2d:
                    copy(layer, "gamma", module.weight)
                    copy(layer, "beta", module.bias)
                    copy(layer, "moving_mean", module.running_mean)
                    copy(layer, "moving_variance", module.running_var)
                else:
                    copy(layer, "alpha", module.alpha, (2, 0, 1))
        expected = {name.decode() for name in weights.attrs["weight_names"]}
        if used != expected:
            raise ValueError(f"{source}: checkpoint tensors did not map one to one")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description="Convert the four released cvcNet checkpoints")
    parser.add_argument("model_dir", type=Path, help="upstream guiding/inference/models directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for variant, digest in WEIGHT_SHA256.items():
        model = convert_weights(args.model_dir / f"cvcnet_{variant}_model.hdf5", variant)
        checkpoint = {
            "format": "cvcnet-torch-v1",
            "variant": variant,
            "upstream": UPSTREAM,
            "upstream_revision": REVISION,
            "source_sha256": digest,
            "source_license": MODEL_LICENSE,
            "state_dict": model.state_dict(),
        }
        path = args.out / f"cvcnet_{variant}.pt"
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            try:
                torch.save(checkpoint, stream)
                stream.close()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
        print(path, flush=True)


if __name__ == "__main__":
    main()
