# Pasdeloup cvcNet weights

These four PyTorch checkpoints contain the pretrained cvcNet weights released by David Pasdeloup and collaborators. The port converts the published Keras HDF5 files to PyTorch state dictionaries.

> [!NOTE]
> - [Original repository](https://github.com/dfpasdel/retrospective_apical_views_standardization)
> - [Original checkpoints at the pinned revision](https://github.com/dfpasdel/retrospective_apical_views_standardization/tree/f8ecf60529d56c1df9ec96db3bf7c683ed33fa7a/guiding/inference/models)

## License

- **Weights.** The converted weights retain [CC BY-NC 4.0](LICENSE_MODELS.txt). If you share them, credit the original creators, retain the license and notices, and identify the HDF5-to-PyTorch conversion.
- **Code.** The PyTorch adaptation of the model code and the guidance coordinates in `feedback.py` are covered by [PolyForm Noncommercial 1.0.0](LICENSE_CODE.txt).

[Upstream license notices](https://github.com/dfpasdel/retrospective_apical_views_standardization#license)

## Matched guidance benchmark

[`benchmarks/pasdeloup.py`](../../benchmarks/pasdeloup.py) moves a virtual probe in EchoXFlow 3D frames with Echo-POSED and with these models, using the same starting poses and motions. Its docstring describes the trial file.

```bash
uv run python -m benchmarks.pasdeloup trials.json --output runs/benchmark
```

## Required citations

Cite both papers when using these models, as requested by the upstream authors.

```bibtex
@article{pasdeloup2023guidance,
  author  = {Pasdeloup, David and Olaisen, Sindre H. and {\O}stvik, Andreas
             and Sabo, Sigbjorn and Pettersen, H{\aa}kon N. and Holte, Espen
             and Grenne, Bj{\o}rnar and St{\o}len, Stian B. and Smistad, Erik
             and Aase, Svein Arne and Dalen, H{\aa}vard and L{\o}vstakken, Lasse},
  title   = {Real-Time Echocardiography Guidance for Optimized Apical Standard Views},
  journal = {Ultrasound in Medicine \& Biology},
  year    = {2023},
  volume  = {49},
  number  = {1},
  pages   = {333--346},
  doi     = {10.1016/j.ultrasmedbio.2022.09.006}
}

@article{sabo2023guidance,
  author  = {Sabo, Sigbjorn and Pasdeloup, David and Pettersen, Hakon Neergaard
             and Smistad, Erik and {\O}stvik, Andreas and Olaisen, Sindre Hellum
             and St{\o}len, Stian Bergseng and Grenne, Bj{\o}rnar Leangen
             and Holte, Espen and Lovstakken, Lasse and Dalen, Havard},
  title   = {Real-time guidance by deep learning of experienced operators to improve
             the standardization of echocardiographic acquisitions},
  journal = {European Heart Journal -- Imaging Methods and Practice},
  year    = {2023},
  volume  = {1},
  number  = {2},
  pages   = {qyad040},
  doi     = {10.1093/ehjimp/qyad040}
}
```

## PyTorch inference

From the repository root, load the rotation checkpoint and pass it a batch of prepared frames.

```python
import torch
from external.pasdeloup.cvcnet import load_checkpoint

model = load_checkpoint("external/pasdeloup/cvcnet_rotation.pt")
with torch.inference_mode():
    probabilities = model(frames)
```

Start with the sector tip at the top, then transpose each grayscale frame (`frame.T`) before resizing. Pass float32 frames of shape `[N, 1, 128, 128]` scaled to `[0, 1]`. The model applies the Gaussian blur. The rotation checkpoint returns 12 probabilities and each tilt checkpoint returns 11. [Upstream frame loader](https://github.com/dfpasdel/retrospective_apical_views_standardization/blob/f8ecf60529d56c1df9ec96db3bf7c683ed33fa7a/guiding/utilities/loader_functions.py)
