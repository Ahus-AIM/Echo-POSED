# Training, evaluation and guidance

Set `ECHOXFLOW_DATA_ROOT` to the extracted [EchoXFlow](https://github.com/Ahus-AIM/EchoXFlow) data. Each of the four members leaves out one of folds 0–3, and fold 4 is held out for evaluation. GPU runs need a CUDA 12.8 compatible driver.

```bash
uv run echo-posed train --device cuda --output runs/trained.pth
uv run echo-posed evaluate --weights runs/trained.pth --device cuda --output runs/held-out.json
```

## Guidance simulation

Pick a 3D recording and simulate guidance towards a target probe rotation within it.

```bash
uv run echo-posed list --stream 3d_brightness_mode
uv run echo-posed simulate RECORDING_ID --target-probe-deg 0 0 15 --output runs/guidance.json
```
