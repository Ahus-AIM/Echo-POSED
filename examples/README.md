# Reference volume

`reference_volume.npz` is one frame of one 3D B-mode recording from the [EchoXFlow dataset](https://huggingface.co/datasets/Ahus-AIM/EchoXFlow) by Ahus-AIM. It is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([license text](../LICENSE)).

| | |
| --- | --- |
| Exam | `exam_3f97719e5f5fa2eb` (patient fold 4, not used for training) |
| Recording | `recording_e2a4382b18086986` |
| Frame | 11 (zero-based) of 34, from `data/3d_brightness_mode` |
| Source | `exams/exam_3f97719e5f5fa2eb.tar` at dataset revision `6e2d58e5d465578306fd2d52882395705bbdadad` |

Changes: the single frame was extracted, rotated 180 degrees about the depth axis (elevation and azimuth reversed) to match the usual echo display, and stored as a compressed NumPy archive. Intensities are unmodified. The archive holds only the following arrays.

| Array | Content |
| --- | --- |
| `volume` | `uint8` beamspace intensities on (elevation, azimuth, depth) axes |
| `depth_start_m`, `depth_end_m` | Depth range in metres |
| `azimuth_width_rad`, `elevation_width_rad` | Sector widths in radians |

Cite the EchoXFlow paper when using this volume. The reference is in the [main README](../README.md#citation).
