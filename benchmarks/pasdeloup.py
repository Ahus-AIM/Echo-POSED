"""Matched closed-loop guidance with Echo-POSED and Pasdeloup et al.'s cvcNet.

Both controllers move a virtual probe in the same EchoXFlow 3D frames with the
same two motions, P = P0 Rz(alpha) Rx(beta) with beta within 20 degrees: rotate,
then tilt, repeated, in steps of at most 3 degrees and 100 updates in total.

    uv run python -m benchmarks.pasdeloup trials.json --output runs/benchmark

The trial file lists the starting poses and the target cines:

    {"trials": [{
      "id": "example",
      "view": "A4C",
      "recording": "recording_...",
      "frames": [0],
      "start_probe_deg": [0, 0, 40],
      "targets": [{"recording": "recording_...", "stream": "2d_brightness_mode"}]
    }]}

``frames`` are the stitched 3D frames observed at every pose; the first one is
saved at the end. ``start_probe`` (a 3x3 matrix) can replace ``start_probe_deg``.
Poses are relative to the 3D volume as stored in EchoXFlow. Scoring the saved
end views, for example with a view classifier, is left to the user.
"""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from echo_posed.data import CINE_STREAM, cine_batches, load_volume, recording
from echo_posed.guidance import save_image
from echo_posed.imaging import reslice
from echo_posed.model import INPUT_SIZE, Ensemble
from echo_posed.pose import angle, axis_rotation, probe_rotation, project_so3
from external.pasdeloup.feedback import (
    CVCNetGuidance,
    native_angle,
    tilt_category,
    wrap_degrees,
)


TILT_BOUND_DEG = 20.0
MAX_STEP_DEG = 3.0
MIN_STEP_DEG = 0.09375
MAX_UPDATES = 100
PATIENCE = 20


def project_two_dof(relative, alpha=0.0, beta=0.0, bound=TILT_BOUND_DEG):
    """Closest Rz(alpha) Rx(beta) to a rotation, in degrees, with bounded tilt.

    Returns alpha, beta and the remaining geodesic distance. Ties, including flat
    objectives, go to the command closest to the current one.
    """
    x, y, z, w = Rotation.from_matrix(np.asarray(relative, dtype=np.float64)).as_quat()
    phase = np.rad2deg(np.arctan2(2 * (w * x + z * y), w * w + z * z - x * x - y * y))
    betas = np.array(
        (np.clip(phase, -bound, bound), -bound, bound, np.clip(beta, -bound, bound))
    )
    half = np.deg2rad(betas) / 2
    a = w * np.cos(half) + x * np.sin(half)
    b = z * np.cos(half) + y * np.sin(half)
    score = a * a + b * b
    alphas = np.rad2deg(2 * np.arctan2(b, a))
    travel = np.abs(wrap_degrees(alphas - alpha)) + np.abs(betas - beta)
    travel = np.where(score >= score.max() - 1e-12, travel, np.inf)
    best = int(np.argmin(travel))
    distance = np.rad2deg(2 * np.arccos(np.sqrt(np.clip(score[best], 0, 1))))
    return alpha + wrap_degrees(alphas[best] - alpha), betas[best], distance


class EchoPosedController:
    """Command the reachable pose closest to the predicted desired orientation."""

    tolerance = {"rotation": 5.0, "tilt": 5.0}
    stale_margin = {"rotation": 0.25, "tilt": 0.25}

    def __init__(self, predict: Callable, targets: list[np.ndarray]):
        """``predict`` maps images to [T,3,3] probe rotations; ``targets`` are cines."""
        self.predict = predict
        self.target = self.average(np.stack([self.average(cine) for cine in targets]))

    @staticmethod
    def average(rotations):
        return project_so3(
            torch.from_numpy(np.asarray(rotations, dtype=np.float64).mean(axis=0))
        ).numpy()

    def errors(self, images, pose, initial, alpha, beta):
        predicted = self.average(self.predict(images))
        desired = pose @ predicted.T @ self.target
        goal_alpha, goal_beta, residual = project_two_dof(
            initial.T @ desired, alpha, beta
        )
        error = {
            "rotation": float(wrap_degrees(goal_alpha - alpha)),
            "tilt": float(goal_beta - beta),
        }
        full = angle(torch.from_numpy(predicted.T @ self.target))
        return error, {
            "full_pose_error_deg": math.degrees(float(full)),
            "unreachable_residual_deg": float(residual),
        }

    def step(self, stage, error, cap):
        return float(np.clip(error, -cap, cap))


class CVCNetController:
    """Follow the rotation angle and tilt category differences of cvcNet."""

    tolerance = {"rotation": 5.0, "tilt": 0.5}
    stale_margin = {"rotation": 0.25, "tilt": 0.025}

    def __init__(self, predict: Callable, targets: list[np.ndarray]):
        """``predict`` maps images to [T,23] probabilities; ``targets`` are cines."""
        self.predict = predict
        self.target = self.average(np.stack([self.average(cine) for cine in targets]))

    @staticmethod
    def average(probabilities):
        return np.mean(probabilities, axis=0)

    def errors(self, images, pose, initial, alpha, beta):
        predicted = self.average(self.predict(images))
        current = float(native_angle(predicted[:12]))
        goal = float(native_angle(self.target[:12]))
        tilt = float(tilt_category(predicted[12:]))
        error = {
            "rotation": float(wrap_degrees(current - goal)),
            "tilt": float(tilt - tilt_category(self.target[12:])),
        }
        return error, {"rotation_angle_deg": current, "tilt_category": tilt}

    def step(self, stage, error, cap):
        # The probe turns against the display angle and against the tilt category.
        if stage == "rotation":
            return float(np.clip(wrap_degrees(-error), -cap, cap))
        return float(-np.sign(error) * cap)


def probe_pose(initial: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    z, x = (torch.deg2rad(initial.new_tensor(value)) for value in (alpha, beta))
    return initial @ axis_rotation(z, 2) @ axis_rotation(x, 0)


def run_trial(controller, observe: Callable, initial: torch.Tensor):
    """Alternate rotation and tilt blocks until both errors are within tolerance.

    ``observe`` renders the images seen at a [3,3] probe rotation.
    """
    position = {"rotation": 0.0, "tilt": 0.0}
    updates = 0
    trace = []
    while True:
        for stage in ("rotation", "tilt"):
            cap, previous, best, stale = MAX_STEP_DEG, 0.0, math.inf, 0
            while True:
                pose = probe_pose(initial, position["rotation"], position["tilt"])
                errors, diagnostics = controller.errors(
                    observe(pose),
                    pose.numpy().astype(np.float64),
                    initial.numpy().astype(np.float64),
                    position["rotation"],
                    position["tilt"],
                )
                error = errors[stage]
                sign = float(np.sign(error))
                if previous != 0 and sign != previous:
                    cap = max(MIN_STEP_DEG, cap / 2)
                previous = sign
                if best - abs(error) > controller.stale_margin[stage]:
                    best, stale = abs(error), 0
                else:
                    stale += 1
                if abs(error) <= controller.tolerance[stage]:
                    reason = "target_box"
                elif updates >= MAX_UPDATES:
                    reason = "max_updates"
                elif stale >= PATIENCE:
                    reason = "stagnation"
                else:
                    reason = None
                step = 0.0 if reason else controller.step(stage, error, cap)
                if stage == "tilt":
                    bounded = np.clip(
                        position[stage] + step, -TILT_BOUND_DEG, TILT_BOUND_DEG
                    )
                    step = float(bounded - position[stage])
                trace.append(
                    {
                        "stage": stage,
                        "updates": updates,
                        "alpha_deg": position["rotation"],
                        "beta_deg": position["tilt"],
                        "rotation_error": errors["rotation"],
                        "tilt_error": errors["tilt"],
                        "step_deg": step,
                        "stop_reason": reason,
                        **diagnostics,
                    }
                )
                if reason:
                    break
                position[stage] += step
                updates += 1
        converged = all(
            abs(errors[name]) <= controller.tolerance[name] for name in errors
        )
        if converged or updates >= MAX_UPDATES:
            return {
                "converged": converged,
                "updates": updates,
                "final_probe": pose.tolist(),
                "trace": trace,
            }


def load_cine(data_root, source, depth_cap):
    batches = cine_batches(
        recording(data_root, source["recording"]),
        source.get("stream", CINE_STREAM),
        batch_size=16,
        depth_cap=depth_cap,
    )
    return torch.cat([images for images, _ in batches])


def starting_pose(trial):
    if "start_probe" in trial:
        return torch.tensor(trial["start_probe"], dtype=torch.float32)
    return probe_rotation(
        torch.deg2rad(torch.tensor(trial["start_probe_deg"], dtype=torch.float32))
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trials", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path, default="checkpoints/echo-posed.pth")
    parser.add_argument("--depth-cap", type=float, default=0.16, help="Target cines")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    trials = json.loads(args.trials.read_text(encoding="utf-8"))["trials"]
    if len({trial["id"] for trial in trials}) != len(trials):
        raise ValueError("Trial IDs must be unique")
    ensemble = Ensemble.load(args.weights, device=args.device)
    cvcnet = CVCNetGuidance(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, trial in enumerate(trials):
        view = trial["view"]
        volumes, _, geometry = load_volume(
            recording(args.data_root, trial["recording"]), indices=trial["frames"]
        )

        def observe(pose, volumes=volumes, geometry=geometry):
            poses = torch.stack((torch.eye(3), pose)).expand(len(volumes), 2, 3, 3)
            return reslice(
                volumes,
                geometry,
                poses,
                size=INPUT_SIZE,
                depth=geometry.depth_end,
                width=geometry.azimuth_width,
            )

        cines = [
            load_cine(args.data_root, source, args.depth_cap)
            for source in trial["targets"]
        ]
        predictors = {
            "echo-posed": (
                EchoPosedController,
                lambda images: ensemble(images).rotations[:, 1].cpu().numpy(),
            ),
            "cvcnet": (
                CVCNetController,
                lambda images, view=view: cvcnet(images, view),
            ),
        }
        for name, (controller_type, predict) in predictors.items():
            targets = [
                np.concatenate([predict(batch) for batch in cine.split(32)])
                for cine in cines
            ]
            result = run_trial(
                controller_type(predict, targets), observe, starting_pose(trial)
            )
            stem = f"{number:05d}_{name}"
            final = torch.tensor(result["final_probe"], dtype=torch.float32)
            save_image(observe(final)[0], args.output / f"{stem}.png")
            (args.output / f"{stem}.json").write_text(
                json.dumps(result.pop("trace")) + "\n", encoding="utf-8"
            )
            results.append(
                {
                    "id": trial["id"],
                    "view": view,
                    "controller": name,
                    "files": stem,
                    **result,
                }
            )
            print(trial["id"], name, result["converged"], result["updates"], flush=True)
    (args.output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
