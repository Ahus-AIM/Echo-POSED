"""Command-line workflows for Echo-POSED."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from .data import (
    CINE_STREAM,
    VOLUME_STREAM,
    cine_batches,
    load_volume,
    recording,
    recordings,
)
from .guidance import correction, save_image, simulate
from .imaging import reslice
from .learning import evaluate, train
from .model import INPUT_SIZE, Ensemble
from .pose import TARGETS, local_rotation, probe_rotation


def list_recordings(args):
    available = (
        item
        for item in recordings(args.data_root, content=args.stream, fold=args.fold)
        if item.path().is_dir()
    )
    rows = []
    for item in available:
        rows.append(
            {
                "recording_id": item.recording_id,
                "exam_id": item.exam_id,
                "fold": item.fold,
                "frames": item.frame_count(args.stream),
                "stitch_beat_count": item.stitch_beat_count,
            }
        )
        if len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("No matching extracted EchoXFlow recordings")
    print(json.dumps(rows, indent=2))


def predict(args):
    model = Ensemble.load(args.weights, device=args.device)
    source = recording(args.data_root, args.recording)
    frames = []
    for images, timestamps in cine_batches(
        source,
        args.stream,
        batch_size=args.batch_size,
        depth_cap=args.depth_cap,
    ):
        prediction = model(images)
        for timestamp, rotations, uncertainty in zip(
            timestamps,
            prediction.rotations.tolist(),
            torch.rad2deg(prediction.uncertainty_rad).tolist(),
            strict=True,
        ):
            frames.append(
                {
                    "timestamp_s": float(timestamp),
                    **dict(zip(TARGETS, rotations, strict=True)),
                    "uncertainty_deg": uncertainty,
                }
            )
    write_json(args.output, frames)


def guide(args):
    model = Ensemble.load(args.weights, device=args.device)
    images = []
    times = []
    for identifier, stream, index in (
        (args.current, args.current_stream, args.current_frame),
        (args.target, args.target_stream, args.target_frame),
    ):
        source = recording(args.data_root, identifier)
        for batch, timestamps in cine_batches(
            source,
            stream,
            batch_size=1,
            depth_cap=args.depth_cap,
            frame=index,
        ):
            images.append(batch)
            times.append(float(timestamps[0]))
    prediction = model(torch.cat(images))
    orbital, probe = correction(*prediction.rotations)
    write_json(
        args.output,
        {
            "current_timestamp_s": times[0],
            "target_timestamp_s": times[1],
            "orbital_correction_deg": torch.rad2deg(orbital).tolist(),
            "probe_correction": probe.tolist(),
            "uncertainty_deg": torch.rad2deg(prediction.uncertainty_rad).tolist(),
        },
    )


def write_json(output, value):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def pose(world, probe):
    world_rad = torch.deg2rad(torch.tensor(world, dtype=torch.float32))
    probe_rad = torch.deg2rad(torch.tensor(probe, dtype=torch.float32))
    return torch.stack((local_rotation(world_rad), probe_rotation(probe_rad)))


def render(args):
    source = recording(args.data_root, args.recording)
    volume, _, geometry = load_volume(source, indices=[args.frame])
    width = math.radians(args.sector_width)
    depth = min(args.depth_cap, geometry.depth_end)
    initial = pose(args.world_deg, args.probe_deg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "reslice":
        image = reslice(
            volume, geometry, initial[None], size=INPUT_SIZE, depth=depth, width=width
        )
        save_image(image, args.output)
        return
    target = pose(args.target_world_deg, args.target_probe_deg)
    model = Ensemble.load(args.weights, device=args.device)
    trace, final_image, target_image = simulate(
        model,
        volume,
        geometry,
        initial,
        target,
        depth=depth,
        width=width,
        steps=args.steps,
    )
    write_json(args.output, trace)
    save_image(final_image, args.output.with_suffix(".final.png"))
    save_image(target_image, args.output.with_suffix(".target.png"))


def fit(args):
    train(
        args.data_root,
        args.output,
        device=args.device,
        epochs=args.epochs,
        samples=args.samples,
        limit=args.limit,
        workers=args.workers,
        seed=args.seed,
        pretrained=not args.random_init,
        cache=args.cartesian_cache,
    )


def score(args):
    model = Ensemble.load(args.weights, device=args.device)
    metrics = evaluate(
        model,
        args.data_root,
        samples=args.samples,
        limit=args.limit,
        workers=args.workers,
        seed=args.seed,
    )
    write_json(args.output, metrics)
    print(json.dumps(metrics))


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return value


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    def command(name, handler, help_text):
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        sub.add_argument(
            "--data-root",
            type=Path,
            help="Extracted dataset root. Defaults to ECHOXFLOW_DATA_ROOT",
        )
        return sub

    def weights(sub):
        sub.add_argument(
            "--weights", type=Path, default=Path("checkpoints/echo-posed.pth")
        )
        sub.add_argument("--device", default="cpu")

    listing = command(
        "list", list_recordings, "List extracted 2D or 3D B-mode recordings"
    )
    listing.add_argument(
        "--stream", choices=(CINE_STREAM, VOLUME_STREAM), default=CINE_STREAM
    )
    listing.add_argument("--fold", type=int, choices=range(5))
    listing.add_argument("--limit", type=positive_int, default=10)

    inference = command(
        "predict", predict, "Predict both pose factors for a native 2D B-mode cine"
    )
    inference.add_argument("recording", help="EchoXFlow recording ID or .zarr path")
    inference.add_argument(
        "--stream", default=CINE_STREAM, help="B-mode stream, e.g. 2d_brightness_mode_1"
    )
    inference.add_argument("--batch-size", type=positive_int, default=16)
    inference.add_argument(
        "--depth-cap", type=positive_float, default=0.16, help="Maximum depth in metres"
    )
    inference.add_argument("--output", type=Path, required=True)
    weights(inference)

    guidance = command(
        "guide", guide, "Compute the correction from a current 2D frame to a target"
    )
    for kind in ("current", "target"):
        guidance.add_argument(kind, help="EchoXFlow recording ID or .zarr path")
        guidance.add_argument(f"--{kind}-stream", default=CINE_STREAM)
        guidance.add_argument(f"--{kind}-frame", type=nonnegative_int, default=0)
    guidance.add_argument(
        "--depth-cap", type=positive_float, default=0.16, help="Maximum depth in metres"
    )
    guidance.add_argument("--output", type=Path, required=True)
    weights(guidance)

    for name, help_text in (
        ("reslice", "Render a virtual 2D sector from an EchoXFlow 3D cine"),
        ("simulate", "Run closed-loop guidance between two poses in one 3D frame"),
    ):
        sub = command(name, render, help_text)
        sub.add_argument("recording", help="EchoXFlow 3D recording ID or .zarr path")
        sub.add_argument(
            "--frame",
            type=nonnegative_int,
            default=0,
            help="Frame index after beat stitching",
        )
        sub.add_argument(
            "--probe-deg",
            type=float,
            nargs=3,
            default=(0, 0, 0),
            metavar=("X", "Y", "Z"),
        )
        sub.add_argument(
            "--world-deg", type=float, nargs=2, default=(0, 0), metavar=("X", "Y")
        )
        sub.add_argument(
            "--depth-cap",
            type=positive_float,
            default=0.16,
            help="Maximum depth in metres",
        )
        sub.add_argument(
            "--sector-width",
            type=positive_float,
            default=60,
            help="Virtual sector width in degrees",
        )
        sub.add_argument("--output", type=Path, required=True)
        if name == "simulate":
            sub.add_argument(
                "--target-probe-deg",
                type=float,
                nargs=3,
                required=True,
                metavar=("X", "Y", "Z"),
            )
            sub.add_argument(
                "--target-world-deg",
                type=float,
                nargs=2,
                default=(0, 0),
                metavar=("X", "Y"),
            )
            sub.add_argument("--steps", type=nonnegative_int, default=20)
            weights(sub)

    for name, handler, help_text in (
        ("train", fit, "Train a four-member ensemble on development folds"),
        ("evaluate", score, "Measure pose error and uncertainty on held-out fold 4"),
    ):
        sub = command(name, handler, help_text)
        sub.add_argument(
            "--samples",
            type=positive_int,
            default=256 if name == "train" else 32,
            help="Virtual slices per recording and pass"
            + (", mirrored copies included" if name == "train" else ""),
        )
        sub.add_argument(
            "--limit-per-fold",
            dest="limit",
            type=positive_int,
            help="Cap on extracted recordings for a smoke run",
        )
        sub.add_argument(
            "--workers",
            type=nonnegative_int,
            default=16 if name == "train" else 0,
            help="Parallel dataset workers",
        )
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--output", type=Path, required=True)
        if name == "train":
            sub.add_argument("--epochs", type=positive_int, default=20)
            sub.add_argument(
                "--cartesian-cache",
                type=Path,
                default=Path("runs/cartesian"),
                help="Cartesian copies of the training volumes, written if missing",
            )
            sub.add_argument(
                "--random-init",
                action="store_true",
                help="Skip the ImageNet download for smoke runs",
            )
            sub.add_argument(
                "--device", default="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            weights(sub)

    return root


def main():
    cli = parser()
    args = cli.parse_args()
    if args.command in ("train", "evaluate") and (
        args.samples < 2 or (args.command == "train" and args.samples % 2)
    ):
        cli.error("--samples must be at least two and even for training")
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"echo-posed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
