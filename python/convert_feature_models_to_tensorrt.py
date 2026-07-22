#!/usr/bin/env python3
"""Build TensorRT engines for XFeat, LighterGlue, and JIST ONNX models."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def resolve_trtexec(value: str | None) -> Path:
    candidates = [value, shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise SystemExit("trtexec was not found; pass --trtexec or add it to PATH")


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def engine_path(output_dir: Path, onnx_path: Path, precision: str) -> Path:
    return output_dir / f"{onnx_path.stem}_{precision}.engine"


def build_engine(
    trtexec: Path,
    onnx_path: Path,
    output_path: Path,
    precision: str,
    optimization_level: int,
    workspace_mib: int,
    extra_arguments: list[str],
) -> None:
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX model is not readable: {onnx_path}")
    command = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        "--skipInference",
        f"--builderOptimizationLevel={optimization_level}",
        f"--memPoolSize=workspace:{workspace_mib}M",
    ]
    if precision == "fp16":
        command.append("--fp16")
    command.extend(extra_arguments)
    print("\nBuilding", output_path)
    subprocess.run(command, check=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise SystemExit(f"TensorRT did not create a non-empty engine: {output_path}")


def dynamic_lighterglue_shapes(keypoints: int) -> str:
    return ",".join(
        [
            f"mkpts0:1x{keypoints}x2",
            f"feats0:1x{keypoints}x64",
            f"mkpts1:1x{keypoints}x2",
            f"feats1:1x{keypoints}x64",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xfeat", type=Path, help="fixed-shape XFeat ONNX model")
    parser.add_argument("--lighterglue", type=Path, help="dynamic-shape LighterGlue ONNX model")
    parser.add_argument("--jist", type=Path, help="fixed-shape JIST ONNX model")
    parser.add_argument("--output-dir", type=Path, default=Path("onnx_model"))
    parser.add_argument("--trtexec", help="path to the TensorRT trtexec executable")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--builder-optimization-level", type=int, choices=range(0, 6), default=3)
    parser.add_argument("--workspace-mib", type=positive, default=2048)
    parser.add_argument("--min-keypoints", type=positive, default=1)
    parser.add_argument("--opt-keypoints", type=positive, default=500)
    parser.add_argument("--max-keypoints", type=positive, default=1024)
    args = parser.parse_args()
    if args.xfeat is None and args.lighterglue is None and args.jist is None:
        parser.error("provide at least one of --xfeat, --lighterglue, or --jist")
    if not args.min_keypoints <= args.opt_keypoints <= args.max_keypoints:
        parser.error("keypoint profile sizes must satisfy min <= opt <= max")
    return args


def main() -> None:
    args = parse_args()
    trtexec = resolve_trtexec(args.trtexec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        trtexec=trtexec,
        precision=args.precision,
        optimization_level=args.builder_optimization_level,
        workspace_mib=args.workspace_mib,
    )

    if args.xfeat is not None:
        build_engine(
            onnx_path=args.xfeat.resolve(),
            output_path=engine_path(args.output_dir, args.xfeat, args.precision).resolve(),
            extra_arguments=[],
            **common,
        )
    if args.lighterglue is not None:
        build_engine(
            onnx_path=args.lighterglue.resolve(),
            output_path=engine_path(args.output_dir, args.lighterglue, args.precision).resolve(),
            extra_arguments=[
                f"--minShapes={dynamic_lighterglue_shapes(args.min_keypoints)}",
                f"--optShapes={dynamic_lighterglue_shapes(args.opt_keypoints)}",
                f"--maxShapes={dynamic_lighterglue_shapes(args.max_keypoints)}",
            ],
            **common,
        )
    if args.jist is not None:
        build_engine(
            onnx_path=args.jist.resolve(),
            output_path=engine_path(args.output_dir, args.jist, args.precision).resolve(),
            extra_arguments=[],
            **common,
        )


if __name__ == "__main__":
    main()
