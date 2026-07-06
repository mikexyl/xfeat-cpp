#!/usr/bin/env python3
"""
Convert Fast-FoundationStereo PyTorch checkpoint to two TensorRT engines.

Steps performed:
  1. Export feature_runner.onnx and post_runner.onnx via make_onnx.py
  2. Convert both ONNX files to TRT engines via trtexec (--fp16)

Usage:
  python python/convert_fast_foundation_stereo_to_tensorrt.py \\
      --model_dir thirdparty/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \\
      --save_path onnx_model/fast_foundation_stereo/ \\
      --height 448 --width 640 --valid_iters 8 --max_disp 192

Outputs:
  onnx_model/fast_foundation_stereo/feature_runner.onnx
  onnx_model/fast_foundation_stereo/feature_runner.engine
  onnx_model/fast_foundation_stereo/post_runner.onnx
  onnx_model/fast_foundation_stereo/post_runner.engine
  onnx_model/fast_foundation_stereo/onnx.yaml   (model config)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAKE_ONNX = REPO_ROOT / "thirdparty" / "Fast-FoundationStereo" / "scripts" / "make_onnx.py"
TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")


def run(cmd, **kwargs):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def export_onnx(args):
    """Run make_onnx.py from the Fast-FoundationStereo submodule."""
    cmd = [
        sys.executable, str(MAKE_ONNX),
        "--model_dir", args.model_dir,
        "--save_path", args.save_path,
        "--height", str(args.height),
        "--width", str(args.width),
        "--valid_iters", str(args.valid_iters),
        "--max_disp", str(args.max_disp),
    ]
    run(cmd)


def convert_to_trt(onnx_path: Path, engine_path: Path):
    """Convert ONNX to TensorRT engine using the Python API (avoids trtexec CUDA version issues)."""
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError(
            "tensorrt Python package not found. Install with: pip install tensorrt-cu12"
        )

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building TensorRT engine (fp16) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for: {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    if not engine_path.exists():
        raise RuntimeError(f"Engine not created: {engine_path}")
    print(f"Engine saved: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True,
                        help="Path to model_best_bp2_serialize.pth checkpoint")
    parser.add_argument("--save_path", default="onnx_model/fast_foundation_stereo/",
                        help="Directory for ONNX and engine outputs")
    parser.add_argument("--height", type=int, default=448,
                        help="Input height (must be divisible by 32)")
    parser.add_argument("--width", type=int, default=640,
                        help="Input width (must be divisible by 32)")
    parser.add_argument("--valid_iters", type=int, default=8,
                        help="GRU refinement iterations (fewer = faster, less accurate)")
    parser.add_argument("--max_disp", type=int, default=192,
                        help="Maximum disparity for GWC volume encoding")
    parser.add_argument("--skip_onnx", action="store_true",
                        help="Skip ONNX export (use existing .onnx files in save_path)")
    args = parser.parse_args()

    assert args.height % 32 == 0, "--height must be divisible by 32"
    assert args.width % 32 == 0, "--width must be divisible by 32"

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Step 1: ONNX export
    if not args.skip_onnx:
        print("=" * 60)
        print("Step 1: Exporting ONNX models")
        print("=" * 60)
        export_onnx(args)
    else:
        print("Skipping ONNX export (--skip_onnx)")

    # Step 2: TRT conversion
    print("=" * 60)
    print("Step 2: Converting to TensorRT engines (--fp16)")
    print("=" * 60)

    for name in ("feature_runner", "post_runner"):
        onnx = save_path / f"{name}.onnx"
        engine = save_path / f"{name}.engine"
        if not onnx.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx}")
        convert_to_trt(onnx, engine)

    print("\n" + "=" * 60)
    print("Done! Engines saved to:", save_path)
    print("=" * 60)
    print("\nC++ usage (Params):")
    print(f'  feature_engine_path = "{save_path}/feature_runner.engine"')
    print(f'  post_engine_path    = "{save_path}/post_runner.engine"')
    print(f'  target_size         = cv::Size({args.width}, {args.height})')
    print(f'  max_disp            = {args.max_disp}')
    print("Check onnx.yaml for cv_group and normalize flags.")


if __name__ == "__main__":
    main()
