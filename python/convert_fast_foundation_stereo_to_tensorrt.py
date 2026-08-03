#!/usr/bin/env python3
"""Export Fast-FoundationStereo to its official single TensorRT engine.

The official exporter represents the group-wise correlation volume with the
FFSGWCVolume TensorRT plugin. This helper performs both official steps:

1. ``make_plugin_onnx.py`` exports one ONNX graph.
2. ``build_plugin_trt.py`` builds ``fast_foundationstereo.engine``.

Build the plugin before running this helper:

  cmake --build build --target xfeat_ffs_gwc_plugin

Example:

  python python/convert_fast_foundation_stereo_to_tensorrt.py \
      --model_dir thirdparty/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
      --save_path onnx_model/fast_foundation_stereo/ \
      --height 448 --width 640 --valid_iters 8 --max_disp 192

Outputs:

  onnx_model/fast_foundation_stereo/fast_foundationstereo_plugin.onnx
  onnx_model/fast_foundation_stereo/fast_foundationstereo.engine
  onnx_model/fast_foundation_stereo/onnx.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FFS_ROOT = REPO_ROOT / "thirdparty" / "Fast-FoundationStereo"
MAKE_PLUGIN_ONNX = FFS_ROOT / "scripts" / "make_plugin_onnx.py"
BUILD_PLUGIN_TRT = FFS_ROOT / "scripts" / "build_plugin_trt.py"
DEFAULT_ONNX_NAME = "fast_foundationstereo_plugin.onnx"
DEFAULT_ENGINE_NAME = "fast_foundationstereo.engine"


def run(command: list[str]) -> None:
    print(">>>", " ".join(str(item) for item in command))
    subprocess.run(command, check=True)


def find_plugin_library(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        candidates = [explicit_path]
    else:
        candidates = [
            REPO_ROOT / "build" / "libxfeat_ffs_gwc_plugin.so",
            REPO_ROOT / "build" / "lib" / "libxfeat_ffs_gwc_plugin.so",
            FFS_ROOT / "cpp" / "build" / "libffs_gwc_plugin.so",
            FFS_ROOT / "cpp" / "build" / "lib" / "libffs_gwc_plugin.so",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "FFSGWCVolume plugin library was not found. Build it with\n"
        "  cmake --build build --target xfeat_ffs_gwc_plugin\n"
        f"or pass --plugin-lib. Searched:\n  {searched}"
    )


def export_onnx(args: argparse.Namespace, save_path: Path) -> Path:
    if not MAKE_PLUGIN_ONNX.is_file():
        raise FileNotFoundError(
            f"Official single-engine exporter not found: {MAKE_PLUGIN_ONNX}. "
            "Update the Fast-FoundationStereo submodule."
        )
    onnx_path = save_path / args.onnx_name
    run(
        [
            sys.executable,
            str(MAKE_PLUGIN_ONNX),
            "--model_dir",
            args.model_dir,
            "--save_path",
            str(save_path),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--valid_iters",
            str(args.valid_iters),
            "--max_disp",
            str(args.max_disp),
            "--onnx_name",
            args.onnx_name,
        ]
    )
    return onnx_path


def build_engine(args: argparse.Namespace, onnx_path: Path, engine_path: Path, plugin_library: Path) -> None:
    if not BUILD_PLUGIN_TRT.is_file():
        raise FileNotFoundError(f"Official TensorRT builder not found: {BUILD_PLUGIN_TRT}")
    command = [
        sys.executable,
        str(BUILD_PLUGIN_TRT),
        str(onnx_path),
        str(engine_path),
        "--plugin_lib",
        str(plugin_library),
        "--workspace-mb",
        str(args.workspace_mb),
    ]
    if args.fp32:
        command.append("--fp32")
    run(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True, help="Path to model_best_bp2_serialize.pth")
    parser.add_argument(
        "--save_path",
        default="onnx_model/fast_foundation_stereo/",
        help="Directory for the ONNX graph, TensorRT engine, and YAML metadata",
    )
    parser.add_argument("--height", type=int, default=448, help="Engine input height (divisible by 32)")
    parser.add_argument("--width", type=int, default=640, help="Engine input width (divisible by 32)")
    parser.add_argument("--valid_iters", type=int, default=8, help="GRU refinement iterations")
    parser.add_argument("--max_disp", type=int, default=192, help="Maximum full-resolution disparity")
    parser.add_argument("--onnx-name", default=DEFAULT_ONNX_NAME, help="Plugin ONNX filename")
    parser.add_argument("--plugin-lib", type=Path, help="Path to libxfeat_ffs_gwc_plugin.so")
    parser.add_argument("--workspace-mb", type=int, default=4096, help="TensorRT workspace limit in MiB")
    parser.add_argument("--fp32", action="store_true", help="Disable the TensorRT FP16 builder flag")
    parser.add_argument("--skip-onnx", action="store_true", help="Build from an existing plugin ONNX graph")
    args = parser.parse_args()

    if args.height <= 0 or args.height % 32 != 0:
        parser.error("--height must be positive and divisible by 32")
    if args.width <= 0 or args.width % 32 != 0:
        parser.error("--width must be positive and divisible by 32")
    if args.max_disp <= 0 or args.max_disp % 4 != 0:
        parser.error("--max-disp must be positive and divisible by 4")
    if args.valid_iters <= 0:
        parser.error("--valid-iters must be positive")
    if args.workspace_mb <= 0:
        parser.error("--workspace-mb must be positive")
    if Path(args.onnx_name).name != args.onnx_name or not args.onnx_name.endswith(".onnx"):
        parser.error("--onnx-name must be a filename ending in .onnx")
    return args


def main() -> int:
    args = parse_args()
    save_path = Path(args.save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    plugin_library = find_plugin_library(args.plugin_lib)
    onnx_path = save_path / args.onnx_name
    if not args.skip_onnx:
        onnx_path = export_onnx(args, save_path)
    elif not onnx_path.is_file():
        raise FileNotFoundError(f"Plugin ONNX graph not found: {onnx_path}")

    engine_path = save_path / DEFAULT_ENGINE_NAME
    build_engine(args, onnx_path, engine_path, plugin_library)
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError(f"TensorRT engine was not created: {engine_path}")

    print(f"\nEngine: {engine_path}")
    print(f"Plugin: {plugin_library}")
    print("\nC++ usage:")
    print("  xfeat::FastFoundationStereoDepth::Params params;")
    print(f'  params.engine_path = "{engine_path}";')
    print(f"  params.max_disparity = {args.max_disp};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
