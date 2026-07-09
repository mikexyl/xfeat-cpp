#!/usr/bin/env python3
"""Export official Depth Anything 3 real multi-view forward to ONNX and TensorRT.

This helper targets the DA3 any-view API path:
  images -> DepthAnything3.forward(images, extrinsics=None, intrinsics=None, ...)

It can also export the pose-conditioned path:
  images, extrinsics, intrinsics -> DepthAnything3.forward(...)

The exported input is fixed shape [1, V, 3, H, W]. By default this creates the
first small fixed-view engine for the C++ mono-depth runtime:
  V=3, H=350, W=504, model=depth-anything/DA3-SMALL, FP16 TensorRT.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


DEFAULT_ARTIFACT_DIR = Path("onnx_model/mono_depth/depth_anything_v3")
DEFAULT_ONNX = DEFAULT_ARTIFACT_DIR / "DA3-SMALL_multiview_v3_350x504.onnx"
DEFAULT_ENGINE = DEFAULT_ARTIFACT_DIR / "DA3-SMALL_multiview_v3_350x504_fp16.engine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--da3-repo",
        default=os.environ.get("DEPTH_ANYTHING_V3_REPO", "/tmp/depth-anything-3"),
        help="Path to the upstream Depth-Anything-3 checkout",
    )
    parser.add_argument(
        "--clone-url",
        default="https://github.com/ByteDance-Seed/Depth-Anything-3.git",
        help="Git URL used when --da3-repo does not exist",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Fail instead of cloning the upstream DA3 repository when --da3-repo is missing",
    )
    parser.add_argument("--model", default="depth-anything/DA3-SMALL", help="Hugging Face model id or local directory")
    parser.add_argument("--views", type=int, default=3, help="Fixed number of input views")
    parser.add_argument("--height", type=int, default=350, help="Fixed input height")
    parser.add_argument("--width", type=int, default=504, help="Fixed input width")
    parser.add_argument(
        "--camera-inputs",
        action="store_true",
        help="Expose extrinsics [1,V,4,4] and intrinsics [1,V,3,3] as ONNX/TensorRT inputs",
    )
    parser.add_argument(
        "--normalize-camera-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize input extrinsics inside the exported graph, matching DepthAnything3.inference()",
    )
    parser.add_argument(
        "--ref-view-strategy",
        default="middle",
        help="Fixed DA3 reference view strategy baked into the traced forward graph",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Recorded for parity with DA3 inference preprocessing; forward export uses --height/--width",
    )
    parser.add_argument(
        "--hf-home",
        default=str(DEFAULT_ARTIFACT_DIR / "hf_cache"),
        help="HF_HOME cache directory to use before loading the model",
    )
    parser.add_argument("--device", default="cuda", help="Torch device used for loading and ONNX export")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Output ONNX model path")
    parser.add_argument(
        "--engine",
        "--output",
        dest="engine",
        default=str(DEFAULT_ENGINE),
        help="Output TensorRT engine path",
    )
    parser.add_argument("--workspace-gb", type=float, default=4.0, help="TensorRT workspace limit in GiB")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable TensorRT FP16 precision")
    parser.add_argument("--fp32", action="store_true", help="Disable TensorRT FP16 precision")
    parser.add_argument(
        "--force-fp16-autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch DA3's CUDA autocast selection to avoid BF16 during export",
    )
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export and build from an existing file")
    parser.add_argument("--skip-engine", action="store_true", help="Export ONNX only; do not build TensorRT")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_da3_repo(args: argparse.Namespace) -> Path:
    repo = Path(args.da3_repo).expanduser()
    src = repo / "src" / "depth_anything_3"
    if src.is_dir():
        return repo

    if args.no_clone:
        raise RuntimeError(f"Depth Anything 3 source directory not found: {src}")

    repo.parent.mkdir(parents=True, exist_ok=True)
    if repo.exists():
        raise RuntimeError(f"{repo} exists but does not look like a Depth-Anything-3 checkout")

    run_command(["git", "clone", "--depth", "1", args.clone_url, str(repo)])
    if not src.is_dir():
        raise RuntimeError(f"Depth Anything 3 clone did not create expected source directory: {src}")
    return repo


def add_da3_to_path(repo: Path) -> None:
    src = repo / "src"
    if not src.is_dir():
        raise RuntimeError(f"Depth Anything 3 source directory not found: {src}")
    sys.path.insert(0, str(src))


def get_raw_output(raw: Any, key: str) -> Any:
    if hasattr(raw, "get"):
        value = raw.get(key, None)
    else:
        value = getattr(raw, key, None)
    if value is None:
        raise RuntimeError(f"DA3 forward output does not contain {key!r}")
    return value


def select_output_specs(raw: Any) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    candidates = [
        ("depth", ("depth",)),
        ("depth_conf", ("depth_conf", "conf")),
        ("extrinsics", ("extrinsics",)),
        ("intrinsics", ("intrinsics",)),
    ]
    for output_name, keys in candidates:
        for key in keys:
            try:
                get_raw_output(raw, key)
            except RuntimeError:
                continue
            specs.append((output_name, key))
            break

    if not specs or specs[0][0] != "depth":
        raise RuntimeError("DA3 forward did not produce a depth tensor")
    return specs


def format_shape(tensor: Any) -> str:
    return "[" + ", ".join(str(dim) for dim in tensor.shape) + "]"


@contextlib.contextmanager
def force_fp16_autocast(torch_module: Any, enabled: bool):
    if not enabled or not hasattr(torch_module, "cuda") or not hasattr(torch_module.cuda, "is_bf16_supported"):
        yield
        return

    original = torch_module.cuda.is_bf16_supported
    torch_module.cuda.is_bf16_supported = lambda *args, **kwargs: False
    try:
        yield
    finally:
        torch_module.cuda.is_bf16_supported = original


def patch_da3_for_onnx_export(torch_module: Any) -> None:
    from depth_anything_3.model.dinov2.layers import rope

    def onnx_friendly_position_getter(self: Any, batch_size: int, height: int, width: int, device: Any) -> Any:
        if (height, width) not in self.position_cache:
            y_coords = torch_module.arange(height, device=device)
            x_coords = torch_module.arange(width, device=device)
            yy = y_coords[:, None].expand(height, width)
            xx = x_coords[None, :].expand(height, width)
            positions = torch_module.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    rope.PositionGetter.__call__ = onnx_friendly_position_getter

    def onnx_friendly_affine_inverse(matrix: Any) -> Any:
        rotation = matrix[..., :3, :3]
        translation = matrix[..., :3, 3:]
        padding = matrix[..., 3:, :]
        rotation_t = rotation.transpose(-2, -1)
        return torch_module.cat([torch_module.cat([rotation_t, -rotation_t @ translation], dim=-1), padding], dim=-2)

    for module_name in [
        "depth_anything_3.api",
        "depth_anything_3.model.cam_enc",
        "depth_anything_3.model.da3",
        "depth_anything_3.utils.geometry",
    ]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "affine_inverse"):
            module.affine_inverse = onnx_friendly_affine_inverse


def load_da3_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from depth_anything_3.api import DepthAnything3

    patch_da3_for_onnx_export(torch)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
        if device.index is not None:
            torch.cuda.set_device(device)
        else:
            torch.cuda.set_device(0)

    print(f"Loading DA3 model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device).eval()
    return torch, model


def make_dummy_camera_tensors(args: argparse.Namespace, torch_module: Any, device: Any) -> tuple[Any, Any]:
    extrinsics = (
        torch_module.eye(4, dtype=torch_module.float32, device=device)
        .view(1, 1, 4, 4)
        .repeat(1, args.views, 1, 1)
        .clone()
    )
    intrinsics = (
        torch_module.eye(3, dtype=torch_module.float32, device=device)
        .view(1, 1, 3, 3)
        .repeat(1, args.views, 1, 1)
        .clone()
    )
    focal = 0.5 * float(args.width + args.height)
    intrinsics[:, :, 0, 0] = focal
    intrinsics[:, :, 1, 1] = focal
    intrinsics[:, :, 0, 2] = 0.5 * float(args.width - 1)
    intrinsics[:, :, 1, 2] = 0.5 * float(args.height - 1)
    return extrinsics, intrinsics


def affine_inverse_onnx(torch_module: Any, matrix: Any) -> Any:
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3:]
    padding = matrix[..., 3:, :]
    rotation_t = rotation.transpose(-2, -1)
    return torch_module.cat([torch_module.cat([rotation_t, -rotation_t @ translation], dim=-1), padding], dim=-2)


def normalize_camera_extrinsics(torch_module: Any, extrinsics: Any) -> Any:
    transform = affine_inverse_onnx(torch_module, extrinsics[:, :1])
    extrinsics_norm = extrinsics @ transform
    c2ws = affine_inverse_onnx(torch_module, extrinsics_norm)
    translations = c2ws[..., :3, 3]
    dists = translations.norm(dim=-1)
    sorted_dists, _ = torch_module.sort(dists.reshape(-1))
    median_dist = sorted_dists[sorted_dists.shape[0] // 2]
    median_dist = torch_module.clamp(median_dist, min=1e-1)
    translation = extrinsics_norm[..., :3, 3] / median_dist
    top = torch_module.cat([extrinsics_norm[..., :3, :3], translation[..., None]], dim=-1)
    return torch_module.cat([top, extrinsics_norm[..., 3:, :]], dim=-2)


class DA3ForwardExportWrapper:
    def __new__(cls, torch_module: Any, *args: Any, **kwargs: Any):
        class _Wrapper(torch_module.nn.Module):
            def __init__(
                self,
                model: Any,
                output_specs: list[tuple[str, str]],
                ref_view_strategy: str,
                camera_inputs: bool,
                normalize_camera_inputs: bool,
            ) -> None:
                super().__init__()
                self.model = model
                self.output_specs = output_specs
                self.ref_view_strategy = ref_view_strategy
                self.camera_inputs = camera_inputs
                self.normalize_camera_inputs = normalize_camera_inputs

            def forward(self, images: Any, extrinsics: Any = None, intrinsics: Any = None) -> tuple[Any, ...]:
                if self.camera_inputs and self.normalize_camera_inputs:
                    extrinsics = normalize_camera_extrinsics(torch_module, extrinsics)
                raw = self.model.forward(
                    images,
                    extrinsics=extrinsics if self.camera_inputs else None,
                    intrinsics=intrinsics if self.camera_inputs else None,
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=False,
                    ref_view_strategy=self.ref_view_strategy,
                )
                return tuple(get_raw_output(raw, key) for _, key in self.output_specs)

        return _Wrapper(*args, **kwargs)


def export_onnx(args: argparse.Namespace, torch_module: Any, model: Any) -> list[str]:
    if args.views <= 0 or args.height <= 0 or args.width <= 0:
        raise RuntimeError("--views, --height, and --width must be positive")

    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch_module.device(args.device)
    images = torch_module.zeros((1, args.views, 3, args.height, args.width), dtype=torch_module.float32, device=device)
    camera_inputs = None
    if args.camera_inputs:
        camera_inputs = make_dummy_camera_tensors(args, torch_module, device)

    print(f"Tracing DA3 forward input images: {tuple(images.shape)}")
    if args.camera_inputs:
        assert camera_inputs is not None
        print(f"Tracing DA3 forward input extrinsics: {tuple(camera_inputs[0].shape)}")
        print(f"Tracing DA3 forward input intrinsics: {tuple(camera_inputs[1].shape)}")
    with torch_module.no_grad(), force_fp16_autocast(torch_module, args.force_fp16_autocast):
        raw = model.forward(
            images,
            extrinsics=(
                normalize_camera_extrinsics(torch_module, camera_inputs[0])
                if camera_inputs is not None and args.normalize_camera_inputs
                else camera_inputs[0]
                if camera_inputs is not None
                else None
            ),
            intrinsics=camera_inputs[1] if camera_inputs is not None else None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy=args.ref_view_strategy,
        )
        output_specs = select_output_specs(raw)

        print("Forward outputs selected for ONNX:")
        for output_name, key in output_specs:
            print(f"  {output_name} <- {key}: {format_shape(get_raw_output(raw, key))}")

        wrapper = DA3ForwardExportWrapper(
            torch_module,
            model,
            output_specs,
            args.ref_view_strategy,
            args.camera_inputs,
            args.normalize_camera_inputs,
        ).eval()
        output_names = [name for name, _ in output_specs]
        export_inputs: tuple[Any, ...] = (images,)
        input_names = ["images"]
        if camera_inputs is not None:
            export_inputs = (images, camera_inputs[0], camera_inputs[1])
            input_names.extend(["input_extrinsics", "input_intrinsics"])
        print(f"Exporting ONNX: {onnx_path}")
        torch_module.onnx.export(
            wrapper,
            export_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"Saved ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    inspect_onnx(onnx_path)
    return output_names


def inspect_onnx(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("ONNX package not available; skipping ONNX shape inspection")
        return

    def value_shape(value_info: Any) -> list[Any]:
        shape = value_info.type.tensor_type.shape
        result: list[Any] = []
        for dim in shape.dim:
            if dim.dim_value:
                result.append(dim.dim_value)
            elif dim.dim_param:
                result.append(dim.dim_param)
            else:
                result.append("?")
        return result

    graph = onnx.load(str(onnx_path)).graph
    print("ONNX inputs:")
    for item in graph.input:
        print(f"  {item.name}: {value_shape(item)}")
    print("ONNX outputs:")
    for item in graph.output:
        print(f"  {item.name}: {value_shape(item)}")


def build_tensorrt_engine(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from convert_depth_anything_v3_to_tensorrt import build_profile, require_tensorrt

    trt = require_tensorrt()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX for TensorRT: {onnx_path}")
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(index)}", file=sys.stderr)
        raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gb * (1024**3))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    fp16 = args.fp16 and not args.fp32
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: platform_has_fast_fp16 is false; building without FP16")

    profile_args = SimpleNamespace(
        height=args.height,
        width=args.width,
        min_views=args.views,
        opt_views=args.views,
        max_views=args.views,
        dynamic_views=False,
    )
    profile = build_profile(trt, builder, network, profile_args)
    if profile is not None:
        config.add_optimization_profile(profile)

    input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    output_names = [network.get_output(i).name for i in range(network.num_outputs)]
    print("TensorRT network inputs:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"  {tensor.name}: {tuple(tensor.shape)}")
    print("TensorRT network outputs:")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"  {tensor.name}: {tuple(tensor.shape)}")
    print(f"Building TensorRT engine: {engine_path}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.write_bytes(serialized)
    print(f"Saved TensorRT engine: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    print("\nC++ usage:")
    print("  xfeat::DepthAnythingV3TRT::Params params;")
    print(f'  params.engine_path = "{engine_path}";')
    print("  xfeat::DepthAnythingV3TRT depth(params);")
    print("\nExpected names:")
    print(f"  inputs : {input_names}")
    if args.camera_inputs:
        print(
            f"  outputs: {output_names} "
            "(C++ runtime uses depth and predicted extrinsics for pose-scale alignment; confidence is ignored)"
        )
    else:
        print(f"  outputs: {output_names} (C++ runtime uses depth; confidence/pose/intrinsics are ignored)")


def main() -> int:
    args = parse_args()
    if args.fp32:
        args.fp16 = False

    if args.hf_home:
        hf_home = Path(args.hf_home)
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)

    repo = ensure_da3_repo(args)
    add_da3_to_path(repo)

    if not args.skip_onnx:
        torch_module, model = load_da3_model(args)
        export_onnx(args, torch_module, model)
    else:
        inspect_onnx(Path(args.onnx))

    if not args.skip_engine:
        build_tensorrt_engine(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
