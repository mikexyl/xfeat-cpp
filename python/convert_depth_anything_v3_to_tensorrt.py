#!/usr/bin/env python3
"""
Convert a Depth Anything V3 ONNX model to a TensorRT engine.

The C++ runtime loads .engine files only. This helper is intentionally artifact-free:
provide your own DA3 ONNX file and keep generated files under ignored onnx_model/.
"""
import argparse
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--onnx",
        default="onnx_model/mono_depth/depth_anything_v3/depth_anything_v3.onnx",
        help="Input DA3 ONNX model path",
    )
    parser.add_argument(
        "--output",
        default="onnx_model/mono_depth/depth_anything_v3/depth_anything_v3.engine",
        help="Output TensorRT engine path",
    )
    parser.add_argument("--height", type=int, default=280, help="Model input height for dynamic dimensions")
    parser.add_argument("--width", type=int, default=504, help="Model input width for dynamic dimensions")
    parser.add_argument("--min-views", type=int, default=1, help="Minimum grouped view count for dynamic engines")
    parser.add_argument("--opt-views", type=int, default=1, help="Optimal grouped view count for dynamic engines")
    parser.add_argument("--max-views", type=int, default=1, help="Maximum grouped view count for dynamic engines")
    parser.add_argument(
        "--dynamic-views",
        action="store_true",
        help="Rewrite the ONNX input view/batch dimension to -1 before building the profile",
    )
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 precision")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 precision")
    parser.add_argument("--workspace-gb", type=float, default=4.0, help="TensorRT workspace limit in GiB")
    return parser.parse_args()


def require_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "tensorrt Python package not found. Install a TensorRT Python package matching your CUDA stack."
        ) from exc
    return trt


def build_profile(trt, builder, network, args):
    input_tensor = network.get_input(0)
    shape = list(input_tensor.shape)
    if len(shape) not in (4, 5):
        raise RuntimeError(f"Expected DA3 input rank 4 or 5, got {shape}")

    if len(shape) == 4:
        view_axis, channel_axis, height_axis, width_axis = 0, 1, 2, 3
    else:
        view_axis, channel_axis, height_axis, width_axis = 1, 2, 3, 4

    if args.dynamic_views:
        shape[view_axis] = -1
        input_tensor.shape = tuple(shape)

    shape = list(input_tensor.shape)
    channel_dim = shape[channel_axis]
    if channel_dim not in (3, -1):
        raise RuntimeError(f"Expected RGB channel dimension at axis {channel_axis}, got input shape {shape}")

    height = args.height if shape[height_axis] <= 0 else shape[height_axis]
    width = args.width if shape[width_axis] <= 0 else shape[width_axis]

    if args.min_views <= 0 or args.opt_views <= 0 or args.max_views <= 0:
        raise RuntimeError("View counts must be positive")
    if not (args.min_views <= args.opt_views <= args.max_views):
        raise RuntimeError("--min-views <= --opt-views <= --max-views is required")

    if not any(dim <= 0 for dim in shape):
        return None

    def concrete(views):
        dims = list(shape)
        if len(dims) == 5:
            dims[0] = 1 if dims[0] <= 0 else dims[0]
        dims[view_axis] = views
        dims[channel_axis] = 3
        dims[height_axis] = height
        dims[width_axis] = width
        return tuple(dims)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        concrete(args.min_views),
        concrete(args.opt_views),
        concrete(args.max_views),
    )
    return profile


def main():
    args = parse_args()
    if args.fp32:
        args.fp16 = False

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trt = require_tensorrt()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}")
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

    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: platform_has_fast_fp16 is false; building without FP16")

    profile = build_profile(trt, builder, network, args)
    if profile is not None:
        config.add_optimization_profile(profile)

    input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    output_names = [network.get_output(i).name for i in range(network.num_outputs)]
    print("Inputs:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"  {tensor.name}: {tuple(tensor.shape)}")
    print("Outputs:")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"  {tensor.name}: {tuple(tensor.shape)}")
    print(f"Building TensorRT engine: {output_path}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    output_path.write_bytes(serialized)
    print(f"Saved {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    print("\nC++ usage:")
    print("  xfeat::DepthAnythingV3TRT::Params params;")
    print(f'  params.engine_path = "{output_path}";')
    print("  xfeat::DepthAnythingV3TRT depth(params);")
    print("\nExpected names:")
    print(f"  inputs : {input_names}")
    print(f"  outputs: {output_names} (depth output preferred by name, sky output optional)")


if __name__ == "__main__":
    main()
