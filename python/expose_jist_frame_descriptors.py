#!/usr/bin/env python3
"""Expose JIST's normalized per-frame descriptors as a second ONNX output."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


DEFAULT_INTERNAL_TENSOR = "/model/aggregation/aggregation.4/Div_output_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--internal-tensor",
        default=DEFAULT_INTERNAL_TENSOR,
        help="Normalized [sequence, descriptor_dim] tensor to expose",
    )
    parser.add_argument("--output-name", default="frame_descriptors")
    return parser.parse_args()


def tensor_shape(model: onnx.ModelProto, tensor_name: str) -> list[int]:
    inferred = shape_inference.infer_shapes(model)
    values = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    for value in values:
        if value.name != tensor_name:
            continue
        dimensions: list[int] = []
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                raise RuntimeError(f"{tensor_name} has a dynamic or unknown shape")
            dimensions.append(dimension.dim_value)
        return dimensions
    raise RuntimeError(f"shape inference did not find tensor {tensor_name!r}")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input model is not readable: {args.input}")

    model = onnx.load(args.input)
    produced_tensors = {name for node in model.graph.node for name in node.output}
    if args.internal_tensor not in produced_tensors:
        raise SystemExit(f"internal tensor does not exist: {args.internal_tensor}")
    if any(output.name == args.output_name for output in model.graph.output):
        raise SystemExit(f"output already exists: {args.output_name}")

    shape = tensor_shape(model, args.internal_tensor)
    if len(shape) != 2:
        raise SystemExit(
            f"expected [sequence, descriptor_dim], got {args.internal_tensor} {shape}"
        )

    model.graph.node.append(
        helper.make_node(
            "Identity",
            [args.internal_tensor],
            [args.output_name],
            name="ExposeFrameDescriptors",
        )
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            args.output_name,
            TensorProto.FLOAT,
            shape,
        )
    )
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["jist_frame_descriptor_output"] = args.output_name
    metadata["jist_frame_descriptor_source"] = args.internal_tensor
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"Sequence output: {model.graph.output[0].name}")
    print(f"Frame output:    {args.output_name} {shape}")
    print(f"Saved:           {args.output}")


if __name__ == "__main__":
    main()
