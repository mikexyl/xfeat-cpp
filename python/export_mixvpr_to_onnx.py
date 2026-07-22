#!/usr/bin/env python3
"""Export the official ResNet-50 MixVPR 4096-D checkpoint to ONNX."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


OFFICIAL_REPOSITORY = "https://github.com/amaralibey/MixVPR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("onnx_model/mixvpr_resnet50_4096d.onnx"),
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(torch_module, torchvision_module):
    nn = torch_module.nn
    functional = torch_module.nn.functional

    class ResNet50Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torchvision_module.models.resnet50(weights=None)
            self.model.avgpool = None
            self.model.fc = None
            self.model.layer4 = None

        def forward(self, image):
            features = self.model.conv1(image)
            features = self.model.bn1(features)
            features = self.model.relu(features)
            features = self.model.maxpool(features)
            features = self.model.layer1(features)
            features = self.model.layer2(features)
            return self.model.layer3(features)

    class FeatureMixerLayer(nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.mix = nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, dimension),
                nn.ReLU(),
                nn.Linear(dimension, dimension),
            )

        def forward(self, features):
            return features + self.mix(features)

    class MixVPR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            spatial_dimension = 20 * 20
            self.mix = nn.Sequential(*(FeatureMixerLayer(spatial_dimension) for _ in range(4)))
            self.channel_proj = nn.Linear(1024, 1024)
            self.row_proj = nn.Linear(spatial_dimension, 4)

        def forward(self, features):
            features = features.flatten(2)
            features = self.mix(features)
            features = features.permute(0, 2, 1)
            features = self.channel_proj(features)
            features = features.permute(0, 2, 1)
            features = self.row_proj(features)
            return functional.normalize(features.flatten(1), p=2, dim=-1)

    class MixVPRModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = ResNet50Backbone()
            self.aggregator = MixVPR()

        def forward(self, image):
            return self.aggregator(self.backbone(image))

    return MixVPRModel()


def inspect_onnx(onnx_module, path: Path) -> None:
    model = onnx_module.load(str(path))
    onnx_module.checker.check_model(model)
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise RuntimeError("MixVPR ONNX graph must contain one input and one output")

    def dimensions(value_info) -> list[int]:
        return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]

    input_shape = dimensions(graph.input[0])
    output_shape = dimensions(graph.output[0])
    if graph.input[0].name != "image" or input_shape != [1, 3, 320, 320]:
        raise RuntimeError(f"unexpected ONNX input: {graph.input[0].name} {input_shape}")
    if graph.output[0].name != "descriptor" or output_shape != [1, 4096]:
        raise RuntimeError(f"unexpected ONNX output: {graph.output[0].name} {output_shape}")
    print(f"ONNX input:  {graph.input[0].name} {input_shape}")
    print(f"ONNX output: {graph.output[0].name} {output_shape}")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    import onnx
    import torch
    import torchvision

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device(args.device)
    model = make_model(torch, torchvision)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    dummy = torch.zeros((1, 3, 320, 320), dtype=torch.float32, device=device)
    with torch.inference_mode():
        reference = model(dummy)
    if tuple(reference.shape) != (1, 4096):
        raise RuntimeError(f"unexpected PyTorch output shape: {tuple(reference.shape)}")
    print(f"PyTorch descriptor L2 norm: {reference.norm().item():.8f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["descriptor"],
        dynamo=False,
    )

    onnx_model = onnx.load(str(args.output))
    metadata = {
        "model": "MixVPR ResNet-50 4096-D",
        "source": OFFICIAL_REPOSITORY,
        "checkpoint_sha256": sha256(args.checkpoint),
        "preprocessing": "RGB float32; ImageNet mean/std; 320x320",
    }
    del onnx_model.metadata_props[:]
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, str(args.output))
    inspect_onnx(onnx, args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size / (1024 * 1024):.2f} MiB)")
    print(f"SHA-256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
