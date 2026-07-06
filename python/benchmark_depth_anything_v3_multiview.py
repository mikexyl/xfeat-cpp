#!/usr/bin/env python3
"""Benchmark upstream Depth Anything 3 real multi-view inference."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Input image paths for one multi-view DA3 inference call")
    parser.add_argument(
        "--da3-repo",
        default=os.environ.get("DEPTH_ANYTHING_V3_REPO", "/tmp/depth-anything-3"),
        help="Path to the upstream depth-anything-3 checkout",
    )
    parser.add_argument(
        "--model",
        default="depth-anything/DA3-SMALL",
        help="Hugging Face model id or local model directory",
    )
    parser.add_argument(
        "--hf-home",
        default="onnx_model/mono_depth/depth_anything_v3/hf_cache",
        help="HF_HOME cache directory to use before loading the model",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=5, help="Measured iterations")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 process_res")
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        help="DA3 process_res_method",
    )
    parser.add_argument(
        "--ref-view-strategy",
        default="middle",
        help="DA3 reference view strategy for the grouped input",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="Use DA3 ray pose mode during inference",
    )
    parser.add_argument("--json-out", help="Optional benchmark JSON output path")
    return parser.parse_args()


def add_da3_to_path(repo: Path) -> None:
    src = repo / "src"
    if not src.is_dir():
        raise RuntimeError(f"Depth Anything 3 source directory not found: {src}")
    sys.path.insert(0, str(src))


def mib(bytes_value: int | float) -> float:
    return float(bytes_value) / (1024.0 * 1024.0)


class CudaMemorySampler:
    def __init__(self, torch_module: Any, device: Any, interval_s: float = 0.001) -> None:
        self._torch = torch_module
        self._device = device
        self._interval_s = interval_s
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_bytes = 0

    def _used_bytes(self) -> int:
        free_bytes, total_bytes = self._torch.cuda.mem_get_info(self._device)
        return int(total_bytes - free_bytes)

    def start(self) -> None:
        self.peak_used_bytes = self._used_bytes()
        self._running.set()

        def sample() -> None:
            while self._running.is_set():
                self.peak_used_bytes = max(self.peak_used_bytes, self._used_bytes())
                time.sleep(self._interval_s)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self.peak_used_bytes = max(self.peak_used_bytes, self._used_bytes())
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
        return self.peak_used_bytes


def cuda_snapshot(torch_module: Any, device: Any) -> dict[str, float]:
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    used_bytes = int(total_bytes - free_bytes)
    return {
        "used_mib": mib(used_bytes),
        "free_mib": mib(int(free_bytes)),
        "total_mib": mib(int(total_bytes)),
    }


def timing_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0}
    return {
        "mean": statistics.fmean(values_ms),
        "median": statistics.median(values_ms),
        "min": min(values_ms),
        "max": max(values_ms),
        "stddev": statistics.pstdev(values_ms),
    }


def run_inference(model: Any, args: argparse.Namespace) -> Any:
    return model.inference(
        image=args.images,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        export_dir=None,
    )


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise RuntimeError("--warmup must be non-negative")
    if args.runs <= 0:
        raise RuntimeError("--runs must be positive")

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    add_da3_to_path(Path(args.da3_repo))

    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)

    before_model = cuda_snapshot(torch, device) if device.type == "cuda" else None

    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    after_model = cuda_snapshot(torch, device) if device.type == "cuda" else None

    for _ in range(args.warmup):
        prediction = run_inference(model, args)
        del prediction
        gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    after_warmup = cuda_snapshot(torch, device) if device.type == "cuda" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler = CudaMemorySampler(torch, device) if device.type == "cuda" else None
    if sampler is not None:
        sampler.start()

    times_ms: list[float] = []
    output_shape: dict[str, Any] = {}
    for _ in range(args.runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = run_inference(model, args)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
        output_shape = {
            "depth": list(prediction.depth.shape),
            "conf": list(prediction.conf.shape),
            "extrinsics": list(prediction.extrinsics.shape),
            "intrinsics": list(prediction.intrinsics.shape),
        }
        del prediction
        gc.collect()

    peak_sampled = sampler.stop() if sampler is not None else 0
    after_benchmark = cuda_snapshot(torch, device) if device.type == "cuda" else None
    torch_peak_allocated = mib(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    torch_peak_reserved = mib(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0.0
    stats = timing_stats(times_ms)
    views_per_call = len(args.images)
    calls_per_s = 1000.0 / stats["mean"] if stats["mean"] > 0.0 else 0.0

    result: dict[str, Any] = {
        "mode": "real_multiview",
        "model": args.model,
        "device": str(device),
        "da3_repo": str(args.da3_repo),
        "image_paths": args.images,
        "views_per_call": views_per_call,
        "warmup": args.warmup,
        "runs": args.runs,
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "ref_view_strategy": args.ref_view_strategy,
        "use_ray_pose": args.use_ray_pose,
        "model_load_ms": load_ms,
        "runtime_ms": stats,
        "throughput": {
            "calls_per_s": calls_per_s,
            "views_per_s": calls_per_s * views_per_call,
        },
        "output_shape": output_shape,
        "before_model": before_model,
        "after_model": after_model,
        "after_warmup": after_warmup,
        "after_benchmark": after_benchmark,
        "peak_sampled_used_mib": mib(peak_sampled),
        "torch_peak_allocated_mib": torch_peak_allocated,
        "torch_peak_reserved_mib": torch_peak_reserved,
    }

    print("\nBenchmark result")
    print(f"  mode=real_multiview")
    print(f"  model={args.model}")
    print(f"  views_per_call={views_per_call}, warmup={args.warmup}, runs={args.runs}")
    print(
        "  runtime_ms: "
        f"mean={stats['mean']:.3f}, median={stats['median']:.3f}, min={stats['min']:.3f}, "
        f"max={stats['max']:.3f}, stddev={stats['stddev']:.3f}"
    )
    print(f"  throughput: {calls_per_s:.2f} calls/s, {calls_per_s * views_per_call:.2f} views/s")
    if before_model is not None:
        baseline = before_model["used_mib"]
        print("  vram:")
        for label in ["before_model", "after_model", "after_warmup", "after_benchmark"]:
            snapshot = result[label]
            print(f"    {label:16s} {snapshot['used_mib']:.1f} MiB used (+{snapshot['used_mib'] - baseline:.1f} MiB)")
        print(f"    peak_sampled    {result['peak_sampled_used_mib']:.1f} MiB used")
        print(f"    torch_peak_allocated={torch_peak_allocated:.1f} MiB")
        print(f"    torch_peak_reserved={torch_peak_reserved:.1f} MiB")
    print(f"  output_shape={output_shape}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  json={json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
