#!/usr/bin/env python3
"""Download ONNX Runtime 1.22.0 GPU tarball into the parent workspace directory."""
import os
import sys
import urllib.request
import tarfile
from pathlib import Path

ONNXRT_VERSION = "1.22.0"
ONNXRT_DIR = f"onnxruntime-linux-x64-gpu-{ONNXRT_VERSION}"
ONNXRT_URL = (f"https://github.com/microsoft/onnxruntime/releases/download/"
              f"v{ONNXRT_VERSION}/{ONNXRT_DIR}.tgz")

def main():
    project_root = Path(__file__).resolve().parent.parent
    dest_dir = project_root.parent / ONNXRT_DIR
    marker = dest_dir / "lib" / "libonnxruntime.so"

    if marker.exists():
        print(f"ONNX Runtime already present at: {dest_dir}")
        return

    tgz = Path("/tmp") / f"{ONNXRT_DIR}.tgz"
    print(f"Downloading ONNX Runtime {ONNXRT_VERSION} GPU...")
    print(f"  URL: {ONNXRT_URL}")

    def progress(count, block_size, total):
        pct = min(count * block_size * 100 // total, 100)
        sys.stdout.write(f"\r  Progress: {pct}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(ONNXRT_URL, tgz, reporthook=progress)
    print()

    print(f"Extracting to {project_root.parent} ...")
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(project_root.parent)
    tgz.unlink()
    print(f"Done: {dest_dir}")

if __name__ == "__main__":
    main()
