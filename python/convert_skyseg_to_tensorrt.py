#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Sky Segmentation ONNX model to TensorRT engine using trtexec.
"""
import argparse
import subprocess
from pathlib import Path


def build_engine_with_trtexec(onnx_file_path, engine_file_path, fp16_mode=True, max_workspace_size=4):
    """
    Build TensorRT engine from ONNX model using trtexec command-line tool.
    
    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        fp16_mode: Enable FP16 precision
        max_workspace_size: Maximum workspace size in GB
    
    Returns:
        Path to engine file if successful, None otherwise
    """
    print(f"Building TensorRT engine from: {onnx_file_path}")
    print(f"Output engine: {engine_file_path}")
    print(f"FP16 mode: {fp16_mode}")
    print(f"Max workspace size: {max_workspace_size} GB")
    
    # Build trtexec command
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",  # Full path to trtexec
        f"--onnx={onnx_file_path}",
        f"--saveEngine={engine_file_path}",
    ]
    
    if fp16_mode:
        cmd.append("--fp16")
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run trtexec
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        # Check if engine file was created
        if Path(engine_file_path).exists():
            engine_size_mb = Path(engine_file_path).stat().st_size / (1024 * 1024)
            print(f"\nEngine saved successfully! Size: {engine_size_mb:.2f} MB")
            return engine_file_path
        else:
            print("\nERROR: Engine file was not created")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: trtexec failed with exit code {e.returncode}")
        return None
    except FileNotFoundError:
        print("\nERROR: trtexec not found. Please make sure TensorRT is installed and in PATH")
        print("You may need to install TensorRT or add it to your PATH")
        return None


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert Sky Segmentation ONNX model to TensorRT engine"
    )
    
    parser.add_argument(
        '--onnx',
        type=str,
        default='onnx_model/skyseg.onnx',
        help='Path to input ONNX model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='onnx_model/skyseg.engine',
        help='Path to output TensorRT engine'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Enable FP16 precision (default: True)'
    )
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Use FP32 precision instead of FP16'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='Maximum workspace size in GB (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Handle fp32 flag
    if args.fp32:
        args.fp16 = False
    
    return args


def main():
    args = get_args()
    
    # Check if ONNX file exists
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"Error: ONNX file not found: {onnx_path}")
        return
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TensorRT Engine Conversion")
    print("="*60)
    
    # Build engine
    try:
        engine_path = build_engine_with_trtexec(
            str(onnx_path),
            str(output_path),
            fp16_mode=args.fp16,
            max_workspace_size=args.workspace
        )
        
        if engine_path:
            print("\n" + "="*60)
            print("Conversion completed successfully!")
            print("="*60)
            print(f"\nYou can now test the engine with:")
            print(f"pixi run python python/test_skyseg_trt.py --engine {engine_path} --image examples/gate.png --benchmark")
        else:
            print("\nConversion failed!")
            
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
