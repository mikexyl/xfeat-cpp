#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Sky Segmentation ONNX model.
Based on: https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing
"""
import argparse
import copy
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    """
    Run sky segmentation inference on an image.
    
    Args:
        onnx_session: ONNX Runtime session
        input_size: Tuple of (width, height) for model input
        image: Input image (BGR format)
    
    Returns:
        Segmentation mask (uint8, 0-255)
    """
    # Pre process: Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv.cvtColor(resize_image, cv.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    
    # PyTorch ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[1], input_size[0]).astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process: Normalize to 0-255
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype('uint8')

    return onnx_result


def create_visualization(image, mask, alpha=0.5):
    """
    Create visualization by overlaying the sky mask on the original image.
    
    Args:
        image: Original image (BGR)
        mask: Segmentation mask (grayscale, 0-255)
        alpha: Transparency for overlay
    
    Returns:
        Visualization image
    """
    # Resize mask to match original image size
    mask_resized = cv.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored overlay (sky in blue)
    overlay = image.copy()
    overlay[mask_resized > 127] = [255, 100, 0]  # Blue color for sky
    
    # Blend original and overlay
    blended = cv.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # Create side-by-side comparison
    mask_colored = cv.cvtColor(mask_resized, cv.COLOR_GRAY2BGR)
    comparison = np.hstack([image, mask_colored, blended])
    
    return comparison, mask_resized, blended


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Sky Segmentation ONNX model"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='onnx_model/skyseg.onnx',
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='image/sample1.jpg',
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        default=[320, 320],
        metavar=('WIDTH', 'HEIGHT'),
        help='Model input size (width height)'
    )
    parser.add_argument(
        '--max_size',
        type=int,
        default=640,
        help='Maximum image dimension (images larger will be downsampled)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Overlay transparency (0.0-1.0)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save visualization results'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark test'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='Number of warmup iterations for benchmark'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations for benchmark'
    )
    parser.add_argument(
        '--providers',
        type=str,
        nargs='+',
        default=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        help='ONNX Runtime execution providers'
    )
    
    args = parser.parse_args()
    return args


def downsample_image(image, max_size):
    """
    Downsample image if it's larger than max_size.
    
    Args:
        image: Input image
        max_size: Maximum dimension
    
    Returns:
        Downsampled image
    """
    while image.shape[0] >= max_size and image.shape[1] >= max_size:
        image = cv.pyrDown(image)
    return image


def process_image(onnx_session, image_path, args):
    """
    Process a single image.
    
    Args:
        onnx_session: ONNX Runtime session
        image_path: Path to input image
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Read image
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Original image shape: {image.shape}")
    
    # Downsample if needed
    if args.max_size > 0:
        image = downsample_image(image, args.max_size)
        print(f"Downsampled image shape: {image.shape}")
    
    # Run inference
    start_time = time.time()
    mask = run_inference(onnx_session, args.input_size, image)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"FPS: {1/inference_time:.2f}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min()}, {mask.max()}]")
    
    # Calculate sky percentage
    mask_resized = cv.resize(mask, (image.shape[1], image.shape[0]))
    sky_pixels = np.sum(mask_resized > 127)
    total_pixels = mask_resized.shape[0] * mask_resized.shape[1]
    sky_percentage = (sky_pixels / total_pixels) * 100
    print(f"Sky percentage: {sky_percentage:.2f}%")
    
    # Create visualization
    comparison, mask_full, blended = create_visualization(image, mask, args.alpha)
    
    # Save results
    if args.save:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        base_name = Path(image_path).stem
        cv.imwrite(str(output_dir / f"{base_name}_comparison.jpg"), comparison)
        cv.imwrite(str(output_dir / f"{base_name}_mask.jpg"), mask_full)
        cv.imwrite(str(output_dir / f"{base_name}_overlay.jpg"), blended)
        
        print(f"\nSaved results to output/:")
        print(f"  - {base_name}_comparison.jpg (side-by-side)")
        print(f"  - {base_name}_mask.jpg (mask only)")
        print(f"  - {base_name}_overlay.jpg (overlay)")
    
    # Display results
    if args.show:
        cv.namedWindow("Sky Segmentation Results", 0)
        cv.imshow('Sky Segmentation Results', comparison)
        print("\nPress any key to continue...")
        cv.waitKey(0)
        cv.destroyAllWindows()


def run_benchmark(onnx_session, image_path, args):
    """
    Run benchmark test.
    
    Args:
        onnx_session: ONNX Runtime session
        image_path: Path to input image
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print(f"Running Benchmark")
    print(f"{'='*60}")
    
    # Read and prepare image
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    if args.max_size > 0:
        image = downsample_image(image, args.max_size)
    
    print(f"Image shape: {image.shape}")
    print(f"Input size: {args.input_size}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(args.warmup):
        _ = run_inference(onnx_session, args.input_size, image)
    
    # Benchmark
    print("Running benchmark...")
    times = []
    for i in range(args.iterations):
        start_time = time.time()
        _ = run_inference(onnx_session, args.input_size, image)
        inference_time = time.time() - start_time
        times.append(inference_time)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{args.iterations}")
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"Mean inference time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Median inference time: {median_time*1000:.2f} ms")
    print(f"Min inference time: {min_time*1000:.2f} ms")
    print(f"Max inference time: {max_time*1000:.2f} ms")
    print(f"Mean FPS: {1/mean_time:.2f}")
    print(f"Max FPS: {1/min_time:.2f}")


def main():
    args = get_args()
    
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    print(f"Loading Sky Segmentation model from: {model_path}")
    print(f"Execution providers: {args.providers}")
    
    # Load ONNX model
    try:
        onnx_session = onnxruntime.InferenceSession(
            str(model_path),
            providers=args.providers
        )
        print(f"Model loaded successfully!")
        
        # Print model info
        input_info = onnx_session.get_inputs()[0]
        output_info = onnx_session.get_outputs()[0]
        print(f"Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        print(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get image path(s)
    image_path = Path(args.image)
    
    if not image_path.exists():
        print(f"Error: Image path not found: {image_path}")
        return
    
    # Process images
    if image_path.is_dir():
        # Process all images in directory
        image_files = list(image_path.glob('*.jpg')) + \
                     list(image_path.glob('*.png')) + \
                     list(image_path.glob('*.jpeg'))
        
        if not image_files:
            print(f"No image files found in {image_path}")
            return
        
        print(f"Found {len(image_files)} images")
        
        for img_file in image_files:
            if args.benchmark:
                run_benchmark(onnx_session, img_file, args)
                break  # Only benchmark on first image
            else:
                process_image(onnx_session, img_file, args)
    else:
        # Process single image
        if args.benchmark:
            run_benchmark(onnx_session, image_path, args)
        else:
            process_image(onnx_session, image_path, args)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
