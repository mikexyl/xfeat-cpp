#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Sky Segmentation TensorRT engine.
"""
import argparse
import copy
import time
from pathlib import Path

import cv2 as cv
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT or PyCUDA not available")


TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if TRT_AVAILABLE else None


class TensorRTInference:
    """TensorRT inference engine wrapper."""
    
    def __init__(self, engine_path):
        """
        Initialize TensorRT engine.
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        # Allocate device memory
        self.input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize
        
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        self.stream = cuda.Stream()
        
        print(f"TensorRT engine loaded successfully")
        print(f"Input: {self.input_name}, shape: {self.input_shape}")
        print(f"Output: {self.output_name}, shape: {self.output_shape}")
    
    def infer(self, input_data):
        """
        Run inference.
        
        Args:
            input_data: Input numpy array
        
        Returns:
            Output numpy array
        """
        # Ensure input is contiguous and float32
        input_data = np.ascontiguousarray(input_data.astype(np.float32))
        
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Allocate output buffer
        output_data = np.empty(self.output_shape, dtype=np.float32)
        
        # Copy output from device
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        return output_data
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()


def run_inference(trt_engine, input_size, image):
    """
    Run sky segmentation inference on an image using TensorRT.
    
    Args:
        trt_engine: TensorRT inference engine
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
    trt_result = trt_engine.infer(x)

    # Post process: Normalize to 0-255
    trt_result = np.array(trt_result).squeeze()
    min_value = np.min(trt_result)
    max_value = np.max(trt_result)
    trt_result = (trt_result - min_value) / (max_value - min_value)
    trt_result *= 255
    trt_result = trt_result.astype('uint8')

    return trt_result


def create_visualization(image, mask, alpha=0.5):
    """Create visualization by overlaying the sky mask on the original image."""
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


def downsample_image(image, max_size):
    """Downsample image if it's larger than max_size."""
    while image.shape[0] >= max_size and image.shape[1] >= max_size:
        image = cv.pyrDown(image)
    return image


def run_benchmark(trt_engine, image_path, args):
    """Run benchmark test."""
    print(f"\n{'='*60}")
    print(f"Running TensorRT Benchmark")
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
        _ = run_inference(trt_engine, args.input_size, image)
    
    # Benchmark
    print("Running benchmark...")
    times = []
    for i in range(args.iterations):
        start_time = time.time()
        _ = run_inference(trt_engine, args.input_size, image)
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
    print(f"TensorRT Benchmark Results")
    print(f"{'='*60}")
    print(f"Mean inference time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Median inference time: {median_time*1000:.2f} ms")
    print(f"Min inference time: {min_time*1000:.2f} ms")
    print(f"Max inference time: {max_time*1000:.2f} ms")
    print(f"Mean FPS: {1/mean_time:.2f}")
    print(f"Max FPS: {1/min_time:.2f}")
    print(f"\nSpeedup vs ONNX Runtime CPU (~260ms): {260/(mean_time*1000):.2f}x")


def process_image(trt_engine, image_path, args):
    """Process a single image."""
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
    mask = run_inference(trt_engine, args.input_size, image)
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
        cv.imwrite(str(output_dir / f"{base_name}_trt_comparison.jpg"), comparison)
        cv.imwrite(str(output_dir / f"{base_name}_trt_mask.jpg"), mask_full)
        cv.imwrite(str(output_dir / f"{base_name}_trt_overlay.jpg"), blended)
        
        print(f"\nSaved results to output/:")
        print(f"  - {base_name}_trt_comparison.jpg (side-by-side)")
        print(f"  - {base_name}_trt_mask.jpg (mask only)")
        print(f"  - {base_name}_trt_overlay.jpg (overlay)")
    
    # Display results
    if args.show:
        cv.namedWindow("TensorRT Sky Segmentation Results", 0)
        cv.imshow('TensorRT Sky Segmentation Results', comparison)
        print("\nPress any key to continue...")
        cv.waitKey(0)
        cv.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(
        description="Test Sky Segmentation TensorRT engine"
    )
    
    parser.add_argument(
        '--engine',
        type=str,
        default='onnx_model/skyseg.engine',
        help='Path to TensorRT engine file'
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
        default=10,
        help='Number of warmup iterations for benchmark'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations for benchmark'
    )
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    if not TRT_AVAILABLE:
        print("Error: TensorRT is not available. Please install tensorrt and pycuda.")
        return
    
    # Check if engine file exists
    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"Error: Engine file not found: {engine_path}")
        print(f"\nPlease convert the ONNX model first:")
        print(f"pixi run python python/convert_skyseg_to_tensorrt.py")
        return
    
    print(f"Loading TensorRT engine from: {engine_path}")
    
    # Load TensorRT engine
    try:
        trt_engine = TensorRTInference(str(engine_path))
    except Exception as e:
        print(f"Error loading engine: {e}")
        import traceback
        traceback.print_exc()
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
                run_benchmark(trt_engine, img_file, args)
                break  # Only benchmark on first image
            else:
                process_image(trt_engine, img_file, args)
    else:
        # Process single image
        if args.benchmark:
            run_benchmark(trt_engine, image_path, args)
        else:
            process_image(trt_engine, image_path, args)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
