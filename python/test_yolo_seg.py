#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for YOLO segmentation TensorRT engine.
This script loads the yolo26n-seg.engine model and performs inference on test images.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLO segmentation TensorRT engine"
    )
    
    parser.add_argument(
        '--engine',
        type=str,
        default='onnx_model/yolo26n-seg.engine',
        help='Path to TensorRT engine file'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='image/sample1.jpg',
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=512,
        help='Image size for inference (must match engine build size)'
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
    
    args = parser.parse_args()
    return args


def visualize_results(image, results, save_path=None):
    """
    Visualize segmentation results on the image.
    
    Args:
        image: Input image (numpy array)
        results: YOLO results object
        save_path: Optional path to save the visualization
    
    Returns:
        Annotated image
    """
    # Get the annotated image from results
    annotated = results[0].plot()
    
    # Print detection statistics
    if results[0].boxes is not None:
        num_detections = len(results[0].boxes)
        print(f"Number of detections: {num_detections}")
        
        # Print class names and confidence scores
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names[cls_id]
            print(f"  [{i}] {cls_name}: {conf:.3f}")
    
    # Print segmentation statistics
    if results[0].masks is not None:
        num_masks = len(results[0].masks)
        print(f"Number of segmentation masks: {num_masks}")
    
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved visualization to: {save_path}")
    
    return annotated


def run_inference(model, image_path, args):
    """
    Run inference on a single image.
    
    Args:
        model: YOLO model
        image_path: Path to input image
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Run inference
    start_time = time.time()
    results = model.predict(
        image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False
    )
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"FPS: {1/inference_time:.2f}")
    
    # Visualize results
    save_path = None
    if args.save:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        save_path = str(output_dir / f"result_{Path(image_path).name}")
    
    annotated = visualize_results(image, results, save_path)
    
    # Display results
    if args.show:
        cv2.imshow('YOLO Segmentation Results', annotated)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_benchmark(model, image_path, args):
    """
    Run benchmark test to measure performance.
    
    Args:
        model: YOLO model
        image_path: Path to input image
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print(f"Running Benchmark")
    print(f"{'='*60}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(args.warmup):
        _ = model.predict(
            image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False
        )
    
    # Benchmark
    print("Running benchmark...")
    times = []
    for i in range(args.iterations):
        start_time = time.time()
        _ = model.predict(
            image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False
        )
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
    
    # Check if engine file exists
    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"Error: Engine file not found: {engine_path}")
        print(f"Please make sure the TensorRT engine file exists.")
        return
    
    print(f"Loading YOLO segmentation model from: {engine_path}")
    
    # Load model
    try:
        model = YOLO(str(engine_path), task='segment')
        print(f"Model loaded successfully!")
        print(f"Model type: {type(model)}")
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
                run_benchmark(model, img_file, args)
                break  # Only benchmark on first image
            else:
                run_inference(model, img_file, args)
    else:
        # Process single image
        if args.benchmark:
            run_benchmark(model, image_path, args)
        else:
            run_inference(model, image_path, args)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
