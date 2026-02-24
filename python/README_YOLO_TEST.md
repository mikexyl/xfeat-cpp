# YOLO Segmentation TensorRT Engine Test Script

This script tests the `yolo26n-seg.engine` TensorRT model for instance segmentation.

## Prerequisites

```bash
pip install ultralytics opencv-python numpy
```

## Usage

### Basic Inference

Test on a single image with default settings:
```bash
python python/test_yolo_seg.py --image image/sample1.jpg --show
```

### Save Results

Save visualization to output directory:
```bash
python python/test_yolo_seg.py --image image/sample1.jpg --save
```

### Custom Parameters

Adjust confidence and IoU thresholds:
```bash
python python/test_yolo_seg.py \
    --image image/sample1.jpg \
    --conf 0.5 \
    --iou 0.7 \
    --imgsz 512 \
    --save --show
```

### Benchmark Mode

Run performance benchmark:
```bash
python python/test_yolo_seg.py \
    --image image/sample1.jpg \
    --benchmark \
    --warmup 5 \
    --iterations 100
```

### Process Multiple Images

Process all images in a directory:
```bash
python python/test_yolo_seg.py --image image/ --save
```

## Command Line Arguments

- `--engine`: Path to TensorRT engine file (default: `onnx_model/yolo26n-seg.engine`)
- `--image`: Path to input image or directory (default: `image/sample1.jpg`)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.7)
- `--imgsz`: Image size for inference - must match engine build size (default: 512)
- `--save`: Save visualization results to `output/` directory
- `--show`: Display results in a window
- `--benchmark`: Run benchmark test
- `--warmup`: Number of warmup iterations for benchmark (default: 3)
- `--iterations`: Number of iterations for benchmark (default: 100)

## Output

The script provides:
- **Inference time** and **FPS** for each image
- **Number of detections** and **segmentation masks**
- **Class names** and **confidence scores** for each detection
- **Visualization** with bounding boxes and segmentation masks
- **Benchmark statistics** (mean, median, min, max inference times)

## Example Output

```
============================================================
Processing: image/sample1.jpg
============================================================
Image shape: (480, 640, 3)
Inference time: 12.34 ms
FPS: 81.03
Number of detections: 3
  [0] person: 0.892
  [1] car: 0.756
  [2] bicycle: 0.643
Number of segmentation masks: 3
Saved visualization to: output/result_sample1.jpg
```

## Notes

- The engine file was built with `imgsz=512` and `half=True` (FP16 precision)
- Make sure your input image size matches or is compatible with the engine's expected size
- TensorRT engines are GPU-specific and may not be portable across different GPU architectures
