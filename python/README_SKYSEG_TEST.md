# Sky Segmentation ONNX Model Test Script

This script tests the `skyseg.onnx` model for sky segmentation using ONNX Runtime.

Based on: [Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing)

## Prerequisites

```bash
pip install onnxruntime opencv-python numpy
# Or for GPU support:
pip install onnxruntime-gpu opencv-python numpy
```

## Usage

### Basic Inference

Test on a single image with default settings:
```bash
pixi run python python/test_skyseg.py --image image/sample1.jpg --show --save
```

### Custom Input Size

The model supports different input sizes (default is 320x320):
```bash
pixi run python python/test_skyseg.py \
    --image image/sample1.jpg \
    --input_size 512 512 \
    --save --show
```

### Adjust Overlay Transparency

Control the visualization overlay transparency:
```bash
pixi run python python/test_skyseg.py \
    --image image/sample1.jpg \
    --alpha 0.7 \
    --save --show
```

### Benchmark Mode

Run performance benchmark:
```bash
pixi run python python/test_skyseg.py \
    --image image/sample1.jpg \
    --benchmark \
    --warmup 5 \
    --iterations 100
```

### Process Multiple Images

Process all images in a directory:
```bash
pixi run python python/test_skyseg.py --image image/ --save
```

### CPU-Only Mode

Force CPU execution (useful for testing without GPU):
```bash
pixi run python python/test_skyseg.py \
    --image image/sample1.jpg \
    --providers CPUExecutionProvider \
    --save --show
```

## Command Line Arguments

- `--model`: Path to ONNX model file (default: `onnx_model/skyseg.onnx`)
- `--image`: Path to input image or directory (default: `image/sample1.jpg`)
- `--input_size`: Model input size as width height (default: `320 320`)
- `--max_size`: Maximum image dimension - larger images are downsampled (default: 640)
- `--alpha`: Overlay transparency 0.0-1.0 (default: 0.5)
- `--save`: Save visualization results to `output/` directory
- `--show`: Display results in a window
- `--benchmark`: Run benchmark test
- `--warmup`: Number of warmup iterations for benchmark (default: 3)
- `--iterations`: Number of iterations for benchmark (default: 100)
- `--providers`: ONNX Runtime execution providers (default: `CUDAExecutionProvider CPUExecutionProvider`)

## Output

The script provides:
- **Inference time** and **FPS** for each image
- **Sky percentage** in the image
- **Mask statistics** (shape, value range)
- **Three visualization outputs**:
  - `*_comparison.jpg` - Side-by-side: original | mask | overlay
  - `*_mask.jpg` - Segmentation mask only
  - `*_overlay.jpg` - Blended overlay on original image
- **Benchmark statistics** (mean, median, min, max inference times)

## Example Output

```
============================================================
Processing: image/sample1.jpg
============================================================
Original image shape: (480, 640, 3)
Downsampled image shape: (240, 320, 3)
Inference time: 8.45 ms
FPS: 118.34
Mask shape: (320, 320)
Mask range: [0, 255]
Sky percentage: 42.35%

Saved results to output/:
  - sample1_comparison.jpg (side-by-side)
  - sample1_mask.jpg (mask only)
  - sample1_overlay.jpg (overlay)
```

## Model Details

- **Architecture**: U-2-Net based sky segmentation
- **Input**: RGB image, normalized with ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Output**: Single-channel segmentation mask (0-255)
- **Typical input size**: 320x320 (configurable)

## Notes

- Images larger than `--max_size` are automatically downsampled using pyramid downsampling
- The model uses PyTorch ImageNet normalization
- Output masks are normalized to 0-255 range for visualization
- GPU acceleration is used automatically if available (via CUDA execution provider)
