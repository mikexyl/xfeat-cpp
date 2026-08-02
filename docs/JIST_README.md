# JIST ONNX C++ Wrapper

C++ wrapper for JIST (Joint Image Sequence Transformer) ONNX models in the xfeat-cpp project.

For the TensorRT sequence-retrieval benchmark and its one-frame-per-sequence
argmax refinement, see [JIST Frame Refinement and VPR Evaluation](JIST_FRAME_REFINEMENT.md).

## Overview

JIST is a sequence-based visual place recognition model that processes multiple consecutive images to produce a single descriptor for loop closure detection and place recognition tasks. This wrapper provides an efficient C++ interface for JIST ONNX models with support for both single and batch inference, streaming processing, and database management.

## Features

- **Single & Batch Inference**: Process individual sequences or batches efficiently
- **Streaming Mode**: Rolling buffer for online/real-time processing
- **GPU Acceleration**: CUDA support via ONNX Runtime
- **Database Management**: Built-in similarity search for place recognition
- **Easy Integration**: Similar API to other xfeat-cpp components

## Model Export

First, export your trained JIST model to ONNX format using the provided script:

```bash
cd /workspaces/src/JIST

# Export to ONNX (simplified)
python export_to_onnx.py \
    --model_path JIST_r18_512_seqgem.pth \
    --output_dir exported_models \
    --seq_length 5 \
    --img_shape 288 512 \
    --simplify \
    --verify

# Optional: Export to TensorRT for faster inference
python export_to_onnx.py \
    --model_path JIST_r18_512_seqgem.pth \
    --output_dir exported_models \
    --seq_length 5 \
    --img_shape 288 512 \
    --to_tensorrt \
    --fp16
```

The exported model will be in `exported_models/JIST_r18_512_seqgem_simplified.onnx`.

## Quick Start

### Basic Usage

```cpp
#include <onnxruntime_cxx_api.h>
#include "xfeat-cpp/jist_onnx.h"

// Initialize ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "JistApp");

// Configure JIST parameters
xfeat::JistONNX::Params params;
params.model_path = "JIST_r18_512_seqgem.onnx";
params.use_gpu = true;
params.seq_length = 5;
params.img_height = 288;
params.img_width = 512;
params.descriptor_dim = 512;
params.normalize_output = true;

// Create JIST instance
xfeat::JistONNX jist(env, params);

// Load sequence of images
std::vector<cv::Mat> sequence;
for (int i = 0; i < 5; ++i) {
    sequence.push_back(cv::imread("frame_" + std::to_string(i) + ".jpg"));
}

// Generate descriptor
cv::Mat descriptor = jist.infer(sequence);
std::cout << "Descriptor: " << descriptor.rows << " x " << descriptor.cols << std::endl;
```

### Streaming Mode (Rolling Buffer)

```cpp
xfeat::JistONNX jist(env, params);

// Process video frames one by one
for (const auto& frame : video_frames) {
    cv::Mat descriptor;
    if (jist.add_frame(frame, descriptor)) {
        // Descriptor ready when buffer is full
        std::cout << "Generated descriptor at frame " << i << std::endl;
        // Use descriptor for place recognition
    }
}
```

### Place Recognition Database

```cpp
// Build database from map sequence
xfeat::JistDatabase database;

for (size_t i = 0; i < map_images.size() - seq_length; i += stride) {
    std::vector<cv::Mat> sequence(
        map_images.begin() + i,
        map_images.begin() + i + seq_length
    );
    cv::Mat descriptor = jist.infer(sequence);
    database.add(descriptor);
}

// Query for loop closure
std::vector<cv::Mat> query_sequence = get_current_sequence();
cv::Mat query_descriptor = jist.infer(query_sequence);

float similarity;
int match_idx = database.query(query_descriptor, similarity);

if (similarity > 0.85f) {
    std::cout << "Loop closure detected at index " << match_idx << std::endl;
}

// Get top-k matches
std::vector<int> indices;
std::vector<float> similarities;
database.query_top_k(query_descriptor, 5, indices, similarities);
```

### Batch Inference

```cpp
// Process multiple sequences at once (more efficient)
std::vector<std::vector<cv::Mat>> batch_sequences;
for (int b = 0; b < batch_size; ++b) {
    std::vector<cv::Mat> sequence = get_sequence(b);
    batch_sequences.push_back(sequence);
}

cv::Mat batch_descriptors = jist.infer_batch(batch_sequences);
// batch_descriptors: (batch_size x descriptor_dim)
```

### Save/Load Database

```cpp
// Save database to file
database.save("place_database.yml");

// Load database later
xfeat::JistDatabase loaded_db;
loaded_db.load("place_database.yml");
```

## API Reference

### JistONNX Class

#### Constructor
```cpp
JistONNX(Ort::Env& env, const Params& params);
```

#### Core Methods

**Single Inference**
```cpp
cv::Mat infer(const std::vector<cv::Mat>& image_sequence);
```
- Input: Vector of `seq_length` images (BGR format, any size)
- Output: Descriptor vector (1 x descriptor_dim, CV_32F)
- Images automatically resized to model input size

**Batch Inference**
```cpp
cv::Mat infer_batch(const std::vector<std::vector<cv::Mat>>& batch_sequences);
```
- Input: Batch of image sequences
- Output: Descriptors (batch_size x descriptor_dim, CV_32F)

**Streaming Interface**
```cpp
bool add_frame(const cv::Mat& image, cv::Mat& descriptor);
void reset_buffer();
bool is_buffer_ready() const;
size_t get_buffer_size() const;
```

### JistDatabase Class

#### Methods

```cpp
size_t add(const cv::Mat& descriptor);
int query(const cv::Mat& query_descriptor, float& similarity);
void query_top_k(const cv::Mat& query_descriptor, int k, 
                 std::vector<int>& indices, std::vector<float>& similarities);
cv::Mat get_descriptor(size_t index) const;
size_t size() const;
void clear();
void save(const std::string& filepath) const;
void load(const std::string& filepath);
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | required | Path to ONNX model file |
| `use_gpu` | bool | true | Use CUDA GPU acceleration |
| `seq_length` | int | 5 | Number of frames in sequence |
| `img_height` | int | 288 | Input image height |
| `img_width` | int | 512 | Input image width |
| `descriptor_dim` | int | 512 | Output descriptor dimension |
| `normalize_output` | bool | true | L2-normalize descriptors |

## Building

The JIST wrapper is automatically built with xfeat-cpp:

```bash
cd /workspaces/src/xfeat-cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running the Example

```bash
# Build
cd /workspaces/src/xfeat-cpp/build
make jist_example

# Run with single directory (uses same images for demo)
./examples/jist_example \
    path/to/JIST_r18_512_seqgem.onnx \
    path/to/image_sequence/

# Run with separate query directory
./examples/jist_example \
    path/to/JIST_r18_512_seqgem.onnx \
    path/to/map_sequence/ \
    path/to/query_sequence/
```

The example demonstrates:
1. Basic single sequence inference
2. Building a place database with sliding window
3. Loop closure detection
4. Streaming inference with rolling buffer
5. Batch inference
6. Database save/load

## Performance Tips

1. **Use GPU**: Enable CUDA for 5-10x speedup
2. **Batch Inference**: Process multiple sequences together when possible
3. **Streaming Mode**: Use rolling buffer for online applications
4. **Model Selection**: 
   - ResNet18: Faster, good for real-time
   - ResNet50/101: Better accuracy, slower
5. **Sequence Stride**: Use stride=1 for dense mapping, stride=seq_length for faster processing

## Typical Performance

On NVIDIA RTX 3090:
- Single sequence (seq_length=5): ~50-80 ms
- Batch of 8 sequences: ~200-300 ms (~25-40 ms per sequence)
- Streaming mode: ~50-80 ms per new frame (when buffer full)

## Loop Closure Detection Guidelines

1. **Similarity Threshold**: Typical threshold for loop closure is 0.80-0.90
2. **Top-K Retrieval**: Query top-5 to top-10 for verification
3. **Temporal Consistency**: Verify matches across multiple consecutive sequences
4. **Geometric Verification**: Use feature matching (e.g., XFeat) to verify spatial consistency

## Integration with ROS

Example ROS node structure:

```cpp
class JistPlaceRecognitionNode {
    xfeat::JistONNX jist_;
    xfeat::JistDatabase database_;
    
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::Mat descriptor;
        
        if (jist_.add_frame(image, descriptor)) {
            // Query database
            float similarity;
            int match_idx = database_.query(descriptor, similarity);
            
            if (similarity > loop_closure_threshold_) {
                publishLoopClosure(match_idx, similarity);
            }
            
            // Add to database
            database_.add(descriptor);
        }
    }
};
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `gpu_mem_limit` in session options
- Use smaller batch size
- Process sequences sequentially

### Slow Inference
- Ensure GPU is enabled (`use_gpu = true`)
- Check CUDA is available in ONNX Runtime
- Use simplified ONNX model
- Consider FP16 TensorRT engine

### Wrong Output Dimensions
- Verify model export parameters match wrapper parameters
- Check `seq_length`, `img_height`, `img_width` match model
- Ensure batch dimension is correct

## Citation

If you use JIST in your research, please cite:

```bibtex
@article{jist2024,
  title={JIST: Joint Image Sequence Transformer for Visual Place Recognition},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

Same as xfeat-cpp project.

## See Also

- [JIST Python Implementation](../JIST/)
- [XFeat Feature Matching](xfeat_onnx.h)
