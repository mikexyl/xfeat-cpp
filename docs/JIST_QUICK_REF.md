# JIST Quick Reference Guide

## Quick Start (30 seconds)

```cpp
#include <onnxruntime_cxx_api.h>
#include "xfeat-cpp/jist_onnx.h"

// Setup
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "App");
xfeat::JistONNX::Params params;
params.model_path = "JIST_r18_512_seqgem.onnx";
params.seq_length = 5;
xfeat::JistONNX jist(env, params);

// Inference
std::vector<cv::Mat> images = {img1, img2, img3, img4, img5};
cv::Mat descriptor = jist.infer(images);
```

## Common Patterns

### Pattern 1: Build Map Database
```cpp
xfeat::JistDatabase database;
int stride = 5;  // Non-overlapping sequences

for (size_t i = 0; i < images.size() - seq_length; i += stride) {
    std::vector<cv::Mat> seq(images.begin() + i, 
                             images.begin() + i + seq_length);
    database.add(jist.infer(seq));
}
database.save("map.yml");
```

### Pattern 2: Online Loop Closure Detection
```cpp
database.load("map.yml");
jist.reset_buffer();

for (const auto& frame : video_stream) {
    cv::Mat descriptor;
    if (jist.add_frame(frame, descriptor)) {
        float similarity;
        int match = database.query(descriptor, similarity);
        
        if (similarity > 0.85f) {
            std::cout << "Loop closure at " << match << std::endl;
        }
    }
}
```

### Pattern 3: Batch Processing
```cpp
std::vector<std::vector<cv::Mat>> batch;
for (int i = 0; i < 8; ++i) {
    batch.push_back(get_sequence(i));
}

cv::Mat descriptors = jist.infer_batch(batch);
// descriptors.rows = 8, descriptors.cols = descriptor_dim
```

## Parameter Cheat Sheet

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| `seq_length` | 5 | Frames per sequence |
| `img_height` | 288 | Input height |
| `img_width` | 512 | Input width |
| `descriptor_dim` | 512 | Output dimension |
| `use_gpu` | true | Enable CUDA |
| `normalize_output` | true | L2-normalize |

## Method Quick Reference

### JistONNX Methods

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `infer(sequence)` | Vector of images | 1 descriptor | Single sequence |
| `infer_batch(batch)` | Vector of sequences | N descriptors | Multiple sequences |
| `add_frame(image, desc)` | 1 image | bool + descriptor | Streaming |
| `reset_buffer()` | - | - | Clear buffer |
| `is_buffer_ready()` | - | bool | Check if ready |

### JistDatabase Methods

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `add(descriptor)` | 1 descriptor | index | Add to database |
| `query(desc, sim)` | 1 descriptor | index + similarity | Find best match |
| `query_top_k(desc, k, ...)` | 1 descriptor | indices + similarities | Get top-k |
| `save(path)` | filepath | - | Save database |
| `load(path)` | filepath | - | Load database |
| `size()` | - | count | Database size |

## Timing Reference

On RTX 3090:
- **Single sequence**: ~50-80 ms
- **Batch of 8**: ~200-300 ms (~30 ms/seq)
- **Streaming**: ~50-80 ms per frame

## Common Issues & Solutions

### Issue: "Model not found"
```cpp
// Solution: Check file path
if (!std::ifstream(model_path).good()) {
    std::cerr << "Model not found: " << model_path << std::endl;
}
```

### Issue: "Sequence size mismatch"
```cpp
// Solution: Ensure correct sequence length
if (images.size() != jist.get_seq_length()) {
    std::cerr << "Expected " << jist.get_seq_length() << " images" << std::endl;
}
```

### Issue: "CUDA out of memory"
```cpp
// Solution: Use smaller batch or CPU
params.use_gpu = false;  // Fall back to CPU
// Or reduce batch size
```

### Issue: "Low similarity scores"
```cpp
// Check if normalization is enabled
params.normalize_output = true;  // Should be true for cosine similarity

// Verify descriptors are normalized
double norm = cv::norm(descriptor);
std::cout << "Descriptor norm: " << norm << std::endl;  // Should be ~1.0
```

## Export ONNX Model

```bash
cd /workspaces/src/JIST

# Standard export
python export_to_onnx.py \
    --model_path JIST_r18_512_seqgem.pth \
    --output_dir models \
    --seq_length 5 \
    --img_shape 288 512 \
    --simplify

# With verification
python export_to_onnx.py \
    --model_path JIST_r18_512_seqgem.pth \
    --output_dir models \
    --seq_length 5 \
    --img_shape 288 512 \
    --simplify \
    --verify
```

## Build & Test

```bash
# Build
cd /workspaces/src/xfeat-cpp/build
cmake .. && make -j$(nproc)

# Run example
./examples/jist_example model.onnx images/

# Run tests
./tests/test_jist_onnx
```

## Integration with Other xfeat-cpp Components

### With XFeat for Geometric Verification
```cpp
xfeat::XFeatONNX xfeat(env, xfeat_params);
xfeat::JistONNX jist(env, jist_params);

// Loop closure detection
if (similarity > 0.85f) {
    // Geometric verification with XFeat
    auto matches = xfeat.match(current_image, database_image);
    if (matches.size() > 50) {
        // Confirmed loop closure
    }
}
```

### With NetVLAD for Hybrid Descriptor
```cpp
xfeat::NetVLADONNX netvlad(env, netvlad_params);
xfeat::JistONNX jist(env, jist_params);

// Compare both descriptors
cv::Mat jist_desc = jist.infer(sequence);
cv::Mat netvlad_desc = netvlad.transform(features);

// Combine or compare
```

## Similarity Thresholds

| Scenario | Threshold | Notes |
|----------|-----------|-------|
| Loop closure | 0.80-0.90 | Conservative |
| Place recognition | 0.70-0.80 | More relaxed |
| Geometric verification | 0.85+ | After feature matching |
| Same place, different time | 0.75-0.85 | Lighting changes |

## Memory Usage

Approximate memory per sequence:
- Input tensor: `batch * seq_length * 3 * H * W * 4 bytes`
- Example (batch=1, seq=5, 288x512): ~4.4 MB
- GPU memory: ~500 MB (model + workspace)
- Database: `N * descriptor_dim * 4 bytes` (N = num descriptors)

## Performance Tips

1. **Batch when possible**: 5-10x speedup over sequential
2. **Use GPU**: 10-20x speedup over CPU
3. **Stride selection**: 
   - stride=1: Dense, more accurate, slower
   - stride=seq_length: Fast, sufficient for most cases
4. **Image resolution**: Lower = faster but less accurate
5. **FP16 TensorRT**: 2x faster than ONNX (if available)

## Example: Complete VPR Pipeline

```cpp
// Setup
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VPR");
xfeat::JistONNX jist(env, params);
xfeat::JistDatabase map_db;

// Phase 1: Build map
map_db.load("map_database.yml");

// Phase 2: Online localization
jist.reset_buffer();
int frame_count = 0;

for (const auto& frame : camera_stream) {
    cv::Mat descriptor;
    if (jist.add_frame(frame, descriptor)) {
        // Query database
        std::vector<int> indices;
        std::vector<float> sims;
        map_db.query_top_k(descriptor, 5, indices, sims);
        
        // Check best match
        if (sims[0] > 0.85f) {
            std::cout << "Loop closure: frame " << frame_count 
                      << " matches map index " << indices[0] 
                      << " (sim=" << sims[0] << ")" << std::endl;
        }
    }
    frame_count++;
}
```

---

For detailed documentation, see [JIST_README.md](JIST_README.md)
