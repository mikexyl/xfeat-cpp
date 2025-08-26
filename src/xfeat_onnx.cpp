#include "xfeat-cpp/xfeat_onnx.h"

#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>

#include <iostream>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>

#include "xfeat-cpp/helpers.h"

using namespace xfeat;

cv::Mat normalizePerPixel(const cv::Mat& M1) {
  CV_Assert(M1.type() == CV_32FC(M1.channels()));
  std::cout << "Normalizing per pixel..." << std::endl;
  int C = M1.channels();
  int H = M1.rows;
  int W = M1.cols;

  // Split into channels
  std::vector<cv::Mat> chans;
  cv::split(M1, chans);  // each (H x W), float

  std::cout << "Channels split." << std::endl;

  // Compute squared sum across channels
  cv::Mat sqsum = cv::Mat::zeros(H, W, CV_32F);
  for (int c = 0; c < C; ++c) {
    cv::Mat tmp;
    cv::multiply(chans[c], chans[c], tmp);  // elementwise square
    sqsum += tmp;
  }

  std::cout << "Squared sum computed." << std::endl;

  // sqrt + epsilon
  cv::sqrt(sqsum, sqsum);
  sqsum += 1e-8f;

  std::cout << "Sqrt and epsilon added." << std::endl;

  // Divide each channel by the norm
  for (int c = 0; c < C; ++c) {
    cv::divide(chans[c], sqsum, chans[c]);
  }

  std::cout << "Channels normalized." << std::endl;

  // Merge back
  cv::Mat M1_normed;
  cv::merge(chans, M1_normed);
  std::cout << "Merged normalized channels." << std::endl;
  return M1_normed;
}

XFeatONNX::XFeatONNX(Ort::Env& env,
                     const std::string& xfeat_path,
                     const std::string& interp_bilinear_path,
                     const std::string& interp_bicubic_path,
                     const std::string& interp_nearest_path,
                     bool use_gpu,
                     int nkpts,
                     MatcherType matcher_type,
                     int anms,
                     int nkpts_before_anms,
                     int keypoint_detection,
                     std::unique_ptr<LighterGlueOnnx> lighterglue)
    : xfeat_session_(nullptr),
      interp_bilinear_session_(nullptr),
      interp_bicubic_session_(nullptr),
      interp_nearest_session_(nullptr),
      matcher_type_(matcher_type),
      anms_(anms),
      nkpts_before_anms_(nkpts_before_anms),
      keypoint_detection_(keypoint_detection),
      lighterglue_(std::move(lighterglue)) {
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  if (use_gpu) {
    std::cout << "Attempting to use GPU for ONNX Runtime." << std::endl;

    auto available_providers = Ort::GetAvailableProviders();
    bool cuda_available = false;

    // print available providers
    std::cout << "Available ONNX Runtime providers: ";

    for (const auto& provider : available_providers) {
      std::cout << provider << " ";
      if (provider == "CUDAExecutionProvider") {
        cuda_available = true;
      }
    }
    std::cout << std::endl;

    if (!cuda_available) {
      std::cerr << "Error: CUDAExecutionProvider is not available. Terminating." << std::endl;
      throw std::runtime_error("CUDAExecutionProvider not found.");
    }

    // const auto& api = Ort::GetApi();
    // OrtTensorRTProviderOptionsV2* tensorrt_options;
    // Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

    // // Append the V2 TensorRT provider
    // session_options_.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

    OrtCUDAProviderOptions cuda_options{};
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }

  xfeat_session_ = Ort::Session(env, xfeat_path.c_str(), session_options_);
  interp_bilinear_session_ = Ort::Session(env, interp_bilinear_path.c_str(), session_options_);
  interp_bicubic_session_ = Ort::Session(env, interp_bicubic_path.c_str(), session_options_);
  interp_nearest_session_ = Ort::Session(env, interp_nearest_path.c_str(), session_options_);

  // Get input dimensions from the xfeat model
  auto input_node_names = xfeat_session_.GetInputNames();
  auto input_node_dims = xfeat_session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  input_height_ = static_cast<int>(input_node_dims[2]);
  input_width_ = static_cast<int>(input_node_dims[3]);
  // Get input names for interpolator models (assuming they are consistent)
  auto interp_input_node_names = interp_nearest_session_.GetInputNames();
  interp_input_name1_ = interp_input_node_names[0];
  interp_input_name2_ = interp_input_node_names[1];
  std::cout << "ONNX models loaded." << std::endl;
  std::cout << "Input Dims: H=" << input_height_ << ", W=" << input_width_ << std::endl;

  if (matcher_type_ == MatcherType::LIGHTERGLUE) {
    std::cout << "Using LIGHTERGLUE matcher type." << std::endl;
    // Load LighterGlue model if specified
    if (not lighterglue_) {
      throw std::runtime_error("LighterGlue model path must be provided for LIGHTERGLUE matcher type.");
    }
  } else {
    lighterglue_.reset();
  }

  if (matcher_type_ == MatcherType::GPU_BF) {
    std::cout << "Using GPU BF matcher." << std::endl;
    gpu_matcher_ = std::make_unique<CuMatcher>();
    gpu_matcher_->init(nkpts, nkpts, 64);  // Assuming 128 is the descriptor dimension
  } else {
    gpu_matcher_.reset();
  }

  // TODO(mike): add manully triggered warmup
}

// Placeholder for preprocess_image
std::tuple<cv::Mat, float, float> XFeatONNX::preprocess_image(const cv::Mat& image) {
  // check if image empty
  cv::Mat input_image = image;
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  cv::resize(input_image, input_image, cv::Size(input_width_, input_height_));

  input_image.convertTo(input_image, CV_32F, 1.0 / 255.0);

  // Convert HWC to CHW and batch dimension: (1, 3, H, W)
  std::vector<cv::Mat> chw;
  cv::split(input_image, chw);  // chw[0]=C0, chw[1]=C1, chw[2]=C2, each HxW

  cv::Mat input_tensor(1, 3 * input_height_ * input_width_, CV_32F);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(input_tensor.ptr<float>(0) + c * input_height_ * input_width_,
                chw[c].ptr<float>(),
                input_height_ * input_width_ * sizeof(float));
  }
  // Reshape to (1, 3, H, W)
  input_tensor = input_tensor.reshape(1, {1, 3, input_height_, input_width_});

  float resize_rate_w = static_cast<float>(image.cols) / input_width_;
  float resize_rate_h = static_cast<float>(image.rows) / input_height_;

  auto result = std::make_tuple(input_tensor, resize_rate_w, resize_rate_h);

  return result;
}

// Implemented get_kpts_heatmap
cv::Mat XFeatONNX::get_kpts_heatmap(const Ort::Value& kpts_tensor,  // Should be Ort::Value
                                    float softmax_temp) {
  // Extract shape and data
  auto shape = kpts_tensor.GetTensorTypeAndShapeInfo().GetShape();
  // Expect shape [1, 65, 44, 80]
  if (shape.size() != 4 || shape[1] != 65) {
    throw std::runtime_error("get_kpts_heatmap: input tensor must have shape (1, 65, 44, 80)");
  }
  int B = static_cast<int>(shape[0]);
  int C = static_cast<int>(shape[1]);  // 65
  int H = static_cast<int>(shape[2]);  // 44
  int W = static_cast<int>(shape[3]);  // 80
  const float* data = kpts_tensor.GetTensorData<float>();

  // Pre-calculate dimensions for efficiency
  const int HW = H * W;
  const int total_size = B * C * HW;
  const int scores_size = B * 64 * HW;

  // Create working buffer for exponentials (avoid copying original data)
  std::vector<float> kpts_exp(total_size);

  // Combined softmax operation: multiply by temp and exponentiate in single parallel loop
  tbb::parallel_for(tbb::blocked_range<size_t>(0, total_size), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      kpts_exp[i] = std::exp(data[i] * softmax_temp);
    }
  });

  // Sum over channel axis (axis=1) with parallelization
  std::vector<float> sum_exp(B * HW);

  // Parallelize over spatial dimensions (H*W)
  tbb::parallel_for(tbb::blocked_range<int>(0, B * HW), [&](const tbb::blocked_range<int>& range) {
    for (int hw_idx = range.begin(); hw_idx != range.end(); ++hw_idx) {
      int b = hw_idx / HW;
      int hw = hw_idx % HW;

      float sum = 0.0f;
      // Vectorized sum over channels - better cache locality
      const float* base_ptr = &kpts_exp[b * C * HW + hw];
      for (int c = 0; c < C; ++c) {
        sum += base_ptr[c * HW];
      }
      sum_exp[hw_idx] = sum;
    }
  });

  // Normalize and keep only first 64 channels with parallelization
  std::vector<float> scores(scores_size);

  tbb::parallel_for(tbb::blocked_range<int>(0, B * 64 * HW), [&](const tbb::blocked_range<int>& range) {
    for (int idx = range.begin(); idx != range.end(); ++idx) {
      int b = idx / (64 * HW);
      int c = (idx / HW) % 64;
      int hw = idx % HW;

      float v = kpts_exp[b * C * HW + c * HW + hw];
      float s = sum_exp[b * HW + hw];
      scores[idx] = v / s;
    }
  });

  // Direct output to cv::Mat to avoid extra copy
  const int out_H = H * 8;
  const int out_W = W * 8;
  cv::Mat out_heatmap(out_H, out_W, CV_32F);
  float* out_data = out_heatmap.ptr<float>();

  // Rearrange with optimized memory access pattern and parallelization
  // Process in blocks to improve cache locality
  tbb::parallel_for(tbb::blocked_range<int>(0, H * W), [&](const tbb::blocked_range<int>& range) {
    for (int hw_idx = range.begin(); hw_idx != range.end(); ++hw_idx) {
      int h = hw_idx / W;
      int w = hw_idx % W;

      // Calculate base indices
      const int out_base_h = h * 8;
      const int out_base_w = w * 8;
      const int scores_base = h * W + w;  // For B=1

      // Unroll the 8x8 block for better performance
      for (int gh = 0; gh < 8; ++gh) {
        const int out_row = out_base_h + gh;
        float* out_row_ptr = &out_data[out_row * out_W + out_base_w];

        for (int gw = 0; gw < 8; ++gw) {
          const int c = gh * 8 + gw;
          const int scores_idx = c * HW + scores_base;
          out_row_ptr[gw] = scores[scores_idx];
        }
      }
    }
  });
  return out_heatmap;
}

// Placeholder for nms
cv::Mat XFeatONNX::nms(const Ort::Value& heatmap_tensor,  // Should be Ort::Value
                       float threshold,
                       int kernel_size) {
  // Extract heatmap shape and data
  auto shape = heatmap_tensor.GetTensorTypeAndShapeInfo().GetShape();
  // Assume shape is [1, 1, H, W]
  int H = static_cast<int>(shape[2]);
  int W = static_cast<int>(shape[3]);
  const float* data = heatmap_tensor.GetTensorData<float>();
  // Convert to cv::Mat
  cv::Mat heatmap(H, W, CV_32F);
  std::memcpy(heatmap.data, data, H * W * sizeof(float));

  // Apply max filter (dilate)
  cv::Mat max_filt;
  cv::dilate(heatmap, max_filt, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size)));

  // Find local maxima: (heatmap == max_filt) & (heatmap > threshold)
  cv::Mat mask = (heatmap == max_filt) & (heatmap > threshold);

  // Get coordinates of keypoints
  std::vector<cv::Point> keypoints;
  cv::findNonZero(mask, keypoints);

  // Convert to Nx2 float matrix
  cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    kpt_mat.at<float>(i, 0) = static_cast<float>(keypoints[i].x);
    kpt_mat.at<float>(i, 1) = static_cast<float>(keypoints[i].y);
  }
  return kpt_mat;
}

// Add overload for nms that accepts cv::Mat
cv::Mat XFeatONNX::nms(const cv::Mat& heatmap, float threshold, int kernel_size) {
  // Apply max filter (dilate)
  cv::Mat max_filt;
  cv::dilate(heatmap, max_filt, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size)));
  // Find local maxima: (heatmap == max_filt) & (heatmap > threshold)
  cv::Mat mask = (heatmap == max_filt) & (heatmap > threshold);
  // Get coordinates of keypoints
  std::vector<cv::Point> keypoints;
  cv::findNonZero(mask, keypoints);
  // Convert to Nx2 float matrix
  cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    kpt_mat.at<float>(i, 0) = static_cast<float>(keypoints[i].x);
    kpt_mat.at<float>(i, 1) = static_cast<float>(keypoints[i].y);
  }
  return kpt_mat;
}

DetectionResult XFeatONNX::detect_and_compute(Ort::Session& session,
                                              cv::Mat image,
                                              int top_k,
                                              cv::Mat* heatmap,
                                              cv::Mat* M1,
                                              cv::Mat* x_prep,
                                              std::vector<cv::Vec2d>* std,
                                              int anms,
                                              int nkpts_before_anms,
                                              int keypoint_detection) {
  // if image is in gray scale, convert to BGR
  cv::Mat color_image, gray_image;
  if (image.channels() == 1) {
    gray_image = image;
    cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);
  } else {
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    color_image = image;
  }
  auto [input_tensor, resize_rate_w, resize_rate_h] = preprocess_image(color_image);
  if (x_prep) {
    *x_prep = input_tensor;  // Copy preprocessed image
  }
  assert(input_tensor.type() == CV_32FC3);
  assert(input_tensor.isContinuous());

  auto input_node_names = session.GetInputNames();
  auto output_node_names = session.GetOutputNames();
  std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
  Ort::AllocatorWithDefaultOptions allocator;
  size_t input_tensor_size = 1 * 3 * input_height_ * input_width_;
  Ort::Value input_ort_tensor = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), (float*)input_tensor.data, input_tensor_size, input_shape.data(), input_shape.size());
  std::vector<const char*> input_names_char;
  for (const auto& name : input_node_names) {
    input_names_char.push_back(name.c_str());
  }
  std::vector<const char*> output_names_char;
  for (const auto& name : output_node_names) {
    output_names_char.push_back(name.c_str());
  }
  std::vector<Ort::Value> output_tensors;
  output_tensors = session.Run(Ort::RunOptions{nullptr},
                               input_names_char.data(),
                               &input_ort_tensor,
                               1,
                               output_names_char.data(),
                               output_names_char.size());

  // M1: Feature map, K1: Keypoint logits
  const Ort::Value& M1_tensor = output_tensors[0];
  const Ort::Value& K1_tensor = output_tensors[1];
  // Print output tensor shapes for debugging
  auto M1_shape = M1_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto K1_shape = K1_tensor.GetTensorTypeAndShapeInfo().GetShape();
  // Use GetTensorData for read-only access
  const float* M1_data = M1_tensor.GetTensorData<float>();
  const float* K1_data = K1_tensor.GetTensorData<float>();

  // L2 normalize M1
  std::vector<int64_t> M1_shape_vec = M1_tensor.GetTensorTypeAndShapeInfo().GetShape();
  int B = M1_shape_vec[0];
  int C = M1_shape_vec[1];
  int H = M1_shape_vec[2];
  int W = M1_shape_vec[3];

  if (M1) {
    // Ensure B == 1
    assert(B == 1 && "Only batch size 1 is supported");
    // Create a 3D OpenCV Mat with shape C x H x W
    int sizes[] = {B, static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)};
    *M1 = cv::Mat(4, sizes, CV_32F, const_cast<float*>(M1_data));  // Wrap in Mat without copying
  }

  cv::Mat _M1(C, H * W, CV_32F, (void*)M1_data);  // (C, H*W)
  for (int i = 0; i < H * W; ++i) {
    float norm = 0.0f;
    for (int c = 0; c < C; ++c) norm += _M1.at<float>(c, i) * _M1.at<float>(c, i);
    norm = std::sqrt(norm) + 1e-8f;
    for (int c = 0; c < C; ++c) _M1.at<float>(c, i) /= norm;
  }
  // Reshape back to (C, H, W)
  std::vector<cv::Mat> M1_channels;
  for (int c = 0; c < C; ++c) {
    M1_channels.push_back(cv::Mat(H, W, CV_32F, _M1.ptr<float>(c)));
  }
  cv::Mat M1_normed;
  cv::merge(M1_channels, M1_normed);  // (H, W, C)

  cv::Mat mkpts_mat;

  // Get heatmap K1h
  cv::Mat K1h = get_kpts_heatmap(K1_tensor);

  if (keypoint_detection == 0) {
    // Save heatmap for debugging
    if (heatmap) {
      *heatmap = K1h;  // Copy to output heatmap
    }

    // NMS on K1h (upsampled heatmap)
    mkpts_mat = nms(K1h, 0.05, 5);  // Pass K1h (cv::Mat), not K1_tensor
  } else {
    // run gftt on the original image to get better keypoints
    std::vector<cv::KeyPoint> keypoints;
    // cv::goodFeaturesToTrack(gray_image, keypoints, nkpts_before_anms, 0.001, 20, noArray(), 3, false, 0.04);
    // std::vector<cv::KeyPoint> kpts_fast, kpts_agast;

    // FAST (threshold ~10–30; enable nonmax suppression)
    auto fast = cv::FastFeatureDetector::create(20, /*nonmax*/ true, cv::FastFeatureDetector::TYPE_9_16);
    fast->detect(gray_image, keypoints);

    // resize the keypoints
    for (auto& kp : keypoints) {
      kp.pt.x /= resize_rate_w;
      kp.pt.y /= resize_rate_h;
    }

    // populate mkpts_mat
    mkpts_mat = cv::Mat(keypoints.size(), 2, CV_32F);
    for (size_t i = 0; i < keypoints.size(); ++i) {
      mkpts_mat.at<float>(i, 0) = keypoints[i].pt.x;
      mkpts_mat.at<float>(i, 1) = keypoints[i].pt.y;
    }
  }

  if (anms == 1) {
    std::vector<cv::KeyPoint> keypoints;
    for (int i = 0; i < mkpts_mat.rows; ++i) {
      float x = mkpts_mat.at<float>(i, 0);
      float y = mkpts_mat.at<float>(i, 1);
      keypoints.emplace_back(cv::KeyPoint(x, y, 1));
    }

    std::vector<cv::KeyPoint> anms_keypoints = anms::Ssc(keypoints, top_k, 0.1, 640, 480);
    mkpts_mat = cv::Mat(anms_keypoints.size(), 2, CV_32F);
    for (size_t i = 0; i < anms_keypoints.size(); ++i) {
      mkpts_mat.at<float>(i, 0) = anms_keypoints[i].pt.x;
      mkpts_mat.at<float>(i, 1) = anms_keypoints[i].pt.y;
    }
  }

  // Interpolate for scores (nearest and bilinear)
  // Prepare ONNX input for interpolators
  std::vector<int64_t> kpt_shape = {1, mkpts_mat.rows, 2};
  size_t mkpts_numel = 1 * mkpts_mat.rows * 2;
  if (mkpts_mat.total() != mkpts_numel) {
    throw std::runtime_error("mkpts_mat buffer size does not match shape");
  }
  Ort::Value mkpts_tensor = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), (float*)mkpts_mat.ptr<float>(), mkpts_numel, kpt_shape.data(), kpt_shape.size());
  std::vector<int64_t> K1h_shape = {1, 1, K1h.rows, K1h.cols};
  size_t K1h_numel = 1 * 1 * K1h.rows * K1h.cols;
  if (K1h.total() != K1h_numel) {
    throw std::runtime_error("K1h buffer size does not match shape");
  }
  Ort::Value K1h_tensor = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), (float*)K1h.ptr<float>(), K1h_numel, K1h_shape.data(), K1h_shape.size());
  // Nearest
  std::vector<const char*> interp_input_names = {interp_input_name1_.c_str(), interp_input_name2_.c_str()};
  std::vector<Ort::Value> interp_inputs;
  interp_inputs.push_back(std::move(K1h_tensor));
  interp_inputs.push_back(std::move(mkpts_tensor));
  // Fetch actual output node names for interpolators
  auto interp_nearest_output_names = interp_nearest_session_.GetOutputNames();
  std::vector<const char*> interp_nearest_output_names_char;
  for (const auto& name : interp_nearest_output_names) interp_nearest_output_names_char.push_back(name.c_str());
  auto interp_bilinear_output_names = interp_bilinear_session_.GetOutputNames();
  std::vector<const char*> interp_bilinear_output_names_char;
  for (const auto& name : interp_bilinear_output_names) interp_bilinear_output_names_char.push_back(name.c_str());
  std::vector<Ort::Value> nearest_out;
  nearest_out = interp_nearest_session_.Run(Ort::RunOptions{nullptr},
                                            interp_input_names.data(),
                                            interp_inputs.data(),
                                            2,
                                            interp_nearest_output_names_char.data(),
                                            interp_nearest_output_names_char.size());

  std::vector<Ort::Value> bilinear_out;
  bilinear_out = interp_bilinear_session_.Run(Ort::RunOptions{nullptr},
                                              interp_input_names.data(),
                                              interp_inputs.data(),
                                              2,
                                              interp_bilinear_output_names_char.data(),
                                              interp_bilinear_output_names_char.size());

  auto time_score_computation = std::chrono::high_resolution_clock::now();
  float* nearest_scores = nearest_out[0].GetTensorMutableData<float>();
  float* bilinear_scores = bilinear_out[0].GetTensorMutableData<float>();
  cv::Mat scores_mat(mkpts_mat.rows, 1, CV_32F);
  for (int i = 0; i < mkpts_mat.rows; ++i) {
    scores_mat.at<float>(i, 0) = nearest_scores[i] * bilinear_scores[i];
  }
  // Set invalid keypoints to -1
  for (int i = 0; i < mkpts_mat.rows; ++i) {
    if (mkpts_mat.at<float>(i, 0) == 0 && mkpts_mat.at<float>(i, 1) == 0) {
      scores_mat.at<float>(i, 0) = -1.0f;
    }
  }
  // Sort by scores and select top_k
  std::vector<int> idxs(mkpts_mat.rows);
  std::iota(idxs.begin(), idxs.end(), 0);
  std::sort(
      idxs.begin(), idxs.end(), [&](int a, int b) { return scores_mat.at<float>(a, 0) > scores_mat.at<float>(b, 0); });
  std::vector<cv::Point2f> topk_kpts;
  std::vector<float> topk_scores;
  for (int i = 0; i < (int)idxs.size(); ++i) {
    topk_kpts.push_back(cv::Point2f(mkpts_mat.at<float>(idxs[i], 0), mkpts_mat.at<float>(idxs[i], 1)));
    topk_scores.push_back(scores_mat.at<float>(idxs[i], 0));
  }
  // Interpolate for features (bicubic)
  std::vector<int64_t> topk_shape = {1, (int64_t)topk_kpts.size(), 2};
  size_t topk_numel = 1 * topk_kpts.size() * 2;
  cv::Mat topk_kpts_mat(topk_kpts.size(), 2, CV_32F);
  for (int i = 0; i < topk_kpts.size(); ++i) {
    topk_kpts_mat.at<float>(i, 0) = topk_kpts[i].x;
    topk_kpts_mat.at<float>(i, 1) = topk_kpts[i].y;
  }
  if (topk_kpts_mat.total() != topk_numel) {
    throw std::runtime_error("topk_kpts_mat buffer size does not match shape");
  }
  Ort::Value topk_kpts_tensor = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), (float*)topk_kpts_mat.ptr<float>(), topk_numel, topk_shape.data(), topk_shape.size());

  std::vector<int64_t> M1_shape_interp = {1, C, H, W};
  size_t M1_numel = 1 * C * H * W;
  // Convert M1_normed (H, W, C) to (C, H, W) contiguous buffer
  std::vector<float> M1_chw(C * H * W);
  std::vector<cv::Mat> M1_split;
  cv::split(M1_normed, M1_split);  // M1_split: C x (H,W)
  for (int c = 0; c < C; c++) {
    std::memcpy(&M1_chw[c * H * W], M1_split[c].ptr<float>(), H * W * sizeof(float));
  }
  if (M1_chw.size() != M1_numel) {
    throw std::runtime_error("M1_chw buffer size does not match shape");
  }

  Ort::Value M1_tensor_interp = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), M1_chw.data(), M1_numel, M1_shape_interp.data(), M1_shape_interp.size());
  std::vector<Ort::Value> bicubic_inputs;
  bicubic_inputs.push_back(std::move(M1_tensor_interp));
  bicubic_inputs.push_back(std::move(topk_kpts_tensor));
  // Fetch actual output node names for bicubic interpolator
  auto interp_bicubic_output_names = interp_bicubic_session_.GetOutputNames();
  std::vector<const char*> interp_bicubic_output_names_char;
  for (const auto& name : interp_bicubic_output_names) interp_bicubic_output_names_char.push_back(name.c_str());
  std::vector<Ort::Value> feats_out;
  feats_out = interp_bicubic_session_.Run(Ort::RunOptions{nullptr},
                                          interp_input_names.data(),
                                          bicubic_inputs.data(),
                                          2,
                                          interp_bicubic_output_names_char.data(),
                                          interp_bicubic_output_names_char.size());

  float* feats_ptr = feats_out[0].GetTensorMutableData<float>();
  int feat_dim = C;
  int n_kpts = topk_kpts.size();
  cv::Mat feats_mat(n_kpts, feat_dim, CV_32F, feats_ptr);
  // L2 normalize feats
  for (int i = 0; i < n_kpts; ++i) {
    float norm = 0.0f;
    for (int j = 0; j < feat_dim; ++j) norm += feats_mat.at<float>(i, j) * feats_mat.at<float>(i, j);
    norm = std::sqrt(norm) + 1e-8f;
    for (int j = 0; j < feat_dim; ++j) feats_mat.at<float>(i, j) /= norm;
  }
  // Scale keypoints
  for (auto& pt : topk_kpts) {
    pt.x *= resize_rate_w;
    pt.y *= resize_rate_h;
  }
  // Filter valid keypoints (score > 0)
  std::vector<cv::KeyPoint> valid_kpts;
  std::vector<float> valid_scores;
  std::vector<cv::Mat> valid_feats;
  for (int i = 0; i < n_kpts; ++i) {
    if (topk_scores[i] > 0) {
      valid_kpts.push_back({topk_kpts[i], 1});
      valid_scores.push_back(topk_scores[i]);
      valid_feats.push_back(feats_mat.row(i));
    }
  }
  cv::Mat valid_kpts_mat(valid_kpts.size(), 2, CV_32F);
  for (int i = 0; i < valid_kpts.size(); ++i) {
    valid_kpts_mat.at<float>(i, 0) = valid_kpts[i].pt.x;
    valid_kpts_mat.at<float>(i, 1) = valid_kpts[i].pt.y;
  }
  cv::Mat valid_scores_mat(valid_scores.size(), 1, CV_32F, valid_scores.data());
  cv::Mat valid_feats_mat(valid_feats.size(), feat_dim, CV_32F);
  for (int i = 0; i < valid_feats.size(); ++i) {
    valid_feats[i].copyTo(valid_feats_mat.row(i));
  }
  DetectionResult det;
  det.keypoints = valid_kpts_mat;
  det.scores = valid_scores_mat;
  det.descriptors = valid_feats_mat;

  if (std) {
    std->clear();
    *std = computeUncertaintySobel(K1h, valid_kpts);
    // std times resize rates
    for (auto& s : *std) {
      s[0] *= resize_rate_w;
      s[1] *= resize_rate_h;
    }
  }
  return det;  // Placeholder
}

// match_mkpts: rewritten to match the logic of the Python version
std::vector<std::vector<int>> XFeatONNX::match_mkpts_bf(const cv::Mat& feats1,
                                                        const cv::Mat& feats2,
                                                        float min_cossim) {
  int N1 = feats1.rows;
  int N2 = feats2.rows;
  auto t0 = std::chrono::high_resolution_clock::now();
  cv::Mat cossim = feats1 * feats2.t();    // (N1, N2)
  cv::Mat cossim_t = feats2 * feats1.t();  // (N2, N1)
  auto t1 = std::chrono::high_resolution_clock::now();
  std::stringstream ss;
  // print matrix size
  for (int dim = 0; dim < feats1.dims; ++dim) {
    ss << feats1.size[dim];
    if (dim < feats1.dims - 1) ss << "x";
  }
  std::cout << "Cossim matrix size: " << ss.str() << std::endl;
  std::cout << "Cossim computation time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms"
            << std::endl;

  std::vector<int> match12(N1), match21(N2);
  tbb::parallel_for(0, N1, [&](int i) {
    double maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(cossim.row(i), nullptr, &maxVal, nullptr, &maxLoc);
    match12[i] = maxLoc.x;
  });
  tbb::parallel_for(0, N2, [&](int i) {
    double maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(cossim_t.row(i), nullptr, &maxVal, nullptr, &maxLoc);
    match21[i] = maxLoc.x;
  });

  std::vector<std::vector<int>> idx(feats1.rows, std::vector<int>{});
  tbb::mutex idx_mutex;
  tbb::parallel_for(0, N1, [&](int i) {
    int j = match12[i];
    if (j >= 0 && j < N2 && match21[j] == i) {
      if (min_cossim > 0) {
        // Find max value in cossim.row(i) manually
        float max_cossim = cossim.at<float>(i, 0);
        for (int k = 1; k < cossim.cols; ++k) {
          if (cossim.at<float>(i, k) > max_cossim) {
            max_cossim = cossim.at<float>(i, k);
          }
        }
        if (max_cossim > min_cossim) {
          tbb::mutex::scoped_lock lock(idx_mutex);
          idx[i].push_back(j);
        }
      } else {
        tbb::mutex::scoped_lock lock(idx_mutex);
        idx[i].push_back(j);
      }
    }
  });
  return idx;
}

std::vector<cv::DMatch> XFeatONNX::match(cv::Mat image1,
                                         cv::Mat image2,
                                         int top_k,
                                         float min_cossim,
                                         cv::Mat* heatmap1,
                                         cv::Mat* heatmap2,
                                         TimingStats* timing_stats) {
  auto t0 = std::chrono::high_resolution_clock::now();
  auto result1 = detect_and_compute(xfeat_session_, image1, top_k, heatmap1);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto result2 = detect_and_compute(xfeat_session_, image2, top_k, heatmap2);
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "detected keypoints in image1: " << result1.keypoints.rows << ", image2: " << result2.keypoints.rows
            << std::endl;

  auto match_start = std::chrono::high_resolution_clock::now();
  auto match_result = match(result1, result2, image1, min_cossim, timing_stats);
  auto match_end = std::chrono::high_resolution_clock::now();

  if (timing_stats) {
    (*timing_stats)["detect1"] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    (*timing_stats)["detect2"] = std::chrono::duration<double, std::milli>(t2 - t1).count();
    (*timing_stats)["match"] = std::chrono::duration<double, std::milli>(match_end - match_start).count();
    (*timing_stats)["total"] = std::chrono::duration<double, std::milli>(match_end - t0).count();
  }

  return match_result;
}

// Overload: match using DetectionResult directly
std::vector<cv::DMatch> XFeatONNX::match(const DetectionResult& result1,
                                         const DetectionResult& result2,
                                         cv::Mat image1,
                                         float min_sim,
                                         TimingStats* timing_stats) {
  if (result1.keypoints.empty() || result2.keypoints.empty()) {
    std::cerr << "Detection failed for one or both DetectionResults." << std::endl;
    return {};
  }

  std::vector<std::vector<int>> indexes;
  std::vector<int> best_index;
  std::vector<float> best_scores;

  std::vector<cv::Point2f> keypoints1, keypoints2;
  for (int i = 0; i < result1.keypoints.rows; ++i) {
    keypoints1.emplace_back(result1.keypoints.at<float>(i, 0), result1.keypoints.at<float>(i, 1));
  }
  for (int i = 0; i < result2.keypoints.rows; ++i) {
    keypoints2.emplace_back(result2.keypoints.at<float>(i, 0), result2.keypoints.at<float>(i, 1));
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  switch (matcher_type_) {
    case MatcherType::BF:
      indexes = match_mkpts_bf(result1.descriptors, result2.descriptors, min_sim);
      break;
    case MatcherType::FLANN:
      indexes = match_mkpts_flann(result1.descriptors, result2.descriptors, min_sim);
      break;
    case MatcherType::GPU_BF:
      if (!gpu_matcher_) {
        throw std::runtime_error("GPU BF matcher is not initialized.");
      }
      indexes = gpu_matcher_->match_mkpts(result1.descriptors, result2.descriptors, min_sim);
      break;
    case MatcherType::LIGHTERGLUE:
      if (!lighterglue_) {
        throw std::runtime_error("LighterGlue matcher is not initialized.");
      }
      std::array<float, 2> image_size0{static_cast<float>(input_width_), static_cast<float>(input_height_)};
      std::array<float, 2> image_size1{static_cast<float>(input_width_), static_cast<float>(input_height_)};
      indexes = lighterglue_->match(result1, image_size0, result2, image_size1, min_sim);
      break;
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  if (timing_stats) {
    (*timing_stats)["match_mkpts"] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  int num_matched =
      std::count_if(indexes.begin(), indexes.end(), [](const std::vector<int>& idx) { return !idx.empty(); });

  t0 = std::chrono::high_resolution_clock::now();
  // Select matched keypoints
  std::vector<int> matched_indices1, matched_indices2;
  cv::Mat mkpts1(num_matched, 2, CV_32F);
  cv::Mat mkpts2(num_matched, 2, CV_32F);
  std::set<int> matched_indices2_set;
  std::cout << "Number of matched keypoints: " << num_matched << std::endl;
  for (size_t i = 0; i < indexes.size(); ++i) {
    if (indexes[i].empty()) {
      continue;
    }
    matched_indices2_set.insert(indexes[i].begin(), indexes[i].end());
    if (indexes[i].size() > 1) {
      throw std::runtime_error("Multiple matches found for a single keypoint, which is not supported.");
    }
    // Take the first match
    int j = indexes[i][0];
    if (j < 0 || j >= result2.keypoints.rows) {
      continue;  // Invalid index
    }
    mkpts1.at<float>(i, 0) = result1.keypoints.at<float>(i, 0);
    mkpts1.at<float>(i, 1) = result1.keypoints.at<float>(i, 1);
    mkpts2.at<float>(i, 0) = result2.keypoints.at<float>(j, 0);
    mkpts2.at<float>(i, 1) = result2.keypoints.at<float>(j, 1);
    matched_indices1.push_back(i);
    matched_indices2.push_back(j);
  }

  // Filter matches using homography (RANSAC)
  cv::Mat H;
  auto inliers = calc_warp_corners_and_matches(mkpts1, mkpts2, &H);

  std::vector<int> inlier_indices1, inlier_indices2;
  for (size_t i = 0; i < inliers.size(); ++i) {
    if (inliers[i] > 0) {  // Inlier
      inlier_indices1.push_back(matched_indices1[i]);
      inlier_indices2.push_back(matched_indices2[i]);
    }
  }

  std::vector<cv::DMatch> matches;
  if (not H.empty() and matcher_type_ == MatcherType::GPU_BF) {
    std::cout << "Rematching unmatched keypoints using GPU matcher." << std::endl;
    std::vector<cv::Point2f> kpts1_warped;
    cv::perspectiveTransform(keypoints1, kpts1_warped, H);
    auto [rematch_id1, rematch_id2] = gpu_matcher_->match_mkpts_local(
        result1.descriptors, result2.descriptors, kpts1_warped, keypoints2, 5, min_sim * 0.6);

    for (int i = 0; i < rematch_id1.size(); ++i) {
      if (rematch_id1[i] >= 0 && rematch_id2[i] >= 0) {
        matches.emplace_back(rematch_id1[i], rematch_id2[i], 0.0f);
      }
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  if (timing_stats) {
    (*timing_stats)["match"] = std::chrono::duration<double, std::milli>(t2 - t0).count();
  }

  return matches;
}

DetectionResult XFeatONNX::detect_and_compute(cv::Mat image,
                                              int top_k,
                                              cv::Mat* heatmap,
                                              cv::Mat* M1,
                                              cv::Mat* x_prep,
                                              std::vector<cv::Vec2d>* std) {
  return detect_and_compute(
      xfeat_session_, image, top_k, heatmap, M1, x_prep, std, anms_, nkpts_before_anms_, keypoint_detection_);
}

std::vector<std::vector<int>> XFeatONNX::match_mkpts_flann(const cv::Mat& feats1,
                                                           const cv::Mat& feats2,
                                                           float min_cossim) {
  // Implementation using FLANN matcher with mutual nearest neighbor and min_cossim threshold
  cv::Mat desc1 = feats1;
  if (desc1.type() != CV_32F) desc1.convertTo(desc1, CV_32F);
  cv::Mat desc2 = feats2;
  if (desc2.type() != CV_32F) desc2.convertTo(desc2, CV_32F);

  cv::FlannBasedMatcher matcher;
  // Match descriptors 1->2 and 2->1
  std::vector<cv::DMatch> matches12;
  matcher.match(desc1, desc2, matches12);
  std::vector<cv::DMatch> matches21;
  matcher.match(desc2, desc1, matches21);

  std::vector<std::vector<int>> idx(desc1.rows, std::vector<int>{});

  // Convert min_cossim to a distance threshold
  float maxDist = 0.0f;
  if (min_cossim > 0.0f) {
    maxDist = std::sqrt(std::max(0.0f, 2.0f * (1.0f - min_cossim)));
  }

  // Mutual nearest neighbor check
  for (size_t i = 0; i < matches12.size(); ++i) {
    int j = matches12[i].trainIdx;
    if (j >= 0 && j < (int)matches21.size() && matches21[j].trainIdx == (int)i) {
      float dist = matches12[i].distance;
      if (min_cossim > 0.0f) {
        if (dist <= maxDist) {
          idx[matches12[i].queryIdx].push_back(matches12[i].trainIdx);
        }
      } else {
        idx[matches12[i].queryIdx].push_back(matches12[i].trainIdx);
      }
    }
  }
  return idx;
}

XFeatONNX::XFeatONNX(Ort::Env& env, const Params& params, std::unique_ptr<LighterGlueOnnx> lighterglue)
    : XFeatONNX(env,
                params.xfeat_path,
                params.interp_bilinear_path,
                params.interp_bicubic_path,
                params.interp_nearest_path,
                params.use_gpu,
                params.nkpts,
                params.matcher_type,
                params.anms,
                params.nkpts_before_anms,
                params.keypoint_detection,
                std::move(lighterglue)) {}