#include "xfeat-cpp/xfeat_onnx.h"

#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>

#include <iostream>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace xfeat;

XFeatONNX::XFeatONNX(const std::string& xfeat_path,
                     const std::string& interp_bilinear_path,
                     const std::string& interp_bicubic_path,
                     const std::string& interp_nearest_path,
                     bool use_gpu,
                     MatcherType matcher_type,
                     const std::string& lighterglue_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "XFeatONNX"),
      xfeat_session_(nullptr),
      interp_bilinear_session_(nullptr),
      interp_bicubic_session_(nullptr),
      interp_nearest_session_(nullptr),
      matcher_type_(matcher_type) {
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  if (use_gpu) {
    std::cout << "Attempting to use GPU for ONNX Runtime." << std::endl;

    auto available_providers = Ort::GetAvailableProviders();
    bool cuda_available = false;
    for (const auto& provider : available_providers) {
      if (provider == "CUDAExecutionProvider") {
        cuda_available = true;
        break;
      }
    }

    if (!cuda_available) {
      std::cerr << "Error: CUDAExecutionProvider is not available. Terminating." << std::endl;
      throw std::runtime_error("CUDAExecutionProvider not found.");
    }

    OrtCUDAProviderOptions cuda_options{};
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }

  xfeat_session_ = Ort::Session(env_, xfeat_path.c_str(), session_options_);
  interp_bilinear_session_ = Ort::Session(env_, interp_bilinear_path.c_str(), session_options_);
  interp_bicubic_session_ = Ort::Session(env_, interp_bicubic_path.c_str(), session_options_);
  interp_nearest_session_ = Ort::Session(env_, interp_nearest_path.c_str(), session_options_);

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
    if (lighterglue_path.empty()) {
      throw std::runtime_error("LighterGlue model path must be provided for LIGHTERGLUE matcher type.");
    }
    lighterglue_ = std::make_unique<LighterGlueOnnx>(lighterglue_path, use_gpu);
  } else {
    lighterglue_.reset();
  }

  // TODO(mike): add manully triggered warmup
}

// Placeholder for preprocess_image
std::tuple<cv::Mat, float, float> XFeatONNX::preprocess_image(const cv::Mat& image) {
  cv::Mat input_image;
  cv::resize(image, input_image, cv::Size(input_width_, input_height_));
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

  return std::make_tuple(input_tensor.clone(), resize_rate_w, resize_rate_h);
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
  // Copy data to a contiguous array for easier manipulation
  std::vector<float> kpts(data, data + B * C * H * W);
  // Apply softmax_temp (multiply, not divide, to match Python)
  for (size_t i = 0; i < kpts.size(); ++i) {
    kpts[i] *= softmax_temp;
  }
  // Exponentiate
  for (size_t i = 0; i < kpts.size(); ++i) {
    kpts[i] = std::exp(kpts[i]);
  }
  // Sum over channel axis (axis=1)
  std::vector<float> sum_exp(B * H * W, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
          sum += kpts[((b * C + c) * H + h) * W + w];
        }
        sum_exp[(b * H + h) * W + w] = sum;
      }
    }
  }
  // Normalize and keep only first 64 channels
  std::vector<float> scores(B * 64 * H * W, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < 64; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          float v = kpts[((b * C + c) * H + h) * W + w];
          float s = sum_exp[(b * H + h) * W + w];
          scores[((b * 64 + c) * H + h) * W + w] = v / s;
        }
      }
    }
  }
  // Rearrange: (B, 64, H, W) -> (B, H, W, 8, 8)
  std::vector<float> heatmap(B * H * 8 * W * 8, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int gh = 0; gh < 8; ++gh) {
          for (int gw = 0; gw < 8; ++gw) {
            int c = gh * 8 + gw;
            float v = scores[((b * 64 + c) * H + h) * W + w];
            // (B, H, W, 8, 8)
            heatmap[((((b * H + h) * 8 + gh) * W + w) * 8 + gw)] = v;
          }
        }
      }
    }
  }
  // Reshape to (B, 1, H*8, W*8)
  int out_H = H * 8;
  int out_W = W * 8;
  // Only support B=1 for now
  cv::Mat out_heatmap(out_H, out_W, CV_32F);
  for (int oh = 0; oh < out_H; ++oh) {
    for (int ow = 0; ow < out_W; ++ow) {
      out_heatmap.at<float>(oh, ow) = heatmap[oh * out_W + ow];
    }
  }
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

DetectionResult XFeatONNX::detect_and_compute(Ort::Session& session, const cv::Mat& image, int top_k) {
  auto [input_tensor, resize_rate_w, resize_rate_h] = preprocess_image(image);

  auto input_node_names = session.GetInputNames();
  auto output_node_names = session.GetOutputNames();
  std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value input_ort_tensor = Ort::Value::CreateTensor<float>(allocator.GetInfo(),
                                                                (float*)input_tensor.data,
                                                                input_tensor.total() * input_tensor.channels(),
                                                                input_shape.data(),
                                                                input_shape.size());
  std::vector<const char*> input_names_char;
  for (const auto& name : input_node_names) {
    input_names_char.push_back(name.c_str());
  }
  std::vector<const char*> output_names_char;
  for (const auto& name : output_node_names) {
    output_names_char.push_back(name.c_str());
  }
  std::vector<Ort::Value> output_tensors;
  try {
    output_tensors = session.Run(Ort::RunOptions{nullptr},
                                 input_names_char.data(),
                                 &input_ort_tensor,
                                 1,
                                 output_names_char.data(),
                                 output_names_char.size());
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
    return {};
  }
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
  cv::Mat M1(C, H * W, CV_32F, (void*)M1_data);  // (C, H*W)
  for (int i = 0; i < H * W; ++i) {
    float norm = 0.0f;
    for (int c = 0; c < C; ++c) norm += M1.at<float>(c, i) * M1.at<float>(c, i);
    norm = std::sqrt(norm) + 1e-8f;
    for (int c = 0; c < C; ++c) M1.at<float>(c, i) /= norm;
  }
  // Reshape back to (C, H, W)
  std::vector<cv::Mat> M1_channels;
  for (int c = 0; c < C; ++c) {
    M1_channels.push_back(cv::Mat(H, W, CV_32F, M1.ptr<float>(c)));
  }
  cv::Mat M1_normed;
  cv::merge(M1_channels, M1_normed);  // (H, W, C)

  // Get heatmap K1h
  cv::Mat K1h = get_kpts_heatmap(K1_tensor);

  // Save heatmap for debugging
  static int heatmap_save_counter = 1;
  cv::Mat K1h_norm, K1h_u8;
  cv::normalize(K1h, K1h_norm, 0, 255, cv::NORM_MINMAX);
  K1h_norm.convertTo(K1h_u8, CV_8U);
  std::string fname = "debug_heatmap_cpp_" + std::to_string(heatmap_save_counter++) + ".png";
  cv::imwrite(fname, K1h_u8);

  // NMS on K1h (upsampled heatmap)
  cv::Mat mkpts_mat = nms(K1h, 0.05, 5);  // Pass K1h (cv::Mat), not K1_tensor

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
  std::vector<Ort::Value> nearest_out = interp_nearest_session_.Run(Ort::RunOptions{nullptr},
                                                                    interp_input_names.data(),
                                                                    interp_inputs.data(),
                                                                    2,
                                                                    interp_nearest_output_names_char.data(),
                                                                    interp_nearest_output_names_char.size());
  std::vector<Ort::Value> bilinear_out = interp_bilinear_session_.Run(Ort::RunOptions{nullptr},
                                                                      interp_input_names.data(),
                                                                      interp_inputs.data(),
                                                                      2,
                                                                      interp_bilinear_output_names_char.data(),
                                                                      interp_bilinear_output_names_char.size());
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
  for (int i = 0; i < std::min(top_k, (int)idxs.size()); ++i) {
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
  std::vector<Ort::Value> feats_out = interp_bicubic_session_.Run(Ort::RunOptions{nullptr},
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
  std::vector<cv::Point2f> valid_kpts;
  std::vector<float> valid_scores;
  std::vector<cv::Mat> valid_feats;
  for (int i = 0; i < n_kpts; ++i) {
    if (topk_scores[i] > 0) {
      valid_kpts.push_back(topk_kpts[i]);
      valid_scores.push_back(topk_scores[i]);
      valid_feats.push_back(feats_mat.row(i));
    }
  }
  cv::Mat valid_kpts_mat(valid_kpts.size(), 2, CV_32F);
  for (int i = 0; i < valid_kpts.size(); ++i) {
    valid_kpts_mat.at<float>(i, 0) = valid_kpts[i].x;
    valid_kpts_mat.at<float>(i, 1) = valid_kpts[i].y;
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
  return det;  // Placeholder
}

// match_mkpts: rewritten to match the logic of the Python version
std::tuple<std::vector<int>, std::vector<int>> XFeatONNX::match_mkpts_bf(const cv::Mat& feats1,
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

  std::vector<int> idx0, idx1;
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
          idx0.push_back(i);
          idx1.push_back(j);
        }
      } else {
        tbb::mutex::scoped_lock lock(idx_mutex);
        idx0.push_back(i);
        idx1.push_back(j);
      }
    }
  });
  return {idx0, idx1};
}

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> XFeatONNX::match(const cv::Mat& image1,
                                                                const cv::Mat& image2,
                                                                int top_k,
                                                                float min_cossim,
                                                                TimingStats* timing_stats) {
  auto t0 = std::chrono::high_resolution_clock::now();
  auto result1 = detect_and_compute(xfeat_session_, image1, top_k);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto result2 = detect_and_compute(xfeat_session_, image2, top_k);
  auto t2 = std::chrono::high_resolution_clock::now();

  auto match_start = std::chrono::high_resolution_clock::now();
  auto match_result = match(result1, result2, image1, top_k, min_cossim, timing_stats);
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
std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> XFeatONNX::match(const DetectionResult& result1,
                                                                const DetectionResult& result2,
                                                                const cv::Mat& image1,
                                                                int top_k,
                                                                float min_cossim,
                                                                TimingStats* timing_stats) {
  if (result1.keypoints.empty() || result2.keypoints.empty()) {
    std::cerr << "Detection failed for one or both DetectionResults." << std::endl;
    return {};
  }

  std::vector<int> indexes1, indexes2;
  auto t0 = std::chrono::high_resolution_clock::now();
  switch (matcher_type_) {
    case MatcherType::BF:
      std::tie(indexes1, indexes2) = match_mkpts_bf(result1.descriptors, result2.descriptors, min_cossim);
      break;
    case MatcherType::FLANN:
      std::tie(indexes1, indexes2) = match_mkpts_flann(result1.descriptors, result2.descriptors, min_cossim);
      break;
    case MatcherType::LIGHTERGLUE:
      if (!lighterglue_) {
        throw std::runtime_error("LighterGlue matcher is not initialized.");
      }
      std::array<float, 2> image_size0{static_cast<float>(input_width_), static_cast<float>(input_height_)};
      std::array<float, 2> image_size1{static_cast<float>(input_width_), static_cast<float>(input_height_)};
      std::tie(indexes1, indexes2) = lighterglue_->match(result1, image_size0, result2, image_size1);
      break;
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  if (timing_stats) {
    (*timing_stats)["match_mkpts"] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  t0 = std::chrono::high_resolution_clock::now();
  // Select matched keypoints
  cv::Mat mkpts0(indexes1.size(), 2, CV_32F);
  cv::Mat mkpts1(indexes2.size(), 2, CV_32F);
  for (size_t i = 0; i < indexes1.size(); ++i) {
    mkpts0.at<float>(i, 0) = result1.keypoints.at<float>(indexes1[i], 0);
    mkpts0.at<float>(i, 1) = result1.keypoints.at<float>(indexes1[i], 1);
    mkpts1.at<float>(i, 0) = result2.keypoints.at<float>(indexes2[i], 0);
    mkpts1.at<float>(i, 1) = result2.keypoints.at<float>(indexes2[i], 1);
  }

  // Filter matches using homography (RANSAC)
  auto [keypoints1, keypoints2, matches] = calc_warp_corners_and_matches(mkpts0, mkpts1, image1);

  // Convert filtered keypoints back to cv::Mat for return
  cv::Mat filtered_mkpts0((int)matches.size(), 2, CV_32F);
  cv::Mat filtered_mkpts1((int)matches.size(), 2, CV_32F);
  for (size_t i = 0; i < matches.size(); ++i) {
    filtered_mkpts0.at<float>(i, 0) = keypoints1[i].pt.x;
    filtered_mkpts0.at<float>(i, 1) = keypoints1[i].pt.y;
    filtered_mkpts1.at<float>(i, 0) = keypoints2[i].pt.x;
    filtered_mkpts1.at<float>(i, 1) = keypoints2[i].pt.y;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  if (timing_stats) {
    (*timing_stats)["calc_warp_corners"] = std::chrono::duration<double, std::milli>(t2 - t0).count();
  }

  return std::make_tuple(filtered_mkpts0, filtered_mkpts1, result1.keypoints, result2.keypoints);
}

// Calculate warped corners and matches using homography (like the Python
// version)
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>>
XFeatONNX::calc_warp_corners_and_matches(const cv::Mat& ref_points, const cv::Mat& dst_points, const cv::Mat& image1) {
  // Compute homography (use cv::RANSAC as int for compatibility)
  cv::Mat mask;
  cv::Mat H = cv::findHomography(ref_points, dst_points, cv::RANSAC, 3.5, mask, 200, 0.9);
  if (H.empty()) {
    std::cerr << "Homography estimation failed." << std::endl;
    return {};
  }
  mask = mask.reshape(1, mask.total());

  // Prepare keypoints and matches
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  std::vector<cv::DMatch> matches;
  for (int i = 0; i < ref_points.rows; ++i) {
    if (mask.at<uchar>(i)) {
      keypoints1.emplace_back(ref_points.at<float>(i, 0), ref_points.at<float>(i, 1), 5);
      keypoints2.emplace_back(dst_points.at<float>(i, 0), dst_points.at<float>(i, 1), 5);
      matches.emplace_back(i, i, 0);
    }
  }
  return {keypoints1, keypoints2, matches};
}

DetectionResult XFeatONNX::detect_and_compute(const cv::Mat& image, int top_k) {
  return detect_and_compute(xfeat_session_, image, top_k);
}

std::tuple<std::vector<int>, std::vector<int>> XFeatONNX::match_mkpts_flann(const cv::Mat& feats1,
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

  std::vector<int> idx0, idx1;
  idx0.reserve(matches12.size());
  idx1.reserve(matches12.size());

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
          idx0.push_back((int)i);
          idx1.push_back(j);
        }
      } else {
        idx0.push_back((int)i);
        idx1.push_back(j);
      }
    }
  }
  return {idx0, idx1};
}