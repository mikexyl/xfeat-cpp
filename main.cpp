// filepath: /Users/mikexyl/Workspaces/onnx_ws/src/XFeat-Image-Matching-ONNX-Sample/main.cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// Helper function to convert OpenCV Mat to ONNX tensor
// (Further implementation needed)

// Helper function for NMS
// (Further implementation needed)

// Helper function for heatmap generation
// (Further implementation needed)

class XFeatONNX {
public:
    XFeatONNX(
        const std::string& xfeat_path,
        const std::string& interp_bilinear_path,
        const std::string& interp_bicubic_path,
        const std::string& interp_nearest_path,
        bool use_gpu);

    std::tuple<cv::Mat, cv::Mat> match(
        const cv::Mat& image1,
        const cv::Mat& image2,
        int top_k = 4096,
        float min_cossim = -1.0f);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session xfeat_session_;
    Ort::Session interp_bilinear_session_;
    Ort::Session interp_bicubic_session_;
    Ort::Session interp_nearest_session_;

public: // Make accessible
    int input_width_;
    int input_height_;
private:
    std::string interp_input_name1_;
    std::string interp_input_name2_;


    std::tuple<cv::Mat, float, float> preprocess_image(const cv::Mat& image);

    cv::Mat get_kpts_heatmap(
        const Ort::Value& kpts_tensor,
        float softmax_temp = 1.0f);

    cv::Mat nms(
        const Ort::Value& heatmap_tensor,
        float threshold = 0.05f,
        int kernel_size = 5);

    struct DetectionResult {
        cv::Mat keypoints;
        cv::Mat scores;
        cv::Mat descriptors;
    };

    std::vector<DetectionResult> detect_and_compute(
        Ort::Session& session,
        const cv::Mat& image,
        int top_k = 4096);

    std::tuple<std::vector<int>, std::vector<int>> match_mkpts(
        const cv::Mat& feats1,
        const cv::Mat& feats2,
        float min_cossim = 0.82f);
};

XFeatONNX::XFeatONNX(
    const std::string& xfeat_path,
    const std::string& interp_bilinear_path,
    const std::string& interp_bicubic_path,
    const std::string& interp_nearest_path,
    bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "XFeatONNX"),
      xfeat_session_(nullptr),
      interp_bilinear_session_(nullptr),
      interp_bicubic_session_(nullptr),
      interp_nearest_session_(nullptr)
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    if (use_gpu) {
        // Add CUDA provider if available and requested
        std::cout << "GPU support is requested, but not fully implemented in this snippet." << std::endl;
    }
#ifdef _WIN32
    const std::wstring widexfeatPath = std::wstring(xfeat_path.begin(), xfeat_path.end());
    const std::wstring wideinterp_bilinear_path = std::wstring(interp_bilinear_path.begin(), interp_bilinear_path.end());
    const std::wstring wideinterp_bicubic_path = std::wstring(interp_bicubic_path.begin(), interp_bicubic_path.end());
    const std::wstring wideinterp_nearest_path = std::wstring(interp_nearest_path.begin(), interp_nearest_path.end());
    xfeat_session_ = Ort::Session(env_, widexfeatPath.c_str(), session_options_);
    interp_bilinear_session_ = Ort::Session(env_, wideinterp_bilinear_path.c_str(), session_options_);
    interp_bicubic_session_ = Ort::Session(env_, wideinterp_bicubic_path.c_str(), session_options_);
    interp_nearest_session_ = Ort::Session(env_, wideinterp_nearest_path.c_str(), session_options_);
#else
    xfeat_session_ = Ort::Session(env_, xfeat_path.c_str(), session_options_);
    interp_bilinear_session_ = Ort::Session(env_, interp_bilinear_path.c_str(), session_options_);
    interp_bicubic_session_ = Ort::Session(env_, interp_bicubic_path.c_str(), session_options_);
    interp_nearest_session_ = Ort::Session(env_, interp_nearest_path.c_str(), session_options_);
#endif
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
}

// Placeholder for preprocess_image
std::tuple<cv::Mat, float, float> XFeatONNX::preprocess_image(const cv::Mat& image) {
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(input_width_, input_height_));
    input_image.convertTo(input_image, CV_32F, 1.0 / 255.0);

    // Transpose HWC to CHW
    cv::Mat channels[3];
    cv::split(input_image, channels);
    cv::Mat input_tensor_chw = cv::Mat(input_width_ * input_height_, 3, CV_32F);
    for (int i = 0; i < 3; ++i) {
        channels[i].reshape(1, input_width_ * input_height_).copyTo(input_tensor_chw.col(i));
    }
    input_tensor_chw = input_tensor_chw.reshape(1, {1, 3, input_height_, input_width_});


    float resize_rate_w = static_cast<float>(image.cols) / input_width_;
    float resize_rate_h = static_cast<float>(image.rows) / input_height_;

    return std::make_tuple(input_tensor_chw.clone(), resize_rate_w, resize_rate_h);
}

// Implemented get_kpts_heatmap
cv::Mat XFeatONNX::get_kpts_heatmap(
    const Ort::Value& kpts_tensor, // Should be Ort::Value
    float softmax_temp) {
    // Extract shape and data
    auto shape = kpts_tensor.GetTensorTypeAndShapeInfo().GetShape();
    // Expect shape [1, 1, H, W]
    int H = static_cast<int>(shape[2]);
    int W = static_cast<int>(shape[3]);
    const float* data = kpts_tensor.GetTensorData<float>();
    int N = H * W;
    // Copy logits to vector
    std::vector<float> logits(data, data + N);
    // Apply softmax with temperature
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_logits(N);
    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i) {
        exp_logits[i] = std::exp((logits[i] - max_logit) / softmax_temp);
        sum_exp += exp_logits[i];
    }
    // Normalize
    for (int i = 0; i < N; ++i) {
        exp_logits[i] /= sum_exp;
    }
    // Convert to cv::Mat (H, W)
    cv::Mat heatmap(H, W, CV_32F);
    std::memcpy(heatmap.data, exp_logits.data(), N * sizeof(float));
    return heatmap;
}

// Placeholder for nms
cv::Mat XFeatONNX::nms(
    const Ort::Value& heatmap_tensor, // Should be Ort::Value
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

// Placeholder for detect_and_compute
std::vector<XFeatONNX::DetectionResult> XFeatONNX::detect_and_compute(
    Ort::Session& session,
    const cv::Mat& image,
    int top_k) {
    std::cout << "detect_and_compute called." << std::endl;
    auto [input_tensor, resize_rate_w, resize_rate_h] = preprocess_image(image);
    auto input_node_names = session.GetInputNames();
    auto output_node_names = session.GetOutputNames();
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value input_ort_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(), (float*)input_tensor.data, input_tensor.total() * input_tensor.channels(),
        input_shape.data(), input_shape.size()
    );
    std::vector<const char*> input_names_char;
    for(const auto& name : input_node_names) {
        input_names_char.push_back(name.c_str());
    }
    std::vector<const char*> output_names_char;
    for(const auto& name : output_node_names) {
        output_names_char.push_back(name.c_str());
    }
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names_char.data(), &input_ort_tensor, 1,
            output_names_char.data(), output_names_char.size()
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
        return {};
    }
    // M1: Feature map, K1: Keypoint logits
    const Ort::Value& M1_tensor = output_tensors[0];
    const Ort::Value& K1_tensor = output_tensors[1];
    // Use GetTensorData for read-only access
    const float* M1_data = M1_tensor.GetTensorData<float>();
    const float* K1_data = K1_tensor.GetTensorData<float>();

    // L2 normalize M1
    std::vector<int64_t> M1_shape = M1_tensor.GetTensorTypeAndShapeInfo().GetShape();
    int B = M1_shape[0];
    int C = M1_shape[1];
    int H = M1_shape[2];
    int W = M1_shape[3];
    cv::Mat M1(C, H * W, CV_32F, (void*)M1_data); // (C, H*W)
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
    cv::merge(M1_channels, M1_normed); // (H, W, C)

    // Get heatmap K1h
    cv::Mat K1h = get_kpts_heatmap(K1_tensor);
    // NMS on K1h
    cv::Mat mkpts_mat = nms(K1_tensor); // Pass K1_tensor, not K1h

    // Interpolate for scores (nearest and bilinear)
    // Prepare ONNX input for interpolators
    std::vector<int64_t> kpt_shape = {1, mkpts_mat.rows, 2};
    size_t mkpts_numel = 1 * mkpts_mat.rows * 2;
    if (mkpts_mat.total() != mkpts_numel) {
        throw std::runtime_error("mkpts_mat buffer size does not match shape");
    }
    Ort::Value mkpts_tensor = Ort::Value::CreateTensor<float>(allocator.GetInfo(), (float*)mkpts_mat.ptr<float>(), mkpts_numel, kpt_shape.data(), kpt_shape.size());
    std::vector<int64_t> K1h_shape = {1, 1, K1h.rows, K1h.cols};
    size_t K1h_numel = 1 * 1 * K1h.rows * K1h.cols;
    if (K1h.total() != K1h_numel) {
        throw std::runtime_error("K1h buffer size does not match shape");
    }
    Ort::Value K1h_tensor = Ort::Value::CreateTensor<float>(allocator.GetInfo(), (float*)K1h.ptr<float>(), K1h_numel, K1h_shape.data(), K1h_shape.size());
    // Nearest
    std::vector<const char*> interp_input_names = {interp_input_name1_.c_str(), interp_input_name2_.c_str()};
    std::vector<Ort::Value> interp_inputs;
    interp_inputs.push_back(std::move(K1h_tensor));
    interp_inputs.push_back(std::move(mkpts_tensor));
    // Fetch actual output node names for interpolators
    auto interp_nearest_output_names = interp_nearest_session_.GetOutputNames();
    std::vector<const char*> interp_nearest_output_names_char;
    for(const auto& name : interp_nearest_output_names) interp_nearest_output_names_char.push_back(name.c_str());
    auto interp_bilinear_output_names = interp_bilinear_session_.GetOutputNames();
    std::vector<const char*> interp_bilinear_output_names_char;
    for(const auto& name : interp_bilinear_output_names) interp_bilinear_output_names_char.push_back(name.c_str());
    std::vector<Ort::Value> nearest_out = interp_nearest_session_.Run(Ort::RunOptions{nullptr}, interp_input_names.data(), interp_inputs.data(), 2, interp_nearest_output_names_char.data(), interp_nearest_output_names_char.size());
    std::vector<Ort::Value> bilinear_out = interp_bilinear_session_.Run(Ort::RunOptions{nullptr}, interp_input_names.data(), interp_inputs.data(), 2, interp_bilinear_output_names_char.data(), interp_bilinear_output_names_char.size());
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
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
        return scores_mat.at<float>(a, 0) > scores_mat.at<float>(b, 0);
    });
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
    Ort::Value topk_kpts_tensor = Ort::Value::CreateTensor<float>(allocator.GetInfo(), (float*)topk_kpts_mat.ptr<float>(), topk_numel, topk_shape.data(), topk_shape.size());
    std::vector<int64_t> M1_shape_interp = {1, C, H, W};
    size_t M1_numel = 1 * C * H * W;
    // Convert M1_normed (H, W, C) to (C, H, W) contiguous buffer
    std::vector<float> M1_chw(C * H * W);
    std::vector<cv::Mat> M1_split;
    cv::split(M1_normed, M1_split); // M1_split: C x (H,W)
    for (int c = 0; c < C; c++) {
        std::memcpy(&M1_chw[c * H * W], M1_split[c].ptr<float>(), H * W * sizeof(float));
    }
    if (M1_chw.size() != M1_numel) {
        throw std::runtime_error("M1_chw buffer size does not match shape");
    }
    Ort::Value M1_tensor_interp = Ort::Value::CreateTensor<float>(allocator.GetInfo(), M1_chw.data(), M1_numel, M1_shape_interp.data(), M1_shape_interp.size());
    std::vector<Ort::Value> bicubic_inputs;
    bicubic_inputs.push_back(std::move(M1_tensor_interp));
    bicubic_inputs.push_back(std::move(topk_kpts_tensor));
    // Fetch actual output node names for bicubic interpolator
    auto interp_bicubic_output_names = interp_bicubic_session_.GetOutputNames();
    std::vector<const char*> interp_bicubic_output_names_char;
    for(const auto& name : interp_bicubic_output_names) interp_bicubic_output_names_char.push_back(name.c_str());
    std::vector<Ort::Value> feats_out = interp_bicubic_session_.Run(Ort::RunOptions{nullptr}, interp_input_names.data(), bicubic_inputs.data(), 2, interp_bicubic_output_names_char.data(), interp_bicubic_output_names_char.size());
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
    std::vector<DetectionResult> results;
    DetectionResult det;
    det.keypoints = valid_kpts_mat;
    det.scores = valid_scores_mat;
    det.descriptors = valid_feats_mat;
    results.push_back(det);
    return results; // Placeholder
}

// Placeholder for match_mkpts
std::tuple<std::vector<int>, std::vector<int>> XFeatONNX::match_mkpts(
    const cv::Mat& feats1,
    const cv::Mat& feats2,
    float min_cossim) {
    // feats1: (N1, D), feats2: (N2, D)
    int N1 = feats1.rows;
    int N2 = feats2.rows;
    int D = feats1.cols;
    cv::Mat cossim = feats1 * feats2.t(); // (N1, N2)
    cv::Mat cossim_t = feats2 * feats1.t(); // (N2, N1)
    std::vector<int> match12(N1), match21(N2);
    for (int i = 0; i < N1; ++i) {
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(cossim.row(i), nullptr, &maxVal, nullptr, &maxLoc);
        match12[i] = maxLoc.x;
    }
    for (int i = 0; i < N2; ++i) {
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(cossim_t.row(i), nullptr, &maxVal, nullptr, &maxLoc);
        match21[i] = maxLoc.x;
    }
    std::vector<int> idx0, idx1;
    for (int i = 0; i < N1; ++i) {
        int j = match12[i];
        if (match21[j] == i) {
            float sim = cossim.at<float>(i, j);
            if (min_cossim < 0 || sim > min_cossim) {
                idx0.push_back(i);
                idx1.push_back(j);
            }
        }
    }
    return {idx0, idx1};
}

// Placeholder for XFeatONNX::match
std::tuple<cv::Mat, cv::Mat> XFeatONNX::match(
    const cv::Mat& image1,
    const cv::Mat& image2,
    int top_k,
    float min_cossim) {
    std::cout << "match called." << std::endl;
    std::vector<DetectionResult> result1_vec = detect_and_compute(xfeat_session_, image1, top_k);
    std::vector<DetectionResult> result2_vec = detect_and_compute(xfeat_session_, image2, top_k);

    if (result1_vec.empty() || result2_vec.empty()) {
        std::cerr << "Detection failed for one or both images." << std::endl;
        return std::make_tuple(cv::Mat(), cv::Mat());
    }

    DetectionResult result1 = result1_vec[0];
    DetectionResult result2 = result2_vec[0];

    auto [indexes1, indexes2] = match_mkpts(result1.descriptors, result2.descriptors, min_cossim);

    // Select matched keypoints
    cv::Mat mkpts0(indexes1.size(), 2, CV_32F);
    cv::Mat mkpts1(indexes2.size(), 2, CV_32F);
    for (size_t i = 0; i < indexes1.size(); ++i) {
        mkpts0.at<float>(i, 0) = result1.keypoints.at<float>(indexes1[i], 0);
        mkpts0.at<float>(i, 1) = result1.keypoints.at<float>(indexes1[i], 1);
        mkpts1.at<float>(i, 0) = result2.keypoints.at<float>(indexes2[i], 0);
        mkpts1.at<float>(i, 1) = result2.keypoints.at<float>(indexes2[i], 1);
    }

    return std::make_tuple(mkpts0, mkpts1);
}


int main(int argc, char* argv[]) {
    std::string image1_path = (argc > 1) ? argv[1] : "image/sample1.jpg";
    std::string image2_path = (argc > 2) ? argv[2] : "image/sample2.jpg";

    std::string xfeat_model_path = (argc > 3) ? argv[3] : "onnx_model/xfeat_256x256.onnx";
    std::string interp_bilinear_path = (argc > 4) ? argv[4] : "onnx_model/interpolator_bilinear_256x256.onnx";
    std::string interp_bicubic_path = (argc > 5) ? argv[5] : "onnx_model/interpolator_bicubic_256x256.onnx";
    std::string interp_nearest_path = (argc > 6) ? argv[6] : "onnx_model/interpolator_nearest_256x256.onnx";


    cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_COLOR);

    if (image1.empty() || image2.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return 1;
    }
    std::cout << "Images loaded successfully." << std::endl;


    try {
        XFeatONNX xfeat_onnx(
            xfeat_model_path,
            interp_bilinear_path,
            interp_bicubic_path,
            interp_nearest_path,
            false // use_gpu
        );

        auto [mkpts0, mkpts1] = xfeat_onnx.match(image1, image2);

        std::cout << "Matching complete (partially implemented)." << std::endl;
        // Draw matches using OpenCV's drawMatches
        if (!mkpts0.empty() && !mkpts1.empty()) {
            cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_COLOR);
            cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_COLOR);
            std::vector<cv::KeyPoint> kpts1, kpts2;
            for (int i = 0; i < mkpts0.rows; ++i) {
                kpts1.emplace_back(mkpts0.at<float>(i, 0), mkpts0.at<float>(i, 1), 1.f);
                kpts2.emplace_back(mkpts1.at<float>(i, 0), mkpts1.at<float>(i, 1), 1.f);
            }
            // Create DMatch vector (1-to-1)
            std::vector<cv::DMatch> matches;
            for (int i = 0; i < mkpts0.rows; ++i) {
                matches.emplace_back(i, i, 0.f);
            }
            cv::Mat out_img;
            cv::drawMatches(img1, kpts1, img2, kpts2, matches, out_img, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("Matches", out_img);
            cv::waitKey(0);
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception in main: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in main: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}

