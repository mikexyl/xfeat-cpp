#include "xfeat-cpp/stereo_depth/stereo_depth_vpi.h"

#ifdef HAVE_VPI

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/StereoDisparity.h>

#include <cmath>
#include <cstdint>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vpi/OpenCVInterop.hpp>

#include "stereo_depth_vpi_utils.h"

namespace xfeat {
namespace {

void checkVPI(VPIStatus status, const char* operation) {
  if (status == VPI_SUCCESS) {
    return;
  }
  char message[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
  vpiGetLastStatusMessage(message, sizeof(message));
  throw std::runtime_error(std::string("VPI ") + operation + " failed: " + vpiStatusGetName(status) +
                           (message[0] == '\0' ? "" : std::string(" (") + message + ")"));
}

cv::Mat toGray8(const cv::Mat& image) {
  if (image.depth() != CV_8U) {
    throw std::invalid_argument("VPI stereo input must have 8-bit channels");
  }
  if (image.channels() == 1) {
    return image;
  }
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::invalid_argument("VPI stereo input must have 1, 3, or 4 channels");
  }
  return gray;
}

}  // namespace

class VPIStereoDepth::Impl {
 public:
  explicit Impl(const Params& params) : params_(params) {}

  ~Impl() {
    // Destroying the stream first completes any queued work before its images
    // and payload are released.
    vpiStreamDestroy(stream_);
    vpiImageDestroy(left_);
    vpiImageDestroy(right_);
    vpiImageDestroy(disparity_);
    vpiImageDestroy(confidence_);
    vpiPayloadDestroy(payload_);
  }

  void initialize(const cv::Size& size) {
    if (size.width <= 0 || size.height <= 0) {
      throw std::invalid_argument("VPI stereo working size must be positive");
    }
    if (initialized_) {
      if (size != size_) {
        throw std::invalid_argument("VPI stereo working size cannot change after initialization");
      }
      return;
    }

    size_ = size;
    checkVPI(vpiStreamCreate(0, &stream_), "stream creation");

    VPIStereoDisparityEstimatorCreationParams creation_params;
    checkVPI(vpiInitStereoDisparityEstimatorCreationParams(&creation_params), "creation parameter initialization");
    creation_params.maxDisparity = params_.max_disparity;
    creation_params.downscaleFactor = 1;
    creation_params.includeDiagonals = params_.include_diagonals ? 1 : 0;
    checkVPI(vpiCreateStereoDisparityEstimator(
                 VPI_BACKEND_CUDA, size.width, size.height, VPI_IMAGE_FORMAT_Y8_ER, &creation_params, &payload_),
             "CUDA stereo payload creation");

    checkVPI(vpiImageCreate(size.width, size.height, VPI_IMAGE_FORMAT_Y8_ER, 0, &left_), "left image creation");
    checkVPI(vpiImageCreate(size.width, size.height, VPI_IMAGE_FORMAT_Y8_ER, 0, &right_), "right image creation");
    checkVPI(vpiImageCreate(size.width, size.height, VPI_IMAGE_FORMAT_S16, 0, &disparity_), "disparity image creation");
    checkVPI(vpiImageCreate(size.width, size.height, VPI_IMAGE_FORMAT_U16, 0, &confidence_),
             "confidence image creation");
    initialized_ = true;
  }

  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity, cv::Mat& valid) {
    initialize(left.size());

    VPIImage wrapped_left = nullptr;
    VPIImage wrapped_right = nullptr;
    bool disparity_locked = false;
    bool confidence_locked = false;
    try {
      checkVPI(vpiImageCreateWrapperOpenCVMat(left, 0, &wrapped_left), "left OpenCV wrapper creation");
      checkVPI(vpiImageCreateWrapperOpenCVMat(right, 0, &wrapped_right), "right OpenCV wrapper creation");

      VPIConvertImageFormatParams conversion_params;
      checkVPI(vpiInitConvertImageFormatParams(&conversion_params), "conversion parameter initialization");
      checkVPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, wrapped_left, left_, &conversion_params),
               "left CUDA image conversion");
      checkVPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, wrapped_right, right_, &conversion_params),
               "right CUDA image conversion");

      VPIStereoDisparityEstimatorParams submit_params;
      checkVPI(vpiInitStereoDisparityEstimatorParams(&submit_params), "submission parameter initialization");
      submit_params.maxDisparity = 0;
      submit_params.minDisparity = params_.min_disparity;
      submit_params.p1 = params_.p1;
      submit_params.p2 = params_.p2;
      submit_params.confidenceThreshold = params_.confidence_threshold;
      submit_params.confidenceType = VPI_STEREO_CONFIDENCE_ABSOLUTE;
      submit_params.uniqueness = params_.uniqueness;

      checkVPI(vpiSubmitStereoDisparityEstimator(
                   stream_, VPI_BACKEND_CUDA, payload_, left_, right_, disparity_, confidence_, &submit_params),
               "CUDA stereo submission");
      checkVPI(vpiStreamSync(stream_), "stream synchronization");

      VPIImageData disparity_data;
      checkVPI(vpiImageLockData(disparity_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &disparity_data),
               "disparity lock");
      disparity_locked = true;
      cv::Mat disparity_view;
      checkVPI(vpiImageDataExportOpenCVMat(disparity_data, &disparity_view), "disparity OpenCV export");
      disparity = disparity_view.clone();
      checkVPI(vpiImageUnlock(disparity_), "disparity unlock");
      disparity_locked = false;

      VPIImageData confidence_data;
      checkVPI(vpiImageLockData(confidence_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &confidence_data),
               "confidence lock");
      confidence_locked = true;
      cv::Mat confidence_view;
      checkVPI(vpiImageDataExportOpenCVMat(confidence_data, &confidence_view), "confidence OpenCV export");
      valid = vpi_detail::makeValidityMask(disparity,
                                           confidence_view,
                                           params_.min_valid_disparity,
                                           params_.confidence_threshold,
                                           VPIStereoDepth::kDisparityScale);
      checkVPI(vpiImageUnlock(confidence_), "confidence unlock");
      confidence_locked = false;
    } catch (...) {
      // A conversion may already be queued when a later submission fails.
      // Complete or cancel all stream work before releasing wrapped cv::Mat
      // memory.
      vpiStreamSync(stream_);
      if (disparity_locked) {
        vpiImageUnlock(disparity_);
      }
      if (confidence_locked) {
        vpiImageUnlock(confidence_);
      }
      vpiImageDestroy(wrapped_left);
      vpiImageDestroy(wrapped_right);
      throw;
    }

    // Wrappers cannot outlive the cv::Mat memory supplied by this call.
    vpiImageDestroy(wrapped_left);
    vpiImageDestroy(wrapped_right);
  }

 private:
  Params params_;
  cv::Size size_;
  bool initialized_ = false;
  VPIStream stream_ = nullptr;
  VPIPayload payload_ = nullptr;
  VPIImage left_ = nullptr;
  VPIImage right_ = nullptr;
  VPIImage disparity_ = nullptr;
  VPIImage confidence_ = nullptr;
};

VPIStereoDepth::VPIStereoDepth() : VPIStereoDepth(Params{}) {}

VPIStereoDepth::VPIStereoDepth(const Params& params) : params_(params), impl_(std::make_unique<Impl>(params)) {
  if (params_.min_disparity < 0 || params_.max_disparity < 1 || params_.max_disparity > 256 ||
      params_.min_disparity >= params_.max_disparity) {
    throw std::invalid_argument("VPI disparity range must satisfy 0 <= min < max <= 256");
  }
  if (params_.min_valid_disparity < params_.min_disparity || params_.min_valid_disparity >= params_.max_disparity) {
    throw std::invalid_argument("VPI minimum valid disparity must be within the search range");
  }
  if (params_.p1 <= 0 || params_.p2 < params_.p1 || params_.p2 >= 256) {
    throw std::invalid_argument("VPI penalties must satisfy 0 < p1 <= p2 < 256");
  }
  if (params_.confidence_threshold < 0 || params_.confidence_threshold > UINT16_MAX) {
    throw std::invalid_argument("VPI confidence threshold must be in [0, 65535]");
  }
  if (!(params_.uniqueness == -1.0f ||
        (std::isfinite(params_.uniqueness) && params_.uniqueness >= 0.0f && params_.uniqueness <= 1.0f))) {
    throw std::invalid_argument("VPI uniqueness must be -1 or in [0, 1]");
  }
  if ((params_.target_size.width == 0) != (params_.target_size.height == 0) || params_.target_size.width < 0 ||
      params_.target_size.height < 0) {
    throw std::invalid_argument("VPI target_size must be empty or positive");
  }
}

VPIStereoDepth::~VPIStereoDepth() = default;

void VPIStereoDepth::warmup(const cv::Size& image_size) {
  if (image_size.width <= 0 || image_size.height <= 0) {
    throw std::invalid_argument("VPI warmup image size must be positive");
  }
  const cv::Size working_size = params_.target_size.empty() ? image_size : params_.target_size;
  if (working_size.width > image_size.width || working_size.height > image_size.height) {
    throw std::invalid_argument("VPI target_size must not enlarge the stereo input");
  }
  impl_->initialize(working_size);
}

void VPIStereoDepth::compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) {
  if (left.empty() || right.empty()) {
    throw std::invalid_argument("VPI stereo input images are empty");
  }
  if (left.size() != right.size()) {
    throw std::invalid_argument("VPI left and right images must have the same size");
  }

  cv::Mat left_gray = toGray8(left);
  cv::Mat right_gray = toGray8(right);
  const cv::Size original_size = left_gray.size();
  const bool needs_resize = !params_.target_size.empty() && params_.target_size != original_size;
  if (needs_resize) {
    if (params_.target_size.width > original_size.width || params_.target_size.height > original_size.height) {
      throw std::invalid_argument("VPI target_size must not enlarge the stereo input");
    }
    cv::resize(left_gray, left_gray, params_.target_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(right_gray, right_gray, params_.target_size, 0.0, 0.0, cv::INTER_LINEAR);
  }

  cv::Mat working_disparity;
  cv::Mat working_valid;
  impl_->compute(left_gray, right_gray, working_disparity, working_valid);
  if (working_disparity.type() != CV_16S || working_valid.type() != CV_8U) {
    throw std::runtime_error("VPI stereo returned an unexpected output format");
  }

  const int16_t invalid_disparity = static_cast<int16_t>((params_.min_disparity - 1) * getDisparityScale());
  working_disparity.setTo(invalid_disparity, ~working_valid);
  if (!needs_resize) {
    disparity = working_disparity;
    return;
  }

  const double horizontal_scale =
      static_cast<double>(original_size.width) / static_cast<double>(params_.target_size.width);
  vpi_detail::restoreDisparityWithValidity(
      working_disparity, working_valid, original_size, horizontal_scale, invalid_disparity, &disparity);
}

}  // namespace xfeat

#endif  // HAVE_VPI
