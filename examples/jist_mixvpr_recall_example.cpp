#include <algorithm>
#include <array>
#include <boost/program_options.hpp>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "xfeat-cpp/lighterglue_trt.h"
#include "xfeat-cpp/place_recognition/jist_trt.h"
#include "xfeat-cpp/place_recognition/mixvpr_trt.h"
#include "xfeat-cpp/xfeat_trt.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Pose {
  double timestamp = 0.0;
  Vec3 translation;
  double yaw = 0.0;
};

struct Dataset {
  std::string name;
  fs::path image_dir;
  fs::path pose_path;
  std::vector<Pose> poses;
};

struct Sequence {
  size_t dataset_index = 0;
  std::vector<fs::path> images;
  std::vector<double> frame_timestamps;
  std::vector<Pose> frame_poses;
  std::vector<double> frame_pose_dts;
  size_t first_source_index = 0;
  size_t last_source_index = 0;
  double timestamp = 0.0;
  Pose pose;
  double pose_dt = 0.0;
};

struct GroundTruthReference {
  size_t dataset = 0;
  double timestamp = 0.0;
  double distance = 0.0;
  double yaw_difference = 0.0;
};

struct GroundTruthLoop {
  size_t query_dataset = 0;
  double query_timestamp = 0.0;
  std::vector<GroundTruthReference> references;
};

struct Detection {
  size_t first_dataset = 0;
  size_t second_dataset = 0;
  double first_time = 0.0;
  double second_time = 0.0;
  double retrieval_score = 0.0;
  bool selected_frame_positive = false;
  fs::path first_image;
  fs::path second_image;
};

enum class EvaluationMode {
  kLastFrame,
  kTemporalAny,
};

struct FramePair {
  size_t first = 0;
  size_t second = 0;
};

struct RetrievalResult {
  uint64_t candidate_pairs = 0;
  std::vector<Detection> detections;
};

struct VerificationParams {
  int xfeat_top_k = 500;
  int minimum_matches = 20;
  int minimum_inliers = 15;
  double minimum_inlier_ratio = 0.25;
  double ransac_reprojection_threshold = 2.0;
  double ransac_confidence = 0.999;
  int ransac_max_iterations = 2000;
};

struct VerificationStats {
  uint64_t requested_pairs = 0;
  uint64_t accepted_requests = 0;
  uint64_t unique_pairs = 0;
  uint64_t pair_cache_hits = 0;
  uint64_t feature_extractions = 0;
  double feature_extraction_ms = 0.0;
  double lighterglue_ms = 0.0;
  double ransac_ms = 0.0;
};

struct Metrics {
  uint64_t candidate_pairs = 0;
  uint64_t ground_truth_loops = 0;
  uint64_t predicted_positives = 0;
  uint64_t detected_ground_truth_loops = 0;
  uint64_t correct_predicted_positives = 0;
  uint64_t false_positives = 0;
  uint64_t false_negatives = 0;
};

struct MetricRates {
  double recall = std::numeric_limits<double>::quiet_NaN();
  double precision = std::numeric_limits<double>::quiet_NaN();
  double f1 = std::numeric_limits<double>::quiet_NaN();
};

bool isImagePath(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" ||
         extension == ".tif" || extension == ".tiff";
}

std::vector<fs::path> listImages(const fs::path& image_dir) {
  if (!fs::is_directory(image_dir)) {
    throw std::runtime_error("Image directory does not exist: " + image_dir.string());
  }

  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(image_dir)) {
    if (entry.is_regular_file() && isImagePath(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  if (images.empty()) {
    throw std::runtime_error("No images found in: " + image_dir.string());
  }
  return images;
}

double imageTimestampSeconds(const fs::path& image_path) {
  const std::string stem = image_path.stem().string();
  const long double value = std::stold(stem);
  if (value > 1.0e12L) {
    return static_cast<double>(value * 1.0e-9L);
  }
  return static_cast<double>(value);
}

std::vector<Pose> readPoses(const fs::path& pose_path) {
  std::ifstream input(pose_path);
  if (!input) {
    throw std::runtime_error("Failed to open pose file: " + pose_path.string());
  }

  std::vector<Pose> poses;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }

    std::istringstream row(line);
    Pose pose;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;
    if (row >> pose.timestamp >> pose.translation.x >> pose.translation.y >> pose.translation.z >> qx >> qy >> qz >>
        qw) {
      pose.yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
      poses.push_back(pose);
    }
  }

  if (poses.empty()) {
    throw std::runtime_error("Pose file has no valid rows: " + pose_path.string());
  }
  std::sort(poses.begin(), poses.end(), [](const Pose& lhs, const Pose& rhs) { return lhs.timestamp < rhs.timestamp; });
  return poses;
}

Pose nearestPose(const std::vector<Pose>& poses, double timestamp, double max_dt, double* dt_out) {
  const auto next = std::lower_bound(
      poses.begin(), poses.end(), timestamp, [](const Pose& pose, double value) { return pose.timestamp < value; });

  const Pose* best = next == poses.end() ? nullptr : &*next;
  if (next != poses.begin()) {
    const Pose* previous = &*(next - 1);
    if (best == nullptr || std::abs(previous->timestamp - timestamp) < std::abs(best->timestamp - timestamp)) {
      best = previous;
    }
  }
  if (best == nullptr) {
    throw std::runtime_error("No pose available for timestamp " + std::to_string(timestamp));
  }

  const double dt = best->timestamp - timestamp;
  if (std::abs(dt) > max_dt) {
    throw std::runtime_error("Nearest pose is " + std::to_string(dt) + " seconds from image timestamp " +
                             std::to_string(timestamp) + ", exceeding --pose-max-dt");
  }
  if (dt_out != nullptr) {
    *dt_out = dt;
  }
  return *best;
}

double translationDistance(const Pose& lhs, const Pose& rhs) {
  const double dx = lhs.translation.x - rhs.translation.x;
  const double dy = lhs.translation.y - rhs.translation.y;
  const double dz = lhs.translation.z - rhs.translation.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double yawDifference(const Pose& lhs, const Pose& rhs) {
  return std::abs(std::remainder(lhs.yaw - rhs.yaw, 2.0 * std::acos(-1.0)));
}

std::vector<Pose> sampleTrajectory(const std::vector<Pose>& poses, double sample_period) {
  std::vector<Pose> samples;
  for (double timestamp = poses.front().timestamp; timestamp <= poses.back().timestamp; timestamp += sample_period) {
    samples.push_back(nearestPose(poses, timestamp, std::numeric_limits<double>::infinity(), nullptr));
  }
  if (poses.back().timestamp - samples.back().timestamp > 0.5 * sample_period) {
    samples.push_back(poses.back());
  }
  return samples;
}

std::vector<GroundTruthLoop> extractGroundTruthLoops(const std::vector<std::vector<Pose>>& trajectories,
                                                     double positive_distance,
                                                     double positive_yaw_difference,
                                                     double min_time_separation) {
  std::vector<GroundTruthLoop> loops;
  for (size_t query_dataset = 0; query_dataset < trajectories.size(); ++query_dataset) {
    for (const Pose& query_pose : trajectories[query_dataset]) {
      GroundTruthLoop loop;
      loop.query_dataset = query_dataset;
      loop.query_timestamp = query_pose.timestamp;
      for (size_t reference_dataset = 0; reference_dataset <= query_dataset; ++reference_dataset) {
        for (const Pose& reference_pose : trajectories[reference_dataset]) {
          if (reference_pose.timestamp >= query_pose.timestamp) {
            break;
          }
          if (reference_dataset == query_dataset &&
              query_pose.timestamp - reference_pose.timestamp < min_time_separation) {
            continue;
          }
          const double distance = translationDistance(reference_pose, query_pose);
          const double yaw_difference = yawDifference(reference_pose, query_pose);
          if (distance <= positive_distance && yaw_difference <= positive_yaw_difference) {
            loop.references.push_back({reference_dataset, reference_pose.timestamp, distance, yaw_difference});
          }
        }
      }
      if (!loop.references.empty()) {
        loops.push_back(std::move(loop));
      }
    }
  }
  return loops;
}

std::vector<Sequence> makeSequences(const std::vector<fs::path>& all_images,
                                    const std::vector<Pose>& poses,
                                    size_t dataset_index,
                                    size_t n_skip,
                                    size_t n_seq,
                                    size_t max_sequences,
                                    double pose_max_dt) {
  std::vector<size_t> sampled_indices;
  sampled_indices.reserve((all_images.size() + n_skip - 1) / n_skip);
  for (size_t index = 0; index < all_images.size(); index += n_skip) {
    sampled_indices.push_back(index);
  }

  size_t sequence_count = sampled_indices.size() / n_seq;
  if (max_sequences > 0) {
    sequence_count = std::min(sequence_count, max_sequences);
  }

  std::vector<Sequence> sequences;
  sequences.reserve(sequence_count);
  for (size_t sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
    const size_t sampled_start = sequence_index * n_seq;
    Sequence sequence;
    sequence.dataset_index = dataset_index;
    sequence.images.reserve(n_seq);
    sequence.frame_timestamps.reserve(n_seq);
    sequence.frame_poses.reserve(n_seq);
    sequence.frame_pose_dts.reserve(n_seq);
    for (size_t offset = 0; offset < n_seq; ++offset) {
      const fs::path& image_path = all_images[sampled_indices[sampled_start + offset]];
      const double timestamp = imageTimestampSeconds(image_path);
      double pose_dt = 0.0;
      sequence.images.push_back(image_path);
      sequence.frame_timestamps.push_back(timestamp);
      sequence.frame_poses.push_back(nearestPose(poses, timestamp, pose_max_dt, &pose_dt));
      sequence.frame_pose_dts.push_back(pose_dt);
    }

    sequence.first_source_index = sampled_indices[sampled_start];
    sequence.last_source_index = sampled_indices[sampled_start + n_seq - 1];
    sequence.timestamp = sequence.frame_timestamps.back();
    sequence.pose = sequence.frame_poses.back();
    sequence.pose_dt = sequence.frame_pose_dts.back();
    sequences.push_back(std::move(sequence));
  }
  return sequences;
}

std::vector<size_t> jistFrameOffsets(size_t logical_group_size, size_t jist_input_frames) {
  if (logical_group_size < jist_input_frames || jist_input_frames == 0) {
    throw std::invalid_argument("Logical sequence must contain at least as many images as the JIST engine input");
  }

  std::vector<size_t> offsets;
  offsets.reserve(jist_input_frames);
  for (size_t frame = 0; frame < jist_input_frames; ++frame) {
    offsets.push_back(jist_input_frames == 1
                          ? logical_group_size - 1
                          : static_cast<size_t>(std::llround(static_cast<double>(frame) * (logical_group_size - 1) /
                                                             static_cast<double>(jist_input_frames - 1))));
  }
  return offsets;
}

std::vector<cv::Mat> loadJistImages(const Sequence& sequence, size_t jist_input_frames) {
  std::vector<cv::Mat> images;
  images.reserve(jist_input_frames);
  for (const size_t group_offset : jistFrameOffsets(sequence.images.size(), jist_input_frames)) {
    const fs::path& path = sequence.images[group_offset];
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image: " + path.string());
    }
    images.push_back(std::move(image));
  }
  return images;
}

bool detectionMatchesLoop(const Detection& detection, const GroundTruthLoop& loop, double tolerance) {
  if (detection.second_dataset != loop.query_dataset ||
      std::abs(detection.second_time - loop.query_timestamp) > tolerance) {
    return false;
  }
  return std::any_of(loop.references.begin(), loop.references.end(), [&](const GroundTruthReference& reference) {
    return detection.first_dataset == reference.dataset &&
           std::abs(detection.first_time - reference.timestamp) <= tolerance;
  });
}

Metrics evaluateDetections(const std::vector<Detection>& detections,
                           uint64_t candidate_pairs,
                           const std::vector<GroundTruthLoop>& ground_truth_loops,
                           double gt_sample_period,
                           double temporal_match_tolerance,
                           EvaluationMode mode) {
  Metrics metrics;
  metrics.candidate_pairs = candidate_pairs;
  metrics.ground_truth_loops = ground_truth_loops.size();
  metrics.predicted_positives = detections.size();

  const double selected_frame_assignment_tolerance = gt_sample_period;
  for (const GroundTruthLoop& loop : ground_truth_loops) {
    const bool detected = std::any_of(detections.begin(), detections.end(), [&](const Detection& detection) {
      const bool selected_frame_match = detection.selected_frame_positive &&
                                        detectionMatchesLoop(detection, loop, selected_frame_assignment_tolerance);
      return selected_frame_match ||
             (mode == EvaluationMode::kTemporalAny && detectionMatchesLoop(detection, loop, temporal_match_tolerance));
    });
    metrics.detected_ground_truth_loops += detected;
  }

  for (const Detection& detection : detections) {
    bool correct = detection.selected_frame_positive;
    if (!correct && mode == EvaluationMode::kTemporalAny) {
      correct = std::any_of(ground_truth_loops.begin(), ground_truth_loops.end(), [&](const GroundTruthLoop& loop) {
        return detectionMatchesLoop(detection, loop, temporal_match_tolerance);
      });
    }
    metrics.correct_predicted_positives += correct;
  }
  metrics.false_positives = metrics.predicted_positives - metrics.correct_predicted_positives;
  metrics.false_negatives = metrics.ground_truth_loops - metrics.detected_ground_truth_loops;
  return metrics;
}

RetrievalResult retrieveMethod(const cv::Mat& descriptors,
                               const std::vector<Sequence>& sequences,
                               double threshold,
                               double positive_distance,
                               double positive_yaw_difference,
                               double min_time_separation) {
  if (descriptors.rows != static_cast<int>(sequences.size())) {
    throw std::invalid_argument("Descriptor and sequence counts do not match");
  }

  RetrievalResult result;
  for (int first = 0; first < descriptors.rows; ++first) {
    for (int second = first + 1; second < descriptors.rows; ++second) {
      const Sequence& first_sequence = sequences[static_cast<size_t>(first)];
      const Sequence& second_sequence = sequences[static_cast<size_t>(second)];
      if (first_sequence.dataset_index == second_sequence.dataset_index &&
          second_sequence.timestamp - first_sequence.timestamp < min_time_separation) {
        continue;
      }

      ++result.candidate_pairs;
      const double retrieval_score = descriptors.row(first).dot(descriptors.row(second));
      if (retrieval_score < threshold) {
        continue;
      }

      Detection detection;
      detection.first_dataset = first_sequence.dataset_index;
      detection.second_dataset = second_sequence.dataset_index;
      detection.first_time = first_sequence.timestamp;
      detection.second_time = second_sequence.timestamp;
      detection.retrieval_score = retrieval_score;
      detection.selected_frame_positive =
          translationDistance(first_sequence.pose, second_sequence.pose) <= positive_distance &&
          yawDifference(first_sequence.pose, second_sequence.pose) <= positive_yaw_difference;
      detection.first_image = first_sequence.images.back();
      detection.second_image = second_sequence.images.back();
      result.detections.push_back(std::move(detection));
    }
  }
  return result;
}

FramePair selectArgmaxFramePair(const std::vector<double>& similarities, size_t frame_count) {
  const auto best = std::max_element(similarities.begin(), similarities.end());
  const size_t flat_index = static_cast<size_t>(std::distance(similarities.begin(), best));
  return {flat_index / frame_count, flat_index % frame_count};
}

RetrievalResult retrieveFrameRefinedJist(const cv::Mat& sequence_descriptors,
                                         const cv::Mat& frame_descriptors,
                                         const std::vector<Sequence>& sequences,
                                         size_t jist_input_frames,
                                         double threshold,
                                         double positive_distance,
                                         double positive_yaw_difference,
                                         double min_time_separation) {
  if (sequence_descriptors.rows != static_cast<int>(sequences.size()) ||
      frame_descriptors.rows != static_cast<int>(sequences.size() * jist_input_frames) ||
      frame_descriptors.cols != sequence_descriptors.cols) {
    throw std::invalid_argument("JIST sequence/frame descriptors are invalid");
  }

  const std::vector<size_t> frame_offsets = jistFrameOffsets(sequences.front().images.size(), jist_input_frames);
  RetrievalResult result;
  for (int first = 0; first < sequence_descriptors.rows; ++first) {
    for (int second = first + 1; second < sequence_descriptors.rows; ++second) {
      const Sequence& first_sequence = sequences[static_cast<size_t>(first)];
      const Sequence& second_sequence = sequences[static_cast<size_t>(second)];
      if (first_sequence.dataset_index == second_sequence.dataset_index &&
          second_sequence.timestamp - first_sequence.timestamp < min_time_separation) {
        continue;
      }

      ++result.candidate_pairs;
      const double retrieval_score = sequence_descriptors.row(first).dot(sequence_descriptors.row(second));
      if (retrieval_score < threshold) {
        continue;
      }

      std::vector<double> similarities(jist_input_frames * jist_input_frames);
      for (size_t first_frame = 0; first_frame < jist_input_frames; ++first_frame) {
        const int first_row = first * static_cast<int>(jist_input_frames) + static_cast<int>(first_frame);
        for (size_t second_frame = 0; second_frame < jist_input_frames; ++second_frame) {
          const int second_row = second * static_cast<int>(jist_input_frames) + static_cast<int>(second_frame);
          similarities[first_frame * jist_input_frames + second_frame] =
              frame_descriptors.row(first_row).dot(frame_descriptors.row(second_row));
        }
      }

      const FramePair selected = selectArgmaxFramePair(similarities, jist_input_frames);
      const size_t first_offset = frame_offsets[selected.first];
      const size_t second_offset = frame_offsets[selected.second];
      const Pose& first_pose = first_sequence.frame_poses[first_offset];
      const Pose& second_pose = second_sequence.frame_poses[second_offset];
      Detection detection;
      detection.first_dataset = first_sequence.dataset_index;
      detection.second_dataset = second_sequence.dataset_index;
      detection.first_time = first_sequence.frame_timestamps[first_offset];
      detection.second_time = second_sequence.frame_timestamps[second_offset];
      detection.retrieval_score = retrieval_score;
      const bool selected_pair_is_temporally_eligible =
          detection.first_dataset != detection.second_dataset ||
          detection.second_time - detection.first_time >= min_time_separation;
      detection.selected_frame_positive = selected_pair_is_temporally_eligible &&
                                          translationDistance(first_pose, second_pose) <= positive_distance &&
                                          yawDifference(first_pose, second_pose) <= positive_yaw_difference;
      detection.first_image = first_sequence.images[first_offset];
      detection.second_image = second_sequence.images[second_offset];
      result.detections.push_back(std::move(detection));
    }
  }
  return result;
}

class GeometricVerifier {
 public:
  GeometricVerifier(xfeat::XFeatTRT& extractor, xfeat::LighterGlueTRT& matcher, const VerificationParams& params)
      : extractor_(extractor), matcher_(matcher), params_(params) {}

  RetrievalResult filter(const RetrievalResult& input) {
    RetrievalResult output;
    output.candidate_pairs = input.candidate_pairs;
    output.detections.reserve(input.detections.size());
    for (const Detection& detection : input.detections) {
      ++stats_.requested_pairs;
      if (verify(detection.first_image, detection.second_image)) {
        ++stats_.accepted_requests;
        output.detections.push_back(detection);
      }
    }
    return output;
  }

  const VerificationStats& stats() const { return stats_; }

 private:
  struct CachedFeatures {
    xfeat::DetectionResult detection;
    std::array<float, 2> image_size{};
  };

  struct PairDecision {
    bool accepted = false;
    size_t matches = 0;
    size_t inliers = 0;
  };

  CachedFeatures features(const fs::path& image_path) {
    const std::string key = image_path.string();
    const auto found = feature_cache_.find(key);
    if (found != feature_cache_.end()) {
      return found->second;
    }

    cv::Mat image = cv::imread(key, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read verification image: " + key);
    }
    const auto start = std::chrono::steady_clock::now();
    CachedFeatures result;
    result.detection = extractor_.detect_and_compute(image, params_.xfeat_top_k);
    stats_.feature_extraction_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    result.image_size = {static_cast<float>(image.cols), static_cast<float>(image.rows)};
    ++stats_.feature_extractions;
    feature_cache_.emplace(key, result);
    return result;
  }

  bool verify(const fs::path& first_image, const fs::path& second_image) {
    const std::string pair_key = first_image.string() + '\n' + second_image.string();
    const auto cached = pair_cache_.find(pair_key);
    if (cached != pair_cache_.end()) {
      ++stats_.pair_cache_hits;
      return cached->second.accepted;
    }

    ++stats_.unique_pairs;
    const CachedFeatures first = features(first_image);
    const CachedFeatures second = features(second_image);
    const auto match_start = std::chrono::steady_clock::now();
    const std::vector<std::vector<int>> match_indices =
        matcher_.match(first.detection, first.image_size, second.detection, second.image_size);
    stats_.lighterglue_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - match_start).count();

    std::vector<cv::Point2f> first_points;
    std::vector<cv::Point2f> second_points;
    for (size_t first_index = 0; first_index < match_indices.size(); ++first_index) {
      for (const int second_index : match_indices[first_index]) {
        first_points.emplace_back(first.detection.keypoints.at<float>(static_cast<int>(first_index), 0),
                                  first.detection.keypoints.at<float>(static_cast<int>(first_index), 1));
        second_points.emplace_back(second.detection.keypoints.at<float>(second_index, 0),
                                   second.detection.keypoints.at<float>(second_index, 1));
      }
    }

    PairDecision decision;
    decision.matches = first_points.size();
    if (decision.matches >= static_cast<size_t>(params_.minimum_matches)) {
      cv::Mat inlier_mask;
      const auto ransac_start = std::chrono::steady_clock::now();
      const cv::Mat fundamental = cv::findFundamentalMat(first_points,
                                                         second_points,
                                                         cv::FM_RANSAC,
                                                         params_.ransac_reprojection_threshold,
                                                         params_.ransac_confidence,
                                                         params_.ransac_max_iterations,
                                                         inlier_mask);
      stats_.ransac_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ransac_start).count();
      if (!fundamental.empty() && !inlier_mask.empty()) {
        decision.inliers = static_cast<size_t>(cv::countNonZero(inlier_mask));
        const double inlier_ratio = static_cast<double>(decision.inliers) / decision.matches;
        decision.accepted = decision.inliers >= static_cast<size_t>(params_.minimum_inliers) &&
                            inlier_ratio >= params_.minimum_inlier_ratio;
      }
    }
    pair_cache_.emplace(pair_key, decision);
    return decision.accepted;
  }

  xfeat::XFeatTRT& extractor_;
  xfeat::LighterGlueTRT& matcher_;
  VerificationParams params_;
  VerificationStats stats_;
  std::unordered_map<std::string, CachedFeatures> feature_cache_;
  std::unordered_map<std::string, PairDecision> pair_cache_;
};

double ratio(uint64_t numerator, uint64_t denominator) {
  if (denominator == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

MetricRates rates(const Metrics& metrics) {
  MetricRates result;
  result.recall = ratio(metrics.detected_ground_truth_loops, metrics.ground_truth_loops);
  result.precision = ratio(metrics.correct_predicted_positives, metrics.predicted_positives);
  if (std::isfinite(result.recall) && std::isfinite(result.precision) && result.recall + result.precision > 0.0) {
    result.f1 = 2.0 * result.recall * result.precision / (result.recall + result.precision);
  }
  return result;
}

RetrievalResult thresholdRetrieval(const RetrievalResult& retrieval, double threshold) {
  RetrievalResult filtered;
  filtered.candidate_pairs = retrieval.candidate_pairs;
  filtered.detections.reserve(retrieval.detections.size());
  std::copy_if(retrieval.detections.begin(),
               retrieval.detections.end(),
               std::back_inserter(filtered.detections),
               [threshold](const Detection& detection) { return detection.retrieval_score >= threshold; });
  return filtered;
}

std::vector<double> descendingThresholds(double minimum, double maximum, size_t count) {
  std::vector<double> thresholds;
  thresholds.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    const double fraction = static_cast<double>(index) / static_cast<double>(count - 1);
    thresholds.push_back(maximum - fraction * (maximum - minimum));
  }
  return thresholds;
}

void writePrecisionRecallCurveCsv(const fs::path& output_path,
                                  size_t n_seq,
                                  const std::vector<GroundTruthLoop>& ground_truth_loops,
                                  double gt_sample_period,
                                  double temporal_match_tolerance,
                                  const RetrievalResult& jist_last_retrieval,
                                  const RetrievalResult& jist_frame_argmax_retrieval,
                                  const RetrievalResult& mixvpr_last_retrieval,
                                  double jist_threshold_minimum,
                                  double jist_threshold_maximum,
                                  double mixvpr_threshold_minimum,
                                  double mixvpr_threshold_maximum,
                                  size_t threshold_count,
                                  const std::string& verification_stage) {
  if (!output_path.parent_path().empty()) {
    fs::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Failed to open precision-recall CSV: " + output_path.string());
  }
  output << "method,evaluation_mode,verification_stage,n_seq,threshold,candidate_pairs,ground_truth_loops,"
            "detected_ground_truth_loops,correct_predicted_positives,predicted_positives,false_positives,"
            "false_negatives,recall,precision,f1\n";

  const auto write_curve = [&](const std::string& method,
                               const std::string& evaluation_mode,
                               const RetrievalResult& base_retrieval,
                               double threshold_minimum,
                               double threshold_maximum) {
    for (const double threshold : descendingThresholds(threshold_minimum, threshold_maximum, threshold_count)) {
      const RetrievalResult retrieval = thresholdRetrieval(base_retrieval, threshold);
      const Metrics metrics = evaluateDetections(retrieval.detections,
                                                 retrieval.candidate_pairs,
                                                 ground_truth_loops,
                                                 gt_sample_period,
                                                 temporal_match_tolerance,
                                                 EvaluationMode::kLastFrame);
      const MetricRates metric_rates = rates(metrics);
      output << method << ',' << evaluation_mode << ',' << verification_stage << ',' << n_seq << ','
             << std::setprecision(9) << threshold << ',' << metrics.candidate_pairs << ',' << metrics.ground_truth_loops
             << ',' << metrics.detected_ground_truth_loops << ',' << metrics.correct_predicted_positives << ','
             << metrics.predicted_positives << ',' << metrics.false_positives << ',' << metrics.false_negatives << ','
             << metric_rates.recall << ',' << metric_rates.precision << ',' << metric_rates.f1 << '\n';
    }
  };

  if (n_seq != 0) {
    write_curve("JIST", "last-frame", jist_last_retrieval, jist_threshold_minimum, jist_threshold_maximum);
    write_curve("JIST", "frame-argmax", jist_frame_argmax_retrieval, jist_threshold_minimum, jist_threshold_maximum);
  }
  write_curve("MixVPR",
              n_seq == 0 ? "every-frame" : "last-frame",
              mixvpr_last_retrieval,
              mixvpr_threshold_minimum,
              mixvpr_threshold_maximum);
}

std::string percent(double value) {
  if (!std::isfinite(value)) {
    return "n/a";
  }
  std::ostringstream output;
  output << std::fixed << std::setprecision(2) << value * 100.0 << '%';
  return output.str();
}

void printMetrics(const std::string& method, double threshold, const Metrics& metrics) {
  const MetricRates result = rates(metrics);
  std::cout << std::left << std::setw(18) << method << std::right << std::setw(10) << std::fixed << std::setprecision(3)
            << threshold << std::setw(10) << metrics.ground_truth_loops << std::setw(12) << percent(result.recall)
            << std::setw(12) << percent(result.precision) << std::setw(12) << percent(result.f1) << std::setw(11)
            << metrics.detected_ground_truth_loops << std::setw(12) << metrics.correct_predicted_positives
            << std::setw(10) << metrics.false_positives << '\n';
}

void writeGroundTruthLoopsCsv(const fs::path& output_path,
                              const std::vector<GroundTruthLoop>& loops,
                              const std::vector<Dataset>& datasets) {
  if (output_path.empty()) {
    return;
  }
  if (!output_path.parent_path().empty()) {
    fs::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Failed to open GT-loop CSV: " + output_path.string());
  }
  output << "loop_id,query_dataset,query_timestamp,reference_count,reference_dataset,reference_timestamp,distance_m,"
            "yaw_difference_degrees\n";
  for (size_t index = 0; index < loops.size(); ++index) {
    const GroundTruthLoop& loop = loops[index];
    for (const GroundTruthReference& reference : loop.references) {
      output << index << ',' << datasets[loop.query_dataset].name << ',' << std::fixed << std::setprecision(6)
             << loop.query_timestamp << ',' << loop.references.size() << ',' << datasets[reference.dataset].name << ','
             << reference.timestamp << ',' << reference.distance << ','
             << reference.yaw_difference * 180.0 / std::acos(-1.0) << '\n';
    }
  }
}

void writeMetricsCsv(const fs::path& output_path,
                     size_t dataset_count,
                     size_t n_skip,
                     size_t n_seq,
                     size_t jist_input_frames,
                     size_t sequence_count,
                     double gt_sample_period,
                     double temporal_match_tolerance,
                     double positive_yaw_degrees,
                     const Metrics& jist_last_metrics,
                     const Metrics& jist_last_verified_metrics,
                     const Metrics& jist_any_metrics,
                     const Metrics& jist_any_verified_metrics,
                     const Metrics& jist_frame_argmax_metrics,
                     const Metrics& jist_frame_argmax_verified_metrics,
                     const Metrics& mixvpr_last_metrics,
                     const Metrics& mixvpr_last_verified_metrics,
                     double jist_threshold,
                     double mixvpr_threshold,
                     double jist_inference_ms,
                     double mixvpr_inference_ms,
                     const VerificationParams& verification_params,
                     const VerificationStats& verification_stats) {
  if (!output_path.parent_path().empty()) {
    fs::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Failed to open metrics CSV: " + output_path.string());
  }

  output << "method,evaluation_mode,verification_stage,datasets,n_skip,n_seq,jist_input_frames,sequences,"
            "candidate_pairs,"
            "ground_truth_loops,threshold,detected_ground_truth_loops,correct_predicted_positives,"
            "predicted_positives,false_positives,false_negatives,recall,precision,f1,gt_sample_period_seconds,"
            "temporal_match_tolerance_seconds,positive_yaw_degrees,inference_ms,ms_per_sequence,"
            "verification_xfeat_top_k,verification_minimum_matches,verification_minimum_inliers,"
            "verification_minimum_inlier_ratio,verification_ransac_reprojection_threshold_pixels,"
            "verification_requested_pairs,verification_accepted_requests,verification_unique_pairs,"
            "verification_pair_cache_hits,verification_feature_extractions,verification_ms\n";
  const auto write_row = [&](const std::string& method,
                             const std::string& evaluation_mode,
                             const std::string& verification_stage,
                             const Metrics& metrics,
                             double threshold,
                             double inference_ms) {
    const MetricRates result = rates(metrics);
    const double verification_ms =
        verification_stats.feature_extraction_ms + verification_stats.lighterglue_ms + verification_stats.ransac_ms;
    output << method << ',' << evaluation_mode << ',' << verification_stage << ',' << dataset_count << ',' << n_skip
           << ',' << n_seq << ',' << jist_input_frames << ',' << sequence_count << ',' << metrics.candidate_pairs << ','
           << metrics.ground_truth_loops << ',' << std::setprecision(9) << threshold << ','
           << metrics.detected_ground_truth_loops << ',' << metrics.correct_predicted_positives << ','
           << metrics.predicted_positives << ',' << metrics.false_positives << ',' << metrics.false_negatives << ','
           << result.recall << ',' << result.precision << ',' << result.f1 << ',' << gt_sample_period << ','
           << temporal_match_tolerance << ',' << positive_yaw_degrees << ',' << inference_ms << ','
           << inference_ms / sequence_count << ',' << verification_params.xfeat_top_k << ','
           << verification_params.minimum_matches << ',' << verification_params.minimum_inliers << ','
           << verification_params.minimum_inlier_ratio << ',' << verification_params.ransac_reprojection_threshold
           << ',' << verification_stats.requested_pairs << ',' << verification_stats.accepted_requests << ','
           << verification_stats.unique_pairs << ',' << verification_stats.pair_cache_hits << ','
           << verification_stats.feature_extractions << ','
           << (verification_stage == "verified" ? verification_ms : 0.0) << '\n';
  };
  write_row("JIST", "last-frame", "retrieval", jist_last_metrics, jist_threshold, jist_inference_ms);
  write_row("JIST", "last-frame", "verified", jist_last_verified_metrics, jist_threshold, jist_inference_ms);
  write_row("JIST", "temporal-any", "retrieval", jist_any_metrics, jist_threshold, jist_inference_ms);
  write_row("JIST", "temporal-any", "verified", jist_any_verified_metrics, jist_threshold, jist_inference_ms);
  write_row("JIST", "frame-argmax", "retrieval", jist_frame_argmax_metrics, jist_threshold, jist_inference_ms);
  write_row("JIST", "frame-argmax", "verified", jist_frame_argmax_verified_metrics, jist_threshold, jist_inference_ms);
  write_row("MixVPR", "last-frame", "retrieval", mixvpr_last_metrics, mixvpr_threshold, mixvpr_inference_ms);
  write_row("MixVPR", "last-frame", "verified", mixvpr_last_verified_metrics, mixvpr_threshold, mixvpr_inference_ms);
}

}  // namespace

int main(int argc, char** argv) {
  fs::path dataset_root = "/data/graco";
  std::vector<std::string> dataset_names = {
      "ground-01", "ground-02", "ground-03", "ground-04", "ground-05", "ground-06"};
  fs::path jist_engine_path = "onnx_model/trt/JIST_r18_512_seqgem_frames_fp16.engine";
  fs::path mixvpr_engine_path = "onnx_model/trt/mixvpr_resnet50_512d_fp16.engine";
  fs::path xfeat_engine_path = "onnx_model/trt/xfeat_320x224_fp16.engine";
  fs::path lighterglue_engine_path = "onnx_model/trt/lg_320x224_dyn_fp16.engine";
  size_t n_skip = 5;
  size_t n_seq = 5;
  size_t max_sequences = 0;
  double pose_max_dt = 0.02;
  double positive_distance = 5.0;
  double positive_yaw_degrees = 30.0;
  double min_time_separation = 30.0;
  double gt_sample_period = 1.0;
  double temporal_match_tolerance = 5.0;
  VerificationParams verification_params;
  double jist_threshold = 0.9;
  double mixvpr_threshold = 0.6;
  fs::path metrics_csv;
  fs::path gt_loops_csv;
  fs::path pr_curve_csv;
  size_t pr_threshold_count = 40;
  double pr_jist_threshold_minimum = 0.8;
  double pr_jist_threshold_maximum = 0.99;
  double pr_mixvpr_threshold_minimum = 0.4;
  double pr_mixvpr_threshold_maximum = 0.9;
  bool pr_verification_enabled = false;
  bool verbose = false;

  po::options_description options("Multi-dataset JIST versus MixVPR temporal loop-recall benchmark");
  options.add_options()("help,h", "Show this help message")(
      "dataset-root", po::value<fs::path>(&dataset_root)->default_value(dataset_root), "Root of the GRACO datasets")(
      "datasets",
      po::value<std::vector<std::string>>(&dataset_names)
          ->multitoken()
          ->default_value(dataset_names, "ground-01 ground-02 ground-03 ground-04 ground-05 ground-06"),
      "Dataset names; each resolves to <name>_images/camera_left_image_raw and <name>.txt")(
      "jist-engine", po::value<fs::path>(&jist_engine_path)->default_value(jist_engine_path), "JIST TensorRT engine")(
      "mixvpr-engine",
      po::value<fs::path>(&mixvpr_engine_path)->default_value(mixvpr_engine_path),
      "MixVPR ResNet-50 512-D TensorRT engine")(
      "xfeat-engine",
      po::value<fs::path>(&xfeat_engine_path)->default_value(xfeat_engine_path),
      "XFeat TensorRT engine used for geometric verification")(
      "lighterglue-engine",
      po::value<fs::path>(&lighterglue_engine_path)->default_value(lighterglue_engine_path),
      "LighterGlue TensorRT engine used for geometric verification")(
      "xfeat-top-k",
      po::value<int>(&verification_params.xfeat_top_k)->default_value(verification_params.xfeat_top_k),
      "Maximum XFeat keypoints per verification image")(
      "verification-min-matches",
      po::value<int>(&verification_params.minimum_matches)->default_value(verification_params.minimum_matches),
      "Minimum LightGlue matches before fundamental-matrix RANSAC")(
      "ransac-min-inliers",
      po::value<int>(&verification_params.minimum_inliers)->default_value(verification_params.minimum_inliers),
      "Minimum RANSAC fundamental-matrix inliers required to accept a retrieval")(
      "ransac-min-inlier-ratio",
      po::value<double>(&verification_params.minimum_inlier_ratio)
          ->default_value(verification_params.minimum_inlier_ratio),
      "Minimum RANSAC inlier-to-LightGlue-match ratio")(
      "ransac-reprojection-threshold",
      po::value<double>(&verification_params.ransac_reprojection_threshold)
          ->default_value(verification_params.ransac_reprojection_threshold),
      "Fundamental-matrix RANSAC reprojection threshold in source-image pixels")(
      "ransac-confidence",
      po::value<double>(&verification_params.ransac_confidence)->default_value(verification_params.ransac_confidence),
      "Fundamental-matrix RANSAC confidence")("ransac-max-iterations",
                                              po::value<int>(&verification_params.ransac_max_iterations)
                                                  ->default_value(verification_params.ransac_max_iterations),
                                              "Maximum fundamental-matrix RANSAC iterations")(
      "n-skip", po::value<size_t>(&n_skip)->default_value(n_skip), "Keep every n_skip-th image within each dataset")(
      "n-seq",
      po::value<size_t>(&n_seq)->default_value(n_seq),
      "Logical sampled-image group size; groups never cross dataset boundaries")(
      "positive-distance",
      po::value<double>(&positive_distance)->default_value(positive_distance),
      "Maximum 3-D translation distance in metres for a GT radius-search match")(
      "positive-yaw-degrees",
      po::value<double>(&positive_yaw_degrees)->default_value(positive_yaw_degrees),
      "Maximum absolute wrapped yaw difference in degrees for a GT match")(
      "min-time-separation",
      po::value<double>(&min_time_separation)->default_value(min_time_separation),
      "Minimum timestamp separation for same-dataset GT loops and detections; not applied across datasets")(
      "gt-sample-period",
      po::value<double>(&gt_sample_period)->default_value(gt_sample_period),
      "Pose-trajectory sampling period in seconds used for method-independent GT loop queries")(
      "temporal-match-tolerance",
      po::value<double>(&temporal_match_tolerance)->default_value(temporal_match_tolerance),
      "Per-endpoint timestamp tolerance in seconds for temporal-any matching")(
      "pose-max-dt",
      po::value<double>(&pose_max_dt)->default_value(pose_max_dt),
      "Maximum nearest-pose timestamp error for a sequence's final image")(
      "jist-threshold",
      po::value<double>(&jist_threshold)->default_value(jist_threshold),
      "JIST cosine-similarity detection threshold")(
      "mixvpr-threshold",
      po::value<double>(&mixvpr_threshold)->default_value(mixvpr_threshold),
      "MixVPR cosine-similarity detection threshold")(
      "max-sequences",
      po::value<size_t>(&max_sequences)->default_value(max_sequences),
      "Process at most this many groups per dataset; 0 processes all complete groups")(
      "metrics-csv", po::value<fs::path>(&metrics_csv), "Write eight pre/post-verification metric rows to this CSV")(
      "pr-curve-csv",
      po::value<fs::path>(&pr_curve_csv),
      "Write threshold-swept PR rows; verification is controlled by --pr-enable-verification")(
      "pr-enable-verification",
      po::bool_switch(&pr_verification_enabled),
      "Apply XFeat/LightGlue/RANSAC once to the PR sweep's lowest-threshold detection sets")(
      "pr-threshold-count",
      po::value<size_t>(&pr_threshold_count)->default_value(pr_threshold_count),
      "Number of linearly spaced thresholds per PR curve")(
      "pr-jist-threshold-min",
      po::value<double>(&pr_jist_threshold_minimum)->default_value(pr_jist_threshold_minimum),
      "Minimum JIST threshold for PR sweeping")(
      "pr-jist-threshold-max",
      po::value<double>(&pr_jist_threshold_maximum)->default_value(pr_jist_threshold_maximum),
      "Maximum JIST threshold for PR sweeping")(
      "pr-mixvpr-threshold-min",
      po::value<double>(&pr_mixvpr_threshold_minimum)->default_value(pr_mixvpr_threshold_minimum),
      "Minimum MixVPR threshold for PR sweeping")(
      "pr-mixvpr-threshold-max",
      po::value<double>(&pr_mixvpr_threshold_maximum)->default_value(pr_mixvpr_threshold_maximum),
      "Maximum MixVPR threshold for PR sweeping")(
      "gt-loops-csv", po::value<fs::path>(&gt_loops_csv), "Write GT loop queries and radius-search references to CSV")(
      "verbose", po::bool_switch(&verbose), "Enable verbose TensorRT logging");

  try {
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, options), variables);
    if (variables.count("help") != 0) {
      std::cout << options << '\n';
      return 0;
    }
    po::notify(variables);

    if (dataset_names.empty()) {
      throw std::invalid_argument("--datasets must contain at least one dataset name");
    }
    if (n_skip == 0) {
      throw std::invalid_argument("--n-skip must be positive");
    }
    if (n_seq == 0 && pr_curve_csv.empty()) {
      throw std::invalid_argument("--n-seq=0 is the framewise MixVPR PR reference and requires --pr-curve-csv");
    }
    if (pose_max_dt < 0.0 || positive_distance < 0.0 || positive_yaw_degrees < 0.0 || positive_yaw_degrees > 180.0 ||
        min_time_separation < 0.0 || gt_sample_period <= 0.0 || temporal_match_tolerance < 0.0) {
      throw std::invalid_argument("Pose, distance, sampling, and temporal thresholds are invalid");
    }
    if (temporal_match_tolerance < gt_sample_period) {
      throw std::invalid_argument("--temporal-match-tolerance must be at least --gt-sample-period");
    }
    if (jist_threshold < -1.0 || jist_threshold > 1.0 || mixvpr_threshold < -1.0 || mixvpr_threshold > 1.0) {
      throw std::invalid_argument("Cosine-similarity thresholds must be in [-1, 1]");
    }
    if (verification_params.xfeat_top_k <= 0 || verification_params.minimum_matches < 8 ||
        verification_params.minimum_inliers < 8 || verification_params.minimum_inlier_ratio < 0.0 ||
        verification_params.minimum_inlier_ratio > 1.0 || verification_params.ransac_reprojection_threshold <= 0.0 ||
        verification_params.ransac_confidence <= 0.0 || verification_params.ransac_confidence >= 1.0 ||
        verification_params.ransac_max_iterations <= 0) {
      throw std::invalid_argument("XFeat/LightGlue/RANSAC verification parameters are invalid");
    }
    if (pr_threshold_count < 2 || pr_jist_threshold_minimum < -1.0 || pr_jist_threshold_maximum > 1.0 ||
        pr_jist_threshold_minimum >= pr_jist_threshold_maximum || pr_mixvpr_threshold_minimum < -1.0 ||
        pr_mixvpr_threshold_maximum > 1.0 || pr_mixvpr_threshold_minimum >= pr_mixvpr_threshold_maximum) {
      throw std::invalid_argument("Precision-recall threshold sweep parameters are invalid");
    }

    std::vector<Dataset> datasets;
    datasets.reserve(dataset_names.size());
    for (const std::string& name : dataset_names) {
      Dataset dataset;
      dataset.name = name;
      dataset.image_dir = dataset_root / (name + "_images") / "camera_left_image_raw";
      dataset.pose_path = dataset_root / (name + ".txt");
      dataset.poses = readPoses(dataset.pose_path);
      datasets.push_back(std::move(dataset));
    }

    // Ground truth is deliberately extracted before models or method detections are created.
    std::vector<std::vector<Pose>> sampled_trajectories;
    sampled_trajectories.reserve(datasets.size());
    for (const Dataset& dataset : datasets) {
      sampled_trajectories.push_back(sampleTrajectory(dataset.poses, gt_sample_period));
    }
    const double positive_yaw_difference = positive_yaw_degrees * std::acos(-1.0) / 180.0;
    const std::vector<GroundTruthLoop> ground_truth_loops =
        extractGroundTruthLoops(sampled_trajectories, positive_distance, positive_yaw_difference, min_time_separation);
    writeGroundTruthLoopsCsv(gt_loops_csv, ground_truth_loops, datasets);

    size_t query_loops_with_self_references = 0;
    size_t query_loops_with_cross_references = 0;
    uint64_t self_references = 0;
    uint64_t cross_references = 0;
    for (const GroundTruthLoop& loop : ground_truth_loops) {
      bool has_self_reference = false;
      bool has_cross_reference = false;
      for (const GroundTruthReference& reference : loop.references) {
        if (reference.dataset == loop.query_dataset) {
          ++self_references;
          has_self_reference = true;
        } else {
          ++cross_references;
          has_cross_reference = true;
        }
      }
      query_loops_with_self_references += has_self_reference;
      query_loops_with_cross_references += has_cross_reference;
    }
    std::cout << "Method-independent GT radius search\n"
              << "  Datasets:             " << datasets.size() << '\n'
              << "  GT query sampling:    " << gt_sample_period << " s\n"
              << "  Positive radius:      " << positive_distance << " m\n"
              << "  Positive yaw error:   " << positive_yaw_degrees << " deg\n"
              << "  Same-run exclusion:   " << min_time_separation << " s\n"
              << "  GT loop queries:      " << ground_truth_loops.size() << '\n'
              << "  Queries with self refs:  " << query_loops_with_self_references << '\n'
              << "  Queries with cross refs: " << query_loops_with_cross_references << '\n'
              << "  Positive references: " << self_references + cross_references << " (" << self_references << " self, "
              << cross_references << " cross-run)\n"
              << "  Temporal-any window:  +/-" << temporal_match_tolerance << " s at both loop endpoints\n";
    if (!gt_loops_csv.empty()) {
      std::cout << "  GT-loop CSV:          " << gt_loops_csv << '\n';
    }
    std::cout << '\n';
    if (ground_truth_loops.empty()) {
      throw std::runtime_error("No GT loop queries were found; adjust --positive-distance or GT settings");
    }

    std::vector<Sequence> sequences;
    size_t source_image_count = 0;
    double maximum_pose_dt = 0.0;
    std::cout << "Image datasets and method inputs\n";
    for (size_t dataset_index = 0; dataset_index < datasets.size(); ++dataset_index) {
      const Dataset& dataset = datasets[dataset_index];
      const std::vector<fs::path> images = listImages(dataset.image_dir);
      source_image_count += images.size();
      const size_t logical_group_size = n_seq == 0 ? 1 : n_seq;
      std::vector<Sequence> dataset_sequences =
          makeSequences(images, dataset.poses, dataset_index, n_skip, logical_group_size, max_sequences, pose_max_dt);
      for (const Sequence& sequence : dataset_sequences) {
        for (const double pose_dt : sequence.frame_pose_dts) {
          maximum_pose_dt = std::max(maximum_pose_dt, std::abs(pose_dt));
        }
      }
      std::cout << "  " << std::left << std::setw(12) << dataset.name << std::right << std::setw(7) << images.size()
                << " images -> " << dataset_sequences.size() << " groups\n";
      sequences.insert(sequences.end(),
                       std::make_move_iterator(dataset_sequences.begin()),
                       std::make_move_iterator(dataset_sequences.end()));
    }
    if (sequences.size() < 2) {
      throw std::runtime_error("Fewer than two complete sequences are available for comparison");
    }

    std::cout << "  Total:        " << source_image_count << " images -> " << sequences.size()
              << (n_seq == 0 ? " framewise samples\n" : " groups\n") << "  Sampling:     keep every " << n_skip
              << "-th image";
    if (n_seq == 0) {
      std::cout << "; n_seq=0 uses every retained image independently\n";
    } else {
      std::cout << ", then non-overlapping n_seq=" << n_seq << '\n';
    }
    std::cout << "  Max pose |dt|: " << std::fixed << std::setprecision(6) << maximum_pose_dt << " s\n\n";

    if (!fs::is_regular_file(mixvpr_engine_path)) {
      throw std::runtime_error("MixVPR TensorRT engine is not readable: " + mixvpr_engine_path.string());
    }
    if (n_seq != 0 && !fs::is_regular_file(jist_engine_path)) {
      throw std::runtime_error("JIST TensorRT engine is not readable: " + jist_engine_path.string());
    }

    if (n_seq == 0) {
      xfeat::MixVPRTRT::Params mixvpr_params;
      mixvpr_params.model_path = mixvpr_engine_path.string();
      mixvpr_params.verbose = verbose;
      xfeat::MixVPRTRT mixvpr(mixvpr_params);
      if (mixvpr.get_descriptor_dim() != 512) {
        throw std::runtime_error("MixVPR engine must have a 512-D output; loaded engine reports " +
                                 std::to_string(mixvpr.get_descriptor_dim()));
      }

      cv::Mat mixvpr_descriptors(static_cast<int>(sequences.size()), mixvpr.get_descriptor_dim(), CV_32F);
      double mixvpr_inference_ms = 0.0;
      std::cout << "TensorRT framewise MixVPR inference (FP16 engine with FP32 I/O)\n"
                << "  Extracting " << sequences.size() << " every-n_skip frame descriptors...\n";
      for (size_t index = 0; index < sequences.size(); ++index) {
        const fs::path& image_path = sequences[index].images.back();
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
          throw std::runtime_error("Failed to read image: " + image_path.string());
        }
        const auto start = std::chrono::steady_clock::now();
        cv::Mat descriptor = mixvpr.infer(image);
        mixvpr_inference_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        descriptor.copyTo(mixvpr_descriptors.row(static_cast<int>(index)));
        if ((index + 1) % 250 == 0 || index + 1 == sequences.size()) {
          std::cout << "    " << (index + 1) << " / " << sequences.size() << '\n';
        }
      }

      const RetrievalResult mixvpr_framewise = retrieveMethod(mixvpr_descriptors,
                                                              sequences,
                                                              pr_mixvpr_threshold_minimum,
                                                              positive_distance,
                                                              positive_yaw_difference,
                                                              min_time_separation);
      RetrievalResult mixvpr_output;
      std::string verification_stage = "off";
      std::cout << "\nFramewise MixVPR precision-recall reference\n"
                << "  Thresholds:       " << pr_mixvpr_threshold_maximum << " -> " << pr_mixvpr_threshold_minimum
                << '\n'
                << "  Points:           " << pr_threshold_count << '\n'
                << "  Base detections:  " << mixvpr_framewise.detections.size() << '\n';

      if (pr_verification_enabled) {
        if (!fs::is_regular_file(xfeat_engine_path)) {
          throw std::runtime_error("XFeat TensorRT engine is not readable: " + xfeat_engine_path.string());
        }
        if (!fs::is_regular_file(lighterglue_engine_path)) {
          throw std::runtime_error("LighterGlue TensorRT engine is not readable: " + lighterglue_engine_path.string());
        }
        xfeat::XFeatTRT::Params xfeat_params;
        xfeat_params.engine_path = xfeat_engine_path.string();
        xfeat_params.nkpts = verification_params.xfeat_top_k;
        xfeat_params.verbose = verbose;
        xfeat::XFeatTRT xfeat_extractor(xfeat_params);
        xfeat::LighterGlueTRT lighterglue(lighterglue_engine_path.string(), verbose);
        GeometricVerifier verifier(xfeat_extractor, lighterglue, verification_params);
        mixvpr_output = verifier.filter(mixvpr_framewise);
        const VerificationStats& stats = verifier.stats();
        const double verification_ms = stats.feature_extraction_ms + stats.lighterglue_ms + stats.ransac_ms;
        verification_stage = "verified";
        std::cout << "  Verified detections: " << mixvpr_output.detections.size() << '\n'
                  << "  Verification work: requested=" << stats.requested_pairs << ", unique=" << stats.unique_pairs
                  << ", pair-cache-hits=" << stats.pair_cache_hits << ", XFeat-images=" << stats.feature_extractions
                  << ", timed-ms=" << verification_ms << '\n';
      } else {
        mixvpr_output = mixvpr_framewise;
        std::cout << "  Verification: OFF (XFeat/LightGlue engines were not loaded)\n";
      }

      const RetrievalResult empty_retrieval;
      writePrecisionRecallCurveCsv(pr_curve_csv,
                                   n_seq,
                                   ground_truth_loops,
                                   gt_sample_period,
                                   temporal_match_tolerance,
                                   empty_retrieval,
                                   empty_retrieval,
                                   mixvpr_output,
                                   pr_jist_threshold_minimum,
                                   pr_jist_threshold_maximum,
                                   pr_mixvpr_threshold_minimum,
                                   pr_mixvpr_threshold_maximum,
                                   pr_threshold_count,
                                   verification_stage);
      std::cout << "  Inference:        " << mixvpr_inference_ms << " ms\n"
                << "  Wrote PR CSV:     " << pr_curve_csv << '\n';
      return 0;
    }

    xfeat::JistTRT::Params jist_params;
    jist_params.model_path = jist_engine_path.string();
    jist_params.verbose = verbose;
    xfeat::JistTRT jist(jist_params);
    const size_t jist_input_frames = static_cast<size_t>(jist.get_seq_length());
    if (n_seq < jist_input_frames) {
      throw std::runtime_error("--n-seq=" + std::to_string(n_seq) + " is smaller than the JIST engine input length " +
                               std::to_string(jist_input_frames));
    }
    if (jist.get_descriptor_dim() != 512) {
      throw std::runtime_error("JIST engine must have a 512-D output; loaded engine reports " +
                               std::to_string(jist.get_descriptor_dim()));
    }
    if (!jist.has_frame_descriptors()) {
      throw std::runtime_error("JIST engine must expose normalized [sequence, 512] frame descriptors");
    }

    xfeat::MixVPRTRT::Params mixvpr_params;
    mixvpr_params.model_path = mixvpr_engine_path.string();
    mixvpr_params.verbose = verbose;
    xfeat::MixVPRTRT mixvpr(mixvpr_params);
    if (mixvpr.get_descriptor_dim() != 512) {
      throw std::runtime_error("MixVPR engine must have a 512-D output; loaded engine reports " +
                               std::to_string(mixvpr.get_descriptor_dim()));
    }

    cv::Mat jist_descriptors(static_cast<int>(sequences.size()), jist.get_descriptor_dim(), CV_32F);
    cv::Mat jist_frame_descriptors(
        static_cast<int>(sequences.size() * jist_input_frames), jist.get_descriptor_dim(), CV_32F);
    cv::Mat mixvpr_descriptors(static_cast<int>(sequences.size()), mixvpr.get_descriptor_dim(), CV_32F);
    double jist_inference_ms = 0.0;
    double mixvpr_inference_ms = 0.0;

    std::cout << "TensorRT inference (both FP16 engines with FP32 I/O)\n"
              << "  JIST:   " << jist_input_frames << " uniformly spaced images across each logical group\n"
              << "  MixVPR: final image of the same logical group\n"
              << "  Extracting " << sequences.size() << " descriptor pairs...\n";
    for (size_t index = 0; index < sequences.size(); ++index) {
      const std::vector<cv::Mat> images = loadJistImages(sequences[index], jist_input_frames);

      auto start = std::chrono::steady_clock::now();
      xfeat::JistTRT::InferenceResult jist_result = jist.infer_with_frame_descriptors(images);
      jist_inference_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      jist_result.sequence_descriptor.copyTo(jist_descriptors.row(static_cast<int>(index)));
      const int first_frame_row = static_cast<int>(index * jist_input_frames);
      jist_result.frame_descriptors.copyTo(
          jist_frame_descriptors.rowRange(first_frame_row, first_frame_row + static_cast<int>(jist_input_frames)));

      start = std::chrono::steady_clock::now();
      cv::Mat mixvpr_descriptor = mixvpr.infer(images.back());
      mixvpr_inference_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      mixvpr_descriptor.copyTo(mixvpr_descriptors.row(static_cast<int>(index)));

      if ((index + 1) % 50 == 0 || index + 1 == sequences.size()) {
        std::cout << "    " << (index + 1) << " / " << sequences.size() << '\n';
      }
    }

    if (!pr_curve_csv.empty()) {
      std::cout << "\nPrecision-recall sweep (verification " << (pr_verification_enabled ? "enabled" : "off") << ")\n"
                << "  JIST thresholds:   " << pr_jist_threshold_maximum << " -> " << pr_jist_threshold_minimum << '\n'
                << "  MixVPR thresholds: " << pr_mixvpr_threshold_maximum << " -> " << pr_mixvpr_threshold_minimum
                << '\n'
                << "  Points per curve:  " << pr_threshold_count << '\n';
      const RetrievalResult jist_last_pr = retrieveMethod(jist_descriptors,
                                                          sequences,
                                                          pr_jist_threshold_minimum,
                                                          positive_distance,
                                                          positive_yaw_difference,
                                                          min_time_separation);
      const RetrievalResult jist_frame_argmax_pr = retrieveFrameRefinedJist(jist_descriptors,
                                                                            jist_frame_descriptors,
                                                                            sequences,
                                                                            jist_input_frames,
                                                                            pr_jist_threshold_minimum,
                                                                            positive_distance,
                                                                            positive_yaw_difference,
                                                                            min_time_separation);
      const RetrievalResult mixvpr_last_pr = retrieveMethod(mixvpr_descriptors,
                                                            sequences,
                                                            pr_mixvpr_threshold_minimum,
                                                            positive_distance,
                                                            positive_yaw_difference,
                                                            min_time_separation);
      const auto write_pr_curves = [&](const RetrievalResult& jist_last,
                                       const RetrievalResult& jist_refined,
                                       const RetrievalResult& mixvpr_last,
                                       const std::string& verification_stage) {
        writePrecisionRecallCurveCsv(pr_curve_csv,
                                     n_seq,
                                     ground_truth_loops,
                                     gt_sample_period,
                                     temporal_match_tolerance,
                                     jist_last,
                                     jist_refined,
                                     mixvpr_last,
                                     pr_jist_threshold_minimum,
                                     pr_jist_threshold_maximum,
                                     pr_mixvpr_threshold_minimum,
                                     pr_mixvpr_threshold_maximum,
                                     pr_threshold_count,
                                     verification_stage);
      };
      std::cout << "  Base detections: JIST=" << jist_last_pr.detections.size()
                << ", JIST-refined=" << jist_frame_argmax_pr.detections.size()
                << ", MixVPR=" << mixvpr_last_pr.detections.size() << '\n';

      if (pr_verification_enabled) {
        if (!fs::is_regular_file(xfeat_engine_path)) {
          throw std::runtime_error("XFeat TensorRT engine is not readable: " + xfeat_engine_path.string());
        }
        if (!fs::is_regular_file(lighterglue_engine_path)) {
          throw std::runtime_error("LighterGlue TensorRT engine is not readable: " + lighterglue_engine_path.string());
        }
        xfeat::XFeatTRT::Params xfeat_params;
        xfeat_params.engine_path = xfeat_engine_path.string();
        xfeat_params.nkpts = verification_params.xfeat_top_k;
        xfeat_params.verbose = verbose;
        xfeat::XFeatTRT xfeat_extractor(xfeat_params);
        xfeat::LighterGlueTRT lighterglue(lighterglue_engine_path.string(), verbose);
        GeometricVerifier verifier(xfeat_extractor, lighterglue, verification_params);
        const RetrievalResult jist_last_verified = verifier.filter(jist_last_pr);
        const RetrievalResult jist_frame_argmax_verified = verifier.filter(jist_frame_argmax_pr);
        const RetrievalResult mixvpr_last_verified = verifier.filter(mixvpr_last_pr);
        const VerificationStats& stats = verifier.stats();
        write_pr_curves(jist_last_verified, jist_frame_argmax_verified, mixvpr_last_verified, "verified");
        const double verification_ms = stats.feature_extraction_ms + stats.lighterglue_ms + stats.ransac_ms;
        std::cout << "  Verified detections: JIST=" << jist_last_verified.detections.size()
                  << ", JIST-refined=" << jist_frame_argmax_verified.detections.size()
                  << ", MixVPR=" << mixvpr_last_verified.detections.size() << '\n'
                  << "  Verification work: requested=" << stats.requested_pairs << ", unique=" << stats.unique_pairs
                  << ", pair-cache-hits=" << stats.pair_cache_hits << ", XFeat-images=" << stats.feature_extractions
                  << ", timed-ms=" << verification_ms << '\n';
      } else {
        write_pr_curves(jist_last_pr, jist_frame_argmax_pr, mixvpr_last_pr, "off");
        std::cout << "  Verification: OFF (XFeat/LightGlue engines were not loaded)\n";
      }
      std::cout << "  Wrote PR CSV: " << pr_curve_csv << '\n';
      return 0;
    }

    const RetrievalResult jist_last_retrieval = retrieveMethod(
        jist_descriptors, sequences, jist_threshold, positive_distance, positive_yaw_difference, min_time_separation);
    const RetrievalResult jist_frame_argmax_retrieval = retrieveFrameRefinedJist(jist_descriptors,
                                                                                 jist_frame_descriptors,
                                                                                 sequences,
                                                                                 jist_input_frames,
                                                                                 jist_threshold,
                                                                                 positive_distance,
                                                                                 positive_yaw_difference,
                                                                                 min_time_separation);
    const RetrievalResult mixvpr_last_retrieval = retrieveMethod(mixvpr_descriptors,
                                                                 sequences,
                                                                 mixvpr_threshold,
                                                                 positive_distance,
                                                                 positive_yaw_difference,
                                                                 min_time_separation);

    if (!fs::is_regular_file(xfeat_engine_path)) {
      throw std::runtime_error("XFeat TensorRT engine is not readable: " + xfeat_engine_path.string());
    }
    if (!fs::is_regular_file(lighterglue_engine_path)) {
      throw std::runtime_error("LighterGlue TensorRT engine is not readable: " + lighterglue_engine_path.string());
    }

    xfeat::XFeatTRT::Params xfeat_params;
    xfeat_params.engine_path = xfeat_engine_path.string();
    xfeat_params.nkpts = verification_params.xfeat_top_k;
    xfeat_params.verbose = verbose;
    xfeat::XFeatTRT xfeat_extractor(xfeat_params);
    xfeat::LighterGlueTRT lighterglue(lighterglue_engine_path.string(), verbose);
    GeometricVerifier verifier(xfeat_extractor, lighterglue, verification_params);

    std::cout << "\nXFeat + LightGlue + fundamental-matrix RANSAC verification\n"
              << "  XFeat top-k:          " << verification_params.xfeat_top_k << '\n'
              << "  Minimum matches:     " << verification_params.minimum_matches << '\n'
              << "  Minimum inliers:     " << verification_params.minimum_inliers << '\n'
              << "  Minimum inlier ratio: " << verification_params.minimum_inlier_ratio << '\n'
              << "  RANSAC threshold:    " << verification_params.ransac_reprojection_threshold << " px\n";
    const RetrievalResult jist_last_verified = verifier.filter(jist_last_retrieval);
    const RetrievalResult jist_frame_argmax_verified = verifier.filter(jist_frame_argmax_retrieval);
    const RetrievalResult mixvpr_last_verified = verifier.filter(mixvpr_last_retrieval);
    const VerificationStats& verification_stats = verifier.stats();

    const auto evaluate = [&](const RetrievalResult& retrieval, EvaluationMode mode) {
      return evaluateDetections(retrieval.detections,
                                retrieval.candidate_pairs,
                                ground_truth_loops,
                                gt_sample_period,
                                temporal_match_tolerance,
                                mode);
    };
    const Metrics jist_last_metrics = evaluate(jist_last_retrieval, EvaluationMode::kLastFrame);
    const Metrics jist_last_verified_metrics = evaluate(jist_last_verified, EvaluationMode::kLastFrame);
    const Metrics jist_any_metrics = evaluate(jist_last_retrieval, EvaluationMode::kTemporalAny);
    const Metrics jist_any_verified_metrics = evaluate(jist_last_verified, EvaluationMode::kTemporalAny);
    const Metrics jist_frame_argmax_metrics = evaluate(jist_frame_argmax_retrieval, EvaluationMode::kLastFrame);
    const Metrics jist_frame_argmax_verified_metrics = evaluate(jist_frame_argmax_verified, EvaluationMode::kLastFrame);
    const Metrics mixvpr_last_metrics = evaluate(mixvpr_last_retrieval, EvaluationMode::kLastFrame);
    const Metrics mixvpr_last_verified_metrics = evaluate(mixvpr_last_verified, EvaluationMode::kLastFrame);

    const MetricRates jist_last_rates = rates(jist_last_metrics);
    const MetricRates jist_any_rates = rates(jist_any_metrics);
    if (jist_any_rates.recall + 1e-12 < jist_last_rates.recall ||
        jist_any_rates.precision + 1e-12 < jist_last_rates.precision) {
      throw std::logic_error("Temporal-any JIST metrics must be supersets of last-frame metrics");
    }
    if (jist_last_verified_metrics.predicted_positives > jist_last_metrics.predicted_positives ||
        jist_frame_argmax_verified_metrics.predicted_positives > jist_frame_argmax_metrics.predicted_positives ||
        mixvpr_last_verified_metrics.predicted_positives > mixvpr_last_metrics.predicted_positives) {
      throw std::logic_error("Geometric verification cannot add retrieval detections");
    }

    std::cout << "\nInference time (image loading excluded)\n"
              << "  JIST:   " << std::fixed << std::setprecision(2) << jist_inference_ms << " ms total, "
              << jist_inference_ms / sequences.size() << " ms/group\n"
              << "  MixVPR: " << mixvpr_inference_ms << " ms total, " << mixvpr_inference_ms / sequences.size()
              << " ms/final frame\n\n";

    const double verification_ms =
        verification_stats.feature_extraction_ms + verification_stats.lighterglue_ms + verification_stats.ransac_ms;
    std::cout << "Geometric-verification work (cache shared across variants)\n"
              << "  Requested pairs:     " << verification_stats.requested_pairs << '\n'
              << "  Accepted requests:   " << verification_stats.accepted_requests << '\n'
              << "  Unique pairs:        " << verification_stats.unique_pairs << '\n'
              << "  Pair-cache hits:     " << verification_stats.pair_cache_hits << '\n'
              << "  XFeat images:        " << verification_stats.feature_extractions << '\n'
              << "  Timed model/RANSAC:  " << std::fixed << std::setprecision(2) << verification_ms << " ms (XFeat "
              << verification_stats.feature_extraction_ms << ", LightGlue " << verification_stats.lighterglue_ms
              << ", RANSAC " << verification_stats.ransac_ms << ")\n\n";

    std::cout << "Fixed-GT sampled-query results\n"
              << std::left << std::setw(18) << "Method / mode" << std::right << std::setw(10) << "Threshold"
              << std::setw(10) << "GT loops" << std::setw(12) << "Recall" << std::setw(12) << "Precision"
              << std::setw(12) << "F1" << std::setw(11) << "GT hit" << std::setw(12) << "Correct det" << std::setw(10)
              << "FP" << '\n';
    printMetrics("JIST-last", jist_threshold, jist_last_metrics);
    printMetrics("JIST-last+GV", jist_threshold, jist_last_verified_metrics);
    printMetrics("JIST-any(+/-5s)", jist_threshold, jist_any_metrics);
    printMetrics("JIST-any+GV", jist_threshold, jist_any_verified_metrics);
    printMetrics("JIST-frame-argmax", jist_threshold, jist_frame_argmax_metrics);
    printMetrics("JIST-argmax+GV", jist_threshold, jist_frame_argmax_verified_metrics);
    printMetrics("MixVPR-last", mixvpr_threshold, mixvpr_last_metrics);
    printMetrics("MixVPR-last+GV", mixvpr_threshold, mixvpr_last_verified_metrics);

    if (!metrics_csv.empty()) {
      writeMetricsCsv(metrics_csv,
                      datasets.size(),
                      n_skip,
                      n_seq,
                      jist_input_frames,
                      sequences.size(),
                      gt_sample_period,
                      temporal_match_tolerance,
                      positive_yaw_degrees,
                      jist_last_metrics,
                      jist_last_verified_metrics,
                      jist_any_metrics,
                      jist_any_verified_metrics,
                      jist_frame_argmax_metrics,
                      jist_frame_argmax_verified_metrics,
                      mixvpr_last_metrics,
                      mixvpr_last_verified_metrics,
                      jist_threshold,
                      mixvpr_threshold,
                      jist_inference_ms,
                      mixvpr_inference_ms,
                      verification_params,
                      verification_stats);
      std::cout << "\nWrote metrics CSV: " << metrics_csv << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
