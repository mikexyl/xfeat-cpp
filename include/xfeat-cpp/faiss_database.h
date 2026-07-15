#pragma once

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/Heap.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#if FAISS_VERSION_MAJOR == 1 && FAISS_VERSION_MINOR <= 7
namespace faiss {
using idx_t = Index::idx_t;
}
#endif

namespace xfeat {

class FaissDatabase {
 public:
  using QueryResults = std::vector<faiss::idx_t>;
  using QueryDistances = std::vector<float>;

  // JIST descriptors use a runtime CPU IndexFlatIP database. No trained
  // index is loaded or transferred to a GPU.
  explicit FaissDatabase(int dim) {
    if (dim <= 0) {
      throw std::invalid_argument(
          "FaissDatabase requires a positive descriptor dimension");
    }
    index_ = std::make_unique<faiss::IndexFlatIP>(dim);
  }

  // Add descriptors to the index
  faiss::idx_t add(const cv::Mat& descriptors) { return add_with_id(index_->ntotal, descriptors); }

  faiss::idx_t add_with_id(size_t id, const cv::Mat& descriptors) {
    CV_Assert(not descriptors.empty());
    if (descriptors.empty()) return -1;
    if (descriptors.type() != CV_32F) {
      throw std::runtime_error("Descriptors must be of type CV_32F");
    }
    if (descriptors.rows != 1 || descriptors.channels() != 1 ||
        descriptors.cols != index_->d) {
      throw std::invalid_argument(
          "Descriptor shape does not match FAISS index dimension: rows=" +
          std::to_string(descriptors.rows) + ", channels=" +
          std::to_string(descriptors.channels()) + ", cols=" +
          std::to_string(descriptors.cols) + ", expected cols=" +
          std::to_string(index_->d));
    }
    if (id_to_index_map_.find(id) != id_to_index_map_.end()) {
      throw std::runtime_error("Descriptor with the same id already exists in the index");
    }
    faiss::idx_t faiss_id = index_->ntotal;
    id_to_index_map_.emplace(id, faiss_id);
    index_->add(1, (float*)descriptors.data);
    return faiss_id;
  }

  cv::Mat get(size_t id) const {
    if (id_to_index_map_.count(id) == 0) {
      return cv::Mat();  // Return an empty Mat if id not found
    }
    faiss::idx_t faiss_id = id_to_index_map_.at(id);
    cv::Mat descriptor(1, index_->d, CV_32F);
    index_->reconstruct(faiss_id, descriptor.ptr<float>());
    return descriptor;
  }

  // Search for k nearest neighbors
  void search(const cv::Mat& query,
              int k,
              QueryResults& indices,
              QueryDistances& distances,
              int max_search_id = -1) const {
    CV_Assert(not query.empty());
    if (query.empty()) return;
    if (query.type() != CV_32F) {
      throw std::runtime_error("Query must be of type CV_32F");
    }

    // check query has only 1 channel
    CV_Assert(query.channels() == 1);
    CV_Assert(query.type() == CV_32F);
    CV_Assert(query.rows == 1);
    if (query.cols != index_->d) {
      throw std::runtime_error("Query dimension does not match index dimension: " + std::to_string(query.cols) +
                               " vs " + std::to_string(index_->d));
    }
    if (k <= 0 || indices.size() < static_cast<size_t>(k) ||
        distances.size() < static_cast<size_t>(k)) {
      throw std::invalid_argument(
          "FAISS search requires positive k and result buffers of size k");
    }

    const size_t search_count =
        max_search_id < 0
            ? static_cast<size_t>(index_->ntotal)
            : std::min(static_cast<size_t>(index_->ntotal),
                       static_cast<size_t>(max_search_id));
    std::fill(indices.begin(), indices.begin() + k, faiss::idx_t{-1});
    std::fill(distances.begin(),
              distances.begin() + k,
              std::numeric_limits<float>::lowest());
    if (search_count == 0) {
      return;
    }

    faiss::float_minheap_array_t result{
        1, static_cast<size_t>(k), indices.data(), distances.data()};
    faiss::knn_inner_product(query.ptr<float>(),
                             index_->get_xb(),
                             static_cast<size_t>(index_->d),
                             1,
                             search_count,
                             &result);
  }

  // Save the index to file
  void save(const std::string& path) const {
    faiss::write_index(index_.get(), path.c_str());
  }

  // Get the dimension of the index
  int dim() const { return index_ ? index_->d : 0; }

  float cosine_similarity(const cv::Mat& a, const cv::Mat& b) const {
    return faiss::fvec_inner_product(a.ptr<float>(), b.ptr<float>(), a.cols);
  }

  float l2_distance(const cv::Mat& a, const cv::Mat& b) const {
    return faiss::fvec_L2sqr(a.ptr<float>(), b.ptr<float>(), a.cols);
  }

  auto nTotal() const { return index_ ? index_->ntotal : 0; }

 private:
  std::unique_ptr<faiss::IndexFlatIP> index_;
  std::unordered_map<size_t, faiss::idx_t> id_to_index_map_;
};

}  // namespace xfeat
