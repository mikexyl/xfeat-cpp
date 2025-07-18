#pragma once

#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <string>

namespace xfeat {

class FaissDatabase {
 public:
  using QueryResults = std::vector<faiss::idx_t>;
  using QueryDistances = std::vector<float>;

  // Load a FAISS index from file, use GPU if available
  explicit FaissDatabase(const std::string& index_path) {
    std::unique_ptr<faiss::Index> cpu_index(faiss::read_index(index_path.c_str()));
    if (!cpu_index) {
      throw std::runtime_error("Failed to load FAISS index from " + index_path);
    }
    res_ = std::make_unique<faiss::gpu::StandardGpuResources>();
    faiss::gpu::GpuClonerOptions opts;
    opts.useFloat16 = false;
    index_.reset(faiss::gpu::index_cpu_to_gpu(res_.get(), 0, cpu_index.get(), &opts));
  }

  // Add descriptors to the index
  void add(const cv::Mat& descriptors) {
    CV_Assert(not descriptors.empty());
    if (descriptors.empty()) return;
    if (descriptors.type() != CV_32F) {
      throw std::runtime_error("Descriptors must be of type CV_32F");
    }
    index_->add(1, (float*)descriptors.data);
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

    faiss::SearchParametersIVF search_params;
    if (max_search_id >= 0) {
      faiss::IDSelectorRange range_selector(0, max_search_id);
      search_params.sel = &range_selector;
    }

    // check query has only 1 channel
    CV_Assert(query.channels() == 1);
    CV_Assert(query.type() == CV_32F);
    CV_Assert(query.rows == 1);
    CV_Assert(query.cols == index_->d);  // query must match index dimension

    index_->search(1, query.ptr<float>(), k, distances.data(), indices.data(), &search_params);
  }

  // Save the index to file
  void save(const std::string& path) const {
    // Move index back to CPU for saving
    std::unique_ptr<faiss::Index> cpu_index(faiss::gpu::index_gpu_to_cpu(index_.get()));
    faiss::write_index(cpu_index.get(), path.c_str());
  }

  // Get the dimension of the index
  int dim() const { return index_ ? index_->d : 0; }

  float l2_distance(const cv::Mat& a, const cv::Mat& b) const {
    return faiss::fvec_L2sqr(a.ptr<float>(), b.ptr<float>(), a.cols);
  }

 private:
  std::unique_ptr<faiss::Index> index_;
  std::unique_ptr<faiss::gpu::StandardGpuResources> res_;
};

}  // namespace xfeat
