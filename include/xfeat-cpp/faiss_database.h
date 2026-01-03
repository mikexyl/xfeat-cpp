#pragma once

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace xfeat {

class FaissDatabase {
 public:
  using QueryResults = std::vector<faiss::idx_t>;
  using QueryDistances = std::vector<float>;

  enum class IndexMode {
    kFlat = 0,
    kIVFFlat = 1,
  };

  // Load a FAISS index from file. Optionally use GPU (default: true)
  explicit FaissDatabase(IndexMode mode = IndexMode::kFlat,
                         const std::string& index_path = {},
                         bool use_gpu = true,
                         int dim = 0)
      : use_gpu_(use_gpu) {
    if (mode == IndexMode::kIVFFlat) {
      // Always read the index on CPU first
      std::unique_ptr<faiss::Index> cpu_index(faiss::read_index(index_path.c_str()));
      if (!cpu_index) {
        throw std::runtime_error("Failed to load FAISS index from " + index_path);
      }
      if (use_gpu_) {
        // Move to GPU
        res_ = std::make_unique<faiss::gpu::StandardGpuResources>();
        res_->setTempMemory(0);
        faiss::gpu::GpuClonerOptions opts;
        opts.useFloat16 = true;
        index_.reset(faiss::gpu::index_cpu_to_gpu(res_.get(), 0, cpu_index.get(), &opts));
      } else {
        // Keep on CPU
        index_ = std::move(cpu_index);
      }
    } else if (mode == IndexMode::kFlat) {
      // Create a flat index with inner product (cosine similarity for normalized vectors)
      if (use_gpu_) {
        res_ = std::make_unique<faiss::gpu::StandardGpuResources>();
        res_->setTempMemory(0);
        faiss::IndexFlatIP* flat_index = new faiss::IndexFlatIP(dim);  // Inner product for cosine similarity
        faiss::gpu::GpuClonerOptions opts;
        opts.useFloat16 = true;
        index_.reset(faiss::gpu::index_cpu_to_gpu(res_.get(), 0, flat_index, &opts));
      } else {
        faiss::IndexFlatIP* flat_index = new faiss::IndexFlatIP(dim);  // Inner product for cosine similarity
        index_.reset(flat_index);
      }
    } else {
      throw std::invalid_argument("Unsupported IndexMode");
    }
  }

  // Add descriptors to the index
  faiss::idx_t add(const cv::Mat& descriptors) { return add_with_id(index_->ntotal, descriptors); }

  faiss::idx_t add_with_id(size_t id, const cv::Mat& descriptors) {
    CV_Assert(not descriptors.empty());
    if (descriptors.empty()) return -1;
    if (descriptors.type() != CV_32F) {
      throw std::runtime_error("Descriptors must be of type CV_32F");
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

    faiss::SearchParametersIVF search_params;
    if (max_search_id >= 0) {
      faiss::IDSelectorRange range_selector(0, max_search_id);
      search_params.sel = &range_selector;
    }

    // check query has only 1 channel
    CV_Assert(query.channels() == 1);
    CV_Assert(query.type() == CV_32F);
    CV_Assert(query.rows == 1);
    if (query.cols != index_->d) {
      throw std::runtime_error("Query dimension does not match index dimension: " + std::to_string(query.cols) +
                               " vs " + std::to_string(index_->d));
    }

    if (not use_gpu_) {
      // GPU variant: pass the search parameters (used by IVF)
      index_->search(1, query.ptr<float>(), k, distances.data(), indices.data(), &search_params);
    } else {
      // CPU variant: call the CPU overload (SearchParameters pointer isn't supported on CPU base)
      index_->search(1, query.ptr<float>(), k, distances.data(), indices.data());
    }
  }

  // Save the index to file
  void save(const std::string& path) const {
    if (use_gpu_) {
      // Move index back to CPU for saving
      std::unique_ptr<faiss::Index> cpu_index(faiss::gpu::index_gpu_to_cpu(index_.get()));
      faiss::write_index(cpu_index.get(), path.c_str());
    } else {
      // Index is already on CPU
      faiss::write_index(index_.get(), path.c_str());
    }
  }

  // Get the dimension of the index
  int dim() const { return index_ ? index_->d : 0; }

  float cosine_similarity(const cv::Mat& a, const cv::Mat& b) const {
    return faiss::fvec_inner_product(a.ptr<float>(), b.ptr<float>(), a.cols);
  }

  auto nTotal() const { return index_ ? index_->ntotal : 0; }

 private:
  std::unique_ptr<faiss::Index> index_;
  std::unique_ptr<faiss::gpu::StandardGpuResources> res_;
  bool use_gpu_ = true;
  std::unordered_map<size_t, faiss::idx_t> id_to_index_map_;
};

}  // namespace xfeat
