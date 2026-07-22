#pragma once

#ifdef HAVE_TENSORRT

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xfeat-cpp/types.h"

namespace xfeat {
namespace trt_detail {
class Engine;
}

class LighterGlueTRT {
 public:
  explicit LighterGlueTRT(const std::string& engine_path, bool verbose = false);
  ~LighterGlueTRT();

  void run(const std::vector<float>& mkpts0,
           const std::vector<float>& feats0,
           const std::array<float, 2>& image0_size,
           const std::vector<float>& mkpts1,
           const std::vector<float>& feats1,
           const std::array<float, 2>& image1_size,
           std::vector<std::array<int64_t, 2>>& matches,
           std::vector<float>& scores);

  std::pair<std::vector<std::array<int64_t, 2>>, std::vector<float>> match(const std::vector<float>& mkpts0,
                                                                           const std::vector<float>& feats0,
                                                                           const std::array<float, 2>& image0_size,
                                                                           const std::vector<float>& mkpts1,
                                                                           const std::vector<float>& feats1,
                                                                           const std::array<float, 2>& image1_size);

  std::vector<std::vector<int>> match(const DetectionResult& det0,
                                      const std::array<float, 2>& image0_size,
                                      const DetectionResult& det1,
                                      const std::array<float, 2>& image1_size,
                                      float min_score = -1.0f,
                                      std::vector<float>* scores_out = nullptr);

 private:
  std::unique_ptr<trt_detail::Engine> engine_;
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
