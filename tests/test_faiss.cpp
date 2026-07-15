#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "xfeat-cpp/faiss_database.h"

namespace xfeat {
namespace {

TEST(FaissDatabase, RequiresPositiveDimension) {
  EXPECT_THROW(FaissDatabase(0), std::invalid_argument);
}

TEST(FaissDatabase, RejectsWrongDescriptorShape) {
  FaissDatabase database(512);
  const cv::Mat wrong = cv::Mat::ones(1, 16, CV_32F);
  EXPECT_THROW(database.add(wrong), std::invalid_argument);
}

TEST(FaissDatabase, AddsAndSearchesCpuFlatDescriptors) {
  FaissDatabase database(4);
  const cv::Mat descriptor =
      (cv::Mat_<float>(1, 4) << 1.0f, 0.0f, 0.0f, 0.0f);
  EXPECT_EQ(database.add(descriptor), 0);
  EXPECT_EQ(database.nTotal(), 1);

  FaissDatabase::QueryResults ids(1, -1);
  FaissDatabase::QueryDistances similarities(1, 0.0f);
  database.search(descriptor, 1, ids, similarities);
  EXPECT_EQ(ids.at(0), 0);
  EXPECT_FLOAT_EQ(similarities.at(0), 1.0f);
}

TEST(FaissDatabase, RestrictsCpuFlatSearchToExclusiveMaximumId) {
  FaissDatabase database(2);
  database.add((cv::Mat_<float>(1, 2) << 1.0f, 0.0f));
  database.add((cv::Mat_<float>(1, 2) << 0.8f, 0.2f));
  database.add((cv::Mat_<float>(1, 2) << 2.0f, 0.0f));

  const cv::Mat query = (cv::Mat_<float>(1, 2) << 1.0f, 0.0f);
  FaissDatabase::QueryResults ids(1, -1);
  FaissDatabase::QueryDistances similarities(1, 0.0f);
  database.search(query, 1, ids, similarities, 2);

  EXPECT_EQ(ids.at(0), 0);
  EXPECT_FLOAT_EQ(similarities.at(0), 1.0f);
}

}  // namespace
}  // namespace xfeat
