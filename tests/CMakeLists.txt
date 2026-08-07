# CMake configuration for xfeat-cpp tests

find_package(GTest REQUIRED)

add_executable(test_xfeat_cv test_xfeat_cv.cpp)
target_link_libraries(test_xfeat_cv PRIVATE ${PROJECT_NAME} GTest::GTest GTest::Main)
target_compile_definitions(test_xfeat_cv PRIVATE XFEAT_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

enable_testing()
add_test(NAME test_xfeat_cv COMMAND test_xfeat_cv)

add_executable(test_faiss test_faiss.cpp)
target_link_libraries(test_faiss PRIVATE ${PROJECT_NAME} GTest::GTest GTest::Main)
add_test(NAME test_faiss COMMAND test_faiss)

add_executable(test_mono_depth test_mono_depth.cpp)
target_link_libraries(test_mono_depth PRIVATE ${PROJECT_NAME} GTest::GTest GTest::Main)
target_compile_definitions(test_mono_depth PRIVATE XFEAT_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_mono_depth COMMAND test_mono_depth)

add_executable(test_stereo_depth test_stereo_depth.cpp)
target_link_libraries(test_stereo_depth PRIVATE ${PROJECT_NAME} GTest::GTest GTest::Main)
add_test(NAME test_stereo_depth COMMAND test_stereo_depth)

add_executable(test_tensorrt_features test_tensorrt_features.cpp)
target_link_libraries(test_tensorrt_features PRIVATE ${PROJECT_NAME} GTest::GTest GTest::Main)
target_compile_definitions(test_tensorrt_features PRIVATE XFEAT_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_tensorrt_features COMMAND test_tensorrt_features)
