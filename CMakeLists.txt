cmake_minimum_required(VERSION 3.10)
project(xfeat-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include the cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(TBB REQUIRED)
find_package(faiss REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)             # adjust to your GPU

enable_language(CUDA)
# ------------- target
add_library(gpu_matcher
    src/gpu_matcher.cu
)

target_compile_features(gpu_matcher PUBLIC cxx_std_17)
target_link_libraries(gpu_matcher
    PRIVATE
        ${OpenCV_LIBS}
        CUDA::cublas            # cuBLAS
        CUDA::cudart            # CUDA runtime
)
target_include_directories(gpu_matcher
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        include
)

set_target_properties(gpu_matcher PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(${PROJECT_NAME} STATIC
    src/xfeat_onnx.cpp
    src/lighterglue_onnx.cpp
    src/xfeat_cv.cpp
    src/xfeat_netvlad_onnx.cpp
    src/netvlad_onnx.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${OpenCV_LIBS}
    ${onnxruntime_LIBRARY}
    TBB::tbb
    TBB::tbbmalloc
    faiss
    gpu_matcher
)

option(CUBLAS_DIR "Path to CUBLAS installation")
message(STATUS "CUBLAS_DIR: ${CUBLAS_DIR}")

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${onnxruntime_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUBLAS_DIR}/include
)

add_subdirectory(scripts)
add_subdirectory(tests)
add_subdirectory(examples)

include(CMakePackageConfigHelpers)

install(TARGETS ${PROJECT_NAME}
    EXPORT xfeat-cppTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(EXPORT xfeat-cppTargets
    FILE xfeat-cppTargets.cmake
    NAMESPACE xfeat-cpp::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/xfeat-cppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/xfeat-cppConfig.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/xfeat-cppConfig.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# install Findonnxruntime.cmake
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Findonnxruntime.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)

# Install and export gpu_matcher target
install(TARGETS gpu_matcher
    EXPORT xfeat-cppTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)