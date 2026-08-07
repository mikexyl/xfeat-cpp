cmake_minimum_required(VERSION 3.22)
project(xfeat-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include the cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/lib64/libcudart.so")
    # Conda/Pixi environments may contain an older libcudart that CMake finds
    # before the toolkit selected by CUDA_HOME. Pin CUDA::cudart to the same
    # toolkit as nvcc so native TensorRT and CUDA calls share one runtime.
    set(CUDA_CUDART "$ENV{CUDA_HOME}/lib64/libcudart.so" CACHE FILEPATH "CUDA runtime library" FORCE)
    set(CUDA_cudart_LIBRARY "$ENV{CUDA_HOME}/lib64/libcudart.so" CACHE FILEPATH "CUDA runtime library" FORCE)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(TBB REQUIRED)
find_package(faiss REQUIRED)
find_package(OpenMP REQUIRED)

# NVIDIA VPI is optional. Its standard Debian installation exports the vpi
# target below /opt/nvidia/vpi4.
find_package(vpi 4.1 QUIET CONFIG HINTS /opt/nvidia/vpi4)
if(vpi_FOUND)
    message(STATUS "Found NVIDIA VPI ${vpi_VERSION} - enabling CUDA stereo")
    set(VPI_AVAILABLE TRUE)
    set(VPI_STEREO_SOURCES src/stereo_depth/stereo_depth_vpi.cpp)
else()
    message(STATUS "NVIDIA VPI not found - VPI CUDA stereo will not be available")
    set(VPI_AVAILABLE FALSE)
    set(VPI_STEREO_SOURCES)
endif()

# Try to find TensorRT (optional for TensorRT-backed depth modules)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/local/lib /usr/lib
          /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
          /usr/local/TensorRT/lib /usr/local/tensorrt/lib
          /opt/TensorRT/lib /opt/tensorrt/lib
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include /usr/local/include
          /usr/include/aarch64-linux-gnu
          /usr/include/x86_64-linux-gnu
          /usr/local/TensorRT/include /usr/local/tensorrt/include
          /opt/TensorRT/include /opt/tensorrt/include
)

if(TENSORRT_LIBRARY)
    message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
    # Verify we can actually use it by trying to find the target
    if(TARGET CUDA::nvinfer)
        set(TENSORRT_AVAILABLE TRUE)
    else()
        # Create imported target for TensorRT
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TENSORRT_LIBRARY}"
        )
        if(TENSORRT_INCLUDE_DIR)
            set_target_properties(TensorRT::nvinfer PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
            )
        endif()
        set(TENSORRT_AVAILABLE TRUE)
    endif()
    set(STEREO_DEPTH_SOURCES
        src/stereo_depth/stereo_depth.cpp
        src/stereo_depth/stereo_depth_lightstereo.cpp
        src/stereo_depth/stereo_depth_fast_foundation_stereo.cpp
        src/stereo_depth/stereo_depth_onnx.cpp
    )
    set(MONO_DEPTH_SOURCES
        src/mono_depth/depth_anything_v3_trt.cpp
    )
    set(TENSORRT_FEATURE_SOURCES
        src/tensorrt/trt_engine.cpp
        src/xfeat_trt.cpp
        src/lighterglue_trt.cpp
        src/place_recognition/jist_trt.cpp
        src/place_recognition/mixvpr_trt.cpp
    )
else()
    message(STATUS "TensorRT not found - TensorRT depth modules will not be available")
    set(TENSORRT_AVAILABLE FALSE)
    set(STEREO_DEPTH_SOURCES
        src/stereo_depth/stereo_depth.cpp
        src/stereo_depth/stereo_depth_onnx.cpp
    )
    set(MONO_DEPTH_SOURCES)
    set(TENSORRT_FEATURE_SOURCES)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
message(STATUS "xfeat-cpp CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

enable_language(CUDA)

if(TENSORRT_AVAILABLE)
    # Official Fast-FoundationStereo single-engine GWC plugin. The runtime
    # wrapper registers this plugin before deserializing the engine, and the
    # conversion helper loads the same shared object while building it.
    add_library(xfeat_ffs_gwc_plugin SHARED
        thirdparty/Fast-FoundationStereo/cpp/src/gwc_volume_plugin.cpp
        thirdparty/Fast-FoundationStereo/cpp/src/depth_kernels.cu
    )
    set_target_properties(xfeat_ffs_gwc_plugin PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(xfeat_ffs_gwc_plugin PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Fast-FoundationStereo/cpp/include>
        $<INSTALL_INTERFACE:include/xfeat-cpp/thirdparty/fast_foundation_stereo>
    )
    target_link_libraries(xfeat_ffs_gwc_plugin PRIVATE TensorRT::nvinfer CUDA::cudart)
endif()

# ------------- Main library target
add_library(${PROJECT_NAME} STATIC
    src/xfeat_onnx.cpp
    src/lighterglue_onnx.cpp
    src/xfeat_cv.cpp
    src/place_recognition/place_recognizer.cpp
    src/place_recognition/jist_onnx.cpp
    src/place_recognition/patchnetvlad_onnx.cpp
    src/place_recognition/patchnetvlad_matcher.cpp
    src/place_recognition/mixvpr_onnx.cpp
    src/segmentation/skyseg_trt.cpp
    ${STEREO_DEPTH_SOURCES}
    ${VPI_STEREO_SOURCES}
    ${MONO_DEPTH_SOURCES}
    ${TENSORRT_FEATURE_SOURCES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set up link libraries based on TensorRT availability
set(XFEAT_LINK_LIBS
    ${OpenCV_LIBS}
    ${onnxruntime_LIBRARY}
    TBB::tbb
    TBB::tbbmalloc
    faiss
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${XFEAT_LINK_LIBS}
)

# Link TensorRT privately if available (not part of public interface)
if(TENSORRT_AVAILABLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::nvinfer xfeat_ffs_gwc_plugin)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_TENSORRT)
endif()

if(VPI_AVAILABLE)
    target_link_libraries(${PROJECT_NAME} PUBLIC vpi)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_VPI)
endif()

add_subdirectory(include/xfeat-cpp/nms/anms)

option(CUBLAS_DIR "Path to CUBLAS installation")
option(XFEAT_BUILD_SCRIPTS "Build xfeat-cpp helper scripts" ON)
option(XFEAT_BUILD_EXAMPLES "Build xfeat-cpp examples" ON)
message(STATUS "CUBLAS_DIR: ${CUBLAS_DIR}")

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${onnxruntime_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${onnxruntime_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
)
if(CUBLAS_DIR AND NOT CUBLAS_DIR STREQUAL "OFF")
    target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CUBLAS_DIR}/include>)
endif()

if(XFEAT_BUILD_SCRIPTS)
    add_subdirectory(scripts)
endif()
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(thirdparty/lightstereo/deploy/cpp)

# Keep dependency samples aligned with the workspace-level examples option.
set(ENABLE_SAMPLES ${XFEAT_BUILD_EXAMPLES} CACHE BOOL "Build samples" FORCE)

if(XFEAT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(CMakePackageConfigHelpers)

set(XFEAT_INSTALL_TARGETS ${PROJECT_NAME})
if(TARGET xfeat_ffs_gwc_plugin)
    list(APPEND XFEAT_INSTALL_TARGETS xfeat_ffs_gwc_plugin)
endif()

install(TARGETS ${XFEAT_INSTALL_TARGETS}
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
if(TARGET xfeat_ffs_gwc_plugin)
    install(FILES thirdparty/Fast-FoundationStereo/cpp/include/ffs_gwc_plugin.hpp
        DESTINATION include/xfeat-cpp/thirdparty/fast_foundation_stereo)
endif()

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
