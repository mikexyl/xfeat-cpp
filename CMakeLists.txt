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
find_package(OpenMP REQUIRED)

# Try to find TensorRT (optional for LightStereo)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/local/lib /usr/lib
          /usr/local/TensorRT/lib /usr/local/tensorrt/lib
          /opt/TensorRT/lib /opt/tensorrt/lib
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
        set(TENSORRT_AVAILABLE TRUE)
    endif()
    set(STEREO_DEPTH_SOURCES
        src/stereo_depth.cpp
        src/stereo_depth_libsgm.cpp
        src/stereo_depth_lightstereo.cpp
        src/stereo_depth_onnx.cpp
    )
else()
    message(STATUS "TensorRT not found - LightStereo depth will not be available")
    set(TENSORRT_AVAILABLE FALSE)
    set(STEREO_DEPTH_SOURCES
        src/stereo_depth.cpp
        src/stereo_depth_libsgm.cpp
        src/stereo_depth_onnx.cpp
    )
endif()

set(CMAKE_CUDA_ARCHITECTURES 89)

enable_language(CUDA)

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
)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set up link libraries based on TensorRT availability
set(XFEAT_LINK_LIBS
    ${OpenCV_LIBS}
    ${onnxruntime_LIBRARY}
    TBB::tbb
    TBB::tbbmalloc
    faiss
    sgm
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

# Add BUILD_OPENCV_WRAPPER definition for stereo_depth_libsgm.cpp
target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_OPENCV_WRAPPER)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${XFEAT_LINK_LIBS}
)

# Link TensorRT privately if available (not part of public interface)
if(TENSORRT_AVAILABLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::nvinfer)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_TENSORRT)
endif()

add_subdirectory(include/xfeat-cpp/nms/anms)

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

add_subdirectory(thirdparty/lightstereo/deploy/cpp)

# set ENABLE_SAMPLES for libsgm
set(ENABLE_SAMPLES ON CACHE BOOL "Build samples" FORCE)

# Enable OpenCV wrapper for libsgm BEFORE adding the subdirectory
set(BUILD_OPENCV_WRAPPER ON CACHE BOOL "Make library compatible with cv::Mat and cv::cuda::GpuMat of OpenCV" FORCE)

# Patch libsgm CMakeLists.txt for proper export support
set(LIBSGM_CMAKE_FILE "${CMAKE_SOURCE_DIR}/thirdparty/libsgm/src/CMakeLists.txt")
file(READ "${LIBSGM_CMAKE_FILE}" LIBSGM_CMAKE_CONTENT)

# Check if patch is needed
string(FIND "${LIBSGM_CMAKE_CONTENT}" "INTERFACE_INCLUDE_DIRECTORIES \${LIBSGM_INCLUDE_DIR}" NEEDS_PATCH)
if(NOT NEEDS_PATCH EQUAL -1)
    message(STATUS "Patching libsgm for proper export support")
    
    # Replace the problematic include directories setup
    string(REPLACE 
        "target_include_directories(\${PROJECT_NAME} PRIVATE \${LIBSGM_INCLUDE_DIR} $<$<BOOL:\${BUILD_OPENCV_WRAPPER}>:\${OpenCV_INCLUDE_DIRS}>)"
        "target_include_directories(\${PROJECT_NAME}\n    PUBLIC\n        $<BUILD_INTERFACE:\${LIBSGM_INCLUDE_DIR}>\n        $<INSTALL_INTERFACE:include>\n    PRIVATE\n        $<$<BOOL:\${BUILD_OPENCV_WRAPPER}>:\${OpenCV_INCLUDE_DIRS}>)"
        LIBSGM_CMAKE_CONTENT "${LIBSGM_CMAKE_CONTENT}")
    
    # Remove the problematic set_target_properties line
    string(REPLACE 
        "set_target_properties(\${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \${LIBSGM_INCLUDE_DIR})\n\n"
        ""
        LIBSGM_CMAKE_CONTENT "${LIBSGM_CMAKE_CONTENT}")
    
    file(WRITE "${LIBSGM_CMAKE_FILE}" "${LIBSGM_CMAKE_CONTENT}")
endif()

add_subdirectory(thirdparty/libsgm)

# Set POSITION_INDEPENDENT_CODE for libsgm after it's added
set_target_properties(sgm PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(examples)

include(CMakePackageConfigHelpers)

install(TARGETS ${PROJECT_NAME} sgm
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
install(DIRECTORY thirdparty/libsgm/include/ DESTINATION include)

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