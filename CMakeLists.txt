cmake_minimum_required(VERSION 3.10)
project(XFeatONNXSample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set ONNX Runtime paths
set(ONNXRUNTIME_ROOTDIR ${CMAKE_SOURCE_DIR}/../onnxruntime-osx-arm64-1.22.0)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOTDIR}/include)
set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib)

include_directories(${ONNXRUNTIME_INCLUDE_DIR})

add_executable(XFeatONNXSample main.cpp)

target_link_libraries(XFeatONNXSample 
    PRIVATE
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_LIBRARY}
)

# Optional: Add compiler flags for optimization or debugging
# target_compile_options(XFeatONNXSample PRIVATE -O2) # Example for release
# target_compile_options(XFeatONNXSample PRIVATE -g)   # Example for debug