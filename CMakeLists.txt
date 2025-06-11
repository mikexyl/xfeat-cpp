cmake_minimum_required(VERSION 3.10)
project(xfeat-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include the cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(onnxruntime REQUIRED)

include_directories(${onnxruntime_INCLUDE_DIR})

add_library(${PROJECT_NAME} src/xfeat_onnx.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS} 
    ${onnxruntime_LIBRARY}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_subdirectory(examples)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)