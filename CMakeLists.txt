cmake_minimum_required(VERSION 3.10)
project(xfeat-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include the cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(TBB REQUIRED)

add_library(${PROJECT_NAME} SHARED
    src/xfeat_onnx.cpp
    src/lighterglue_onnx.cpp
    src/xfeat_cv.cpp)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${OpenCV_LIBS}
    ${onnxruntime_LIBRARY}
    TBB::tbb
    TBB::tbbmalloc
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${onnxruntime_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(examples)
add_subdirectory(tests)

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