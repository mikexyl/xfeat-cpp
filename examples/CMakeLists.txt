# Use newer FindBoost if available
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(PCL REQUIRED COMPONENTS common io visualization filters)

add_executable(xfeat_example main.cpp)
target_link_libraries(xfeat_example 
    PRIVATE
    ${PROJECT_NAME}
    Boost::program_options
)

install(TARGETS xfeat_example DESTINATION bin)


add_executable(feature_nms_example feature_nms.cpp)
target_link_libraries(feature_nms_example
    PRIVATE
    ${PROJECT_NAME}
    Boost::program_options
)

install(TARGETS feature_nms_example DESTINATION bin)

add_executable(stereo_depth_example stereo_depth_example.cpp)
target_link_libraries(stereo_depth_example
    PRIVATE
    ${PROJECT_NAME}
    Boost::program_options
    ${PCL_LIBRARIES}
)
target_include_directories(stereo_depth_example PRIVATE ${PCL_INCLUDE_DIRS})

add_executable(onnx_stereo_depth_example onnx_stereo_depth_example.cpp)
target_link_libraries(onnx_stereo_depth_example
    PRIVATE
    ${PROJECT_NAME}
    ${PCL_LIBRARIES}
)
target_include_directories(onnx_stereo_depth_example PRIVATE ${PCL_INCLUDE_DIRS})

install(TARGETS onnx_stereo_depth_example DESTINATION bin)

add_executable(patchnetvlad_example patchnetvlad_example.cpp)
target_link_libraries(patchnetvlad_example
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS patchnetvlad_example DESTINATION bin)

# Example executable
add_executable(skyseg_example skyseg_example.cpp)
target_link_libraries(skyseg_example xfeat-cpp ${OpenCV_LIBS} nvinfer cudart)