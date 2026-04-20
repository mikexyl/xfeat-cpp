# Use newer FindBoost if available
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(PCL REQUIRED COMPONENTS common io)

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

add_executable(stereo_example stereo_example.cpp)
target_link_libraries(stereo_example
    PRIVATE
    ${PROJECT_NAME}
    Boost::program_options
    ${PCL_LIBRARIES}
)
target_include_directories(stereo_example PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(stereo_example PRIVATE ${PCL_DEFINITIONS})
install(TARGETS stereo_example DESTINATION bin)

add_executable(patchnetvlad_example patchnetvlad_example.cpp)
target_link_libraries(patchnetvlad_example
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS patchnetvlad_example DESTINATION bin)

add_executable(mixvpr_example mixvpr_example.cpp)
target_link_libraries(mixvpr_example
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS mixvpr_example DESTINATION bin)

add_executable(vpr_example vpr_example.cpp)
target_link_libraries(vpr_example
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS vpr_example DESTINATION bin)

# Example executable
add_executable(skyseg_example skyseg_example.cpp)
target_link_libraries(skyseg_example xfeat-cpp ${OpenCV_LIBS} nvinfer cudart)
