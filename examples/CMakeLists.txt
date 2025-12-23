# Use newer FindBoost if available
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(xfeat_example main.cpp)
target_link_libraries(xfeat_example 
    PRIVATE
    ${PROJECT_NAME}
    Boost::program_options
)

install(TARGETS xfeat_example DESTINATION bin)

add_executable(batch_match_example batch_match_example.cpp)
target_link_libraries(batch_match_example 
    PRIVATE
    ${PROJECT_NAME}
    Boost::program_options
)

install(TARGETS batch_match_example DESTINATION bin)


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
)