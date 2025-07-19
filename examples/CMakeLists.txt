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

