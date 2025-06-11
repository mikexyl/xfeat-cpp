add_executable(xfeat_example main.cpp)
target_link_libraries(xfeat_example 
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS xfeat_example DESTINATION bin)

