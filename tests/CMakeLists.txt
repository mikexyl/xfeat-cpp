# CMake configuration for xfeat-cpp tests

find_package(GTest REQUIRED)

add_executable(test_xfeat_cv test_xfeat_cv.cpp)
target_link_libraries(test_xfeat_cv PRIVATE ${PROJECT_NAME} GTest::GTest GTest::Main)

enable_testing()
add_test(NAME test_xfeat_cv COMMAND test_xfeat_cv)
