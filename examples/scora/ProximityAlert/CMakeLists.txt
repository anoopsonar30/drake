cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# So that we can have unit tests, we'll include these in a static library
add_library(proximity_alert INTERFACE)
target_include_directories(proximity_alert INTERFACE drake::drake ${CMAKE_CURRENT_SOURCE_DIR} ${BULLET_INCLUDE_DIRS})

add_executable(shadows shadows.cpp)
target_link_libraries(shadows PRIVATE proximity_alert drake::drake Eigen3::Eigen ${BULLET_LIBRARIES})