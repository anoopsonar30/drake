cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_library(ccopt STATIC
    chance_constraints.cpp
    chance_constraints.h
    joint_chance_constraints.cpp
    joint_chance_constraints.h
    bullet_world_manager.cpp
    bullet_world_manager.h
    utils.h
    utils.cpp
)

target_include_directories(ccopt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BULLET_INCLUDE_DIRS}
)
target_link_libraries(ccopt PUBLIC
    proximity_alert
    drake::drake
    Eigen3::Eigen
    ${BULLET_LIBRARIES}
)

add_executable(cc_opt_runner cc_opt_runner.cpp)
target_link_libraries(cc_opt_runner
    ccopt
    drake::drake
    gflags
    nlohmann_json::nlohmann_json)
