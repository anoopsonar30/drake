cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_executable(drake_test drake_example.cc)
target_link_libraries(drake_test drake::drake gflags)

add_executable(det_opt_workstation workstation_deterministic_trajopt.cc)
target_link_libraries(det_opt_workstation drake::drake gflags)

add_executable(cc_opt_workstation workstation_obstacle_uncertainty_trajopt.cc)
target_link_libraries(cc_opt_workstation
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(hw_demo hw_demo.cc)
target_link_libraries(hw_demo
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(cc_opt_ufo ufo_obstacle_uncertainty_trajopt.cc)
target_link_libraries(cc_opt_ufo
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(ufo_benchmark_conservatism ufo_benchmarking_conservatism.cc)
target_link_libraries(ufo_benchmark_conservatism
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(ufo_benchmark_cost ufo_benchmarking_cost.cc)
target_link_libraries(ufo_benchmark_cost
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(ufo_benchmark_obstacles ufo_benchmarking_obstacles.cc)
target_link_libraries(ufo_benchmark_obstacles
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(ufo_benchmark_generic ufo_benchmarking_generic.cc)
target_link_libraries(ufo_benchmark_generic
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(ufo_benchmark_tunnel ufo_benchmarking_tunnel.cc)
target_link_libraries(ufo_benchmark_tunnel
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(cc_opt_car car_obstacle_uncertainty_trajopt.cc)
target_link_libraries(cc_opt_car
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(car_fixed car_fixed.cc)
target_link_libraries(car_fixed
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(joint_cc_opt_car car_all_uncertainty_trajopt.cc)
target_link_libraries(joint_cc_opt_car
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(car_chekov car_all_uncertainty_chekov.cc)
target_link_libraries(car_chekov
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(car_obs_ira car_obstacle_uncertainty_ira.cc)
target_link_libraries(car_obs_ira
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(joint_cc_opt_mobile_arm mobile_arm_all_uncertainty_trajopt.cc)
target_link_libraries(joint_cc_opt_mobile_arm
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(mobile_arm_fixed mobile_arm_fixed.cc)
target_link_libraries(mobile_arm_fixed
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(mobile_arm_bench mobile_arm_all_uncertainty_trajopt_bench.cc)
target_link_libraries(mobile_arm_bench
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(joint_cc_opt_mobile_arm_chekov mobile_arm_all_uncertainty_chekov.cc)
target_link_libraries(joint_cc_opt_mobile_arm_chekov
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(simple_arm simple_arm.cc)
target_link_libraries(simple_arm
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(moving_obstacle arm_moving_obstacle.cc)
target_link_libraries(moving_obstacle
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(two_arm_moving_obstacle two_arms_moving_obstacle.cc)
target_link_libraries(two_arm_moving_obstacle
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(three_arm_moving_obstacle three_arms_moving_obstacle.cc)
target_link_libraries(three_arm_moving_obstacle
    ccopt
    proximity_alert
    drake::drake
    gflags)

add_executable(four_arm_moving_obstacle four_arms_moving_obstacle.cc)
target_link_libraries(four_arm_moving_obstacle
    ccopt
    proximity_alert
    drake::drake
    gflags)