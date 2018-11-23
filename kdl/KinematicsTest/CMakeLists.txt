CMAKE_MINIMUM_REQUIRED(VERSION 2.6)


find_package(Eigen 3 QUIET)
if(NOT Eigen_FOUND)
  include(config/FindEigen3.cmake)
  set(Eigen_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
endif()
include_directories(${Eigen_INCLUDE_DIR})

include_directories("/home/leus/usr/include/kdl")
link_directories("/home/leus/usr/lib")

add_executable(solvertest solvertest.cpp)
target_link_libraries(solvertest orocos-kdl)

add_executable(root_ik_test root_ik_test.cpp)
target_link_libraries(root_ik_test orocos-kdl)
