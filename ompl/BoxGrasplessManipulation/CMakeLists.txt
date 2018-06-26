cmake_minimum_required(VERSION 2.8)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(ColourBold  "${Esc}[1m")
set(Red         "${Esc}[31m")
message(WARNING "${Red}${ColourBold} Set the following environment variable.${ColourReset}")
message(WARNING "${Red}${ColourBold} export CXX=g++-5")
message(WARNING "${Red}${ColourBold} DO NOT source setup.bash under ros directory to avoid that old ompl library is used. ${ColourReset}")

# compiler for ompl
include(CMakeModules/CompilerSettings.cmake)

# eigen for kdl
find_package(Eigen 3 QUIET)
if(NOT Eigen_FOUND)
  include(CMakeModules/FindEigen3.cmake)
  set(Eigen_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
endif()
include_directories(${Eigen_INCLUDE_DIR})

link_directories("/usr/local/lib/x86_64-linux-gnu") # for ompl

include_directories("/home/leus/usr/include/kdl") # for kdl
link_directories("/home/leus/usr/lib") # for kdl

add_executable(2DofRobotCircleConstraint 2DofRobotCircleConstraint.cpp)
target_link_libraries(2DofRobotCircleConstraint ompl) # for ompl
target_link_libraries(2DofRobotCircleConstraint orocos-kdl) # for kdl

# export CXX=g++-5
# cd ~/test/ompl/2DofRobotCircleConstraint/build/
# rm -rf *
# cmake ..
# make
# ./2DofRobotCircleConstraint
