cmake_minimum_required(VERSION 2.8.3)
project(opt_benchmark)

add_compile_options(-std=c++11)

find_package(catkin COMPONENTS cmake_modules rostest)
find_package(Eigen REQUIRED)

catkin_package()

# for debug
# message("[PKG_CONFIG_EXECUTABLE] ${PKG_CONFIG_EXECUTABLE}")
# message("[Eigen_FOUND] ${Eigen_FOUND}")
# message("[Eigen_LIBRARIES] ${Eigen_LIBRARIES}")
# message("[Eigen_INCLUDE_DIRS] ${Eigen_INCLUDE_DIRS}")

include_directories(${Eigen_INCLUDE_DIRS})

add_executable(eigen_lm src/eigen_lm.cpp)
