cmake_minimum_required(VERSION 2.8.3)
project(opt_benchmark)

add_compile_options(-std=c++11)

find_package(catkin COMPONENTS cmake_modules rostest)
find_package(Eigen REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

catkin_package()

# for debug
# message("[PKG_CONFIG_EXECUTABLE] ${PKG_CONFIG_EXECUTABLE}")
# message("[Eigen_FOUND] ${Eigen_FOUND}")
# message("[Eigen_LIBRARIES] ${Eigen_LIBRARIES}")
# message("[Eigen_INCLUDE_DIRS] ${Eigen_INCLUDE_DIRS}")
# message("[Boost_LIBRARIES] ${Boost_LIBRARIES}")
# message("[Boost_INCLUDE_DIRS] ${Boost_INCLUDE_DIRS}")
# message("[catkin_LIBRARIES] ${catkin_LIBRARIES}")
# message("[catkin_INCLUDE_DIRS] ${catkin_INCLUDE_DIRS}")

include_directories(${Eigen_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

add_executable(eigen_lm src/eigen_lm.cpp)
target_link_libraries(eigen_lm ${Boost_LIBRARIES} ${catkin_LIBRARIES})
