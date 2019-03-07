cmake_minimum_required(VERSION 2.8.3)
project(opt_benchmark)

add_compile_options(-std=c++11)

find_package(catkin COMPONENTS cmake_modules rostest)
find_package(Eigen3 3.3 REQUIRED CONFIG PATHS /home/leus/install/share/eigen3/cmake NO_DEFAULT_PATH)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/OpenCVFindOpenBLAS.cmake)

catkin_package()

# for debug
# message("[EIGEN3_FOUND] ${EIGEN3_FOUND}")
# message("[EIGEN3_USE_FILE] ${EIGEN3_USE_FILE}")
# message("[EIGEN3_LIBRARIES] ${EIGEN3_LIBRARIES}")
# message("[EIGEN3_INCLUDE_DIRS] ${EIGEN3_INCLUDE_DIRS}")
# message("[Boost_LIBRARIES] ${Boost_LIBRARIES}")
# message("[Boost_INCLUDE_DIRS] ${Boost_INCLUDE_DIRS}")
# message("[OpenBLAS_LIB] ${OpenBLAS_LIB}")
# message("[OpenBLAS_INCLUDE_DIR] ${OpenBLAS_INCLUDE_DIR}")
# message("[catkin_LIBRARIES] ${catkin_LIBRARIES}")
# message("[catkin_INCLUDE_DIRS] ${catkin_INCLUDE_DIRS}")

include_directories(${EIGEN3_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

add_executable(eigen_lm src/eigen_lm.cpp)
target_link_libraries(eigen_lm ${Boost_LIBRARIES} ${OpenBLAS_LIB} ${catkin_LIBRARIES})
# target_compile_definitions(eigen_lm PUBLIC -DEIGEN_USE_BLAS)
