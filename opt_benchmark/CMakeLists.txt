cmake_minimum_required(VERSION 2.8.3)
project(opt_benchmark)

add_compile_options(-std=c++11)

find_package(catkin COMPONENTS cmake_modules rostest)

set(EIGEN3.3_FROM_SOURCE false CACHE BOOL "set true to use eigen installed in ~/install directory.")
set(USE_OPENBLAS false CACHE BOOL "set true to use OpenBLAS. this is enabled only when EIGEN3.3_FROM_SOURCE is true.")
set(USE_OPENMP true CACHE BOOL "set true to use OpenMP.")

if(EIGEN3.3_FROM_SOURCE)
  # Assume that Eigen is installed from source with the following cmake command
  # cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/install/
  find_package(Eigen3 3.3 REQUIRED CONFIG PATHS $ENV{HOME}/install/share/eigen3/cmake NO_DEFAULT_PATH)
  message("Eigen built from source is used: ${EIGEN3_USE_FILE}")
  set(Eigen_LIBRARIES ${EIGEN3_LIBRARIES})
  set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
else()
  find_package(Eigen REQUIRED)
  if(USE_OPENBLAS)
    message(WARNING "USE_OPENBLAS is force disabled when EIGEN3.3_FROM_SOURCE is false because apt version cause error. ref. https://forum.kde.org/viewtopic.php?f=74&t=123117#p359150")
  endif()
  set(USE_OPENBLAS false)
endif()
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
if(USE_OPENBLAS)
  include(${PROJECT_SOURCE_DIR}/cmake/OpenCVFindOpenBLAS.cmake)
endif()
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

catkin_package()

# for debug
# message("[Eigen_LIBRARIES] ${Eigen_LIBRARIES}")
# message("[Eigen_INCLUDE_DIRS] ${Eigen_INCLUDE_DIRS}")
# message("[Boost_LIBRARIES] ${Boost_LIBRARIES}")
# message("[Boost_INCLUDE_DIRS] ${Boost_INCLUDE_DIRS}")
# message("[OpenBLAS_LIB] ${OpenBLAS_LIB}")
# message("[OpenBLAS_INCLUDE_DIR] ${OpenBLAS_INCLUDE_DIR}")
# message("[OpenMP_C_FLAGS] ${OpenMP_C_FLAGS}")
# message("[OpenMP_CXX_FLAGS] ${OpenMP_CXX_FLAGS}")
# message("[OpenMP_EXE_LINKER_FLAGS] ${OpenMP_EXE_LINKER_FLAGS}")
# message("[catkin_LIBRARIES] ${catkin_LIBRARIES}")
# message("[catkin_INCLUDE_DIRS] ${catkin_INCLUDE_DIRS}")

include_directories(${Eigen_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

add_executable(eigen_lm src/eigen_lm.cpp)
target_link_libraries(eigen_lm ${Boost_LIBRARIES} ${OpenBLAS_LIB} ${catkin_LIBRARIES})
if(USE_OPENBLAS)
  target_compile_definitions(eigen_lm PUBLIC -DEIGEN_USE_BLAS)
endif()
