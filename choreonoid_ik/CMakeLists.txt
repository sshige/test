cmake_minimum_required(VERSION 2.8.3)
project(choreonoid_ik)

add_compile_options(-std=c++11)

set(USE_OPENMP true CACHE BOOL "set true to use OpenMP.")

find_package(catkin COMPONENTS cmake_modules rostest)
find_package(Eigen REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(FindPkgConfig)
pkg_check_modules(CnoidBody choreonoid-body REQUIRED)

catkin_package()

# for debug
# message("[Eigen_LIBRARIES] ${Eigen_LIBRARIES}")
# message("[Eigen_INCLUDE_DIRS] ${Eigen_INCLUDE_DIRS}")
# message("[Boost_LIBRARIES] ${Boost_LIBRARIES}")
# message("[Boost_INCLUDE_DIRS] ${Boost_INCLUDE_DIRS}")
# message("[OpenMP_C_FLAGS] ${OpenMP_C_FLAGS}")
# message("[OpenMP_CXX_FLAGS] ${OpenMP_CXX_FLAGS}")
# message("[OpenMP_EXE_LINKER_FLAGS] ${OpenMP_EXE_LINKER_FLAGS}")
# message("[CnoidBody_LIBRARIES] ${CnoidBody_LIBRARIES}")
# message("[CnoidBody_LIBRARY_DIRS] ${CnoidBody_LIBRARY_DIRS}")
# message("[CnoidBody_INCLUDE_DIRS] ${CnoidBody_INCLUDE_DIRS}")
# message("[catkin_LIBRARIES] ${catkin_LIBRARIES}")
# message("[catkin_INCLUDE_DIRS] ${catkin_INCLUDE_DIRS}")

include_directories(${Eigen_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CnoidBody_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${CnoidBody_LIBRARY_DIRS})

add_executable(sample_ik src/sample_ik.cpp)
target_link_libraries(sample_ik ${Boost_LIBRARIES} ${CnoidBody_LIBRARIES} ${catkin_LIBRARIES})
