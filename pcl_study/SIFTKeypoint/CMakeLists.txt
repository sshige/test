cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(sift_keypoint)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (sift_keypoint sift_keypoint.cpp)
target_link_libraries (sift_keypoint ${PCL_LIBRARIES})
