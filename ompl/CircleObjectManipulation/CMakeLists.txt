cmake_minimum_required(VERSION 2.8)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(ColourBold  "${Esc}[1m")
set(Red         "${Esc}[31m")
message(WARNING "${Red}${ColourBold} Set the following environment variable.${ColourReset}")
message(WARNING "${Red}${ColourBold} export CXX=g++-5")
message(WARNING "${Red}${ColourBold} export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu")
message(WARNING "${Red}${ColourBold} DO NOT source setup.bash under ros directory to avoid that old ompl library is used. ${ColourReset}")



set(CMAKE_MODULE_PATH
    "${CMAKE_MODULE_PATH}"
    "${CMAKE_ROOT_DIR}/cmake/Modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(GNUInstallDirs)
include(CompilerSettings)
include(OMPLVersion)
include(OMPLUtils)

macro(add_ompl_demo demo_name)
  add_executable(${ARGV})
  target_link_libraries(${demo_name}
    ompl
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
endmacro(add_ompl_demo)

add_ompl_demo(demo_CircleObjectManipulation CircleObjectManipulation.cpp)
