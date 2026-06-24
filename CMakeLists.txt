cmake_minimum_required(VERSION 3.16)
project(gandalf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Eigen3 REQUIRED)

find_package(Qt6 COMPONENTS Widgets QUIET)

if(Qt6_FOUND)
    message(STATUS "Qt6 found — building with GUI support")
    add_executable(gandalf main.cpp gui.cpp)
    target_link_libraries(gandalf Eigen3::Eigen Qt6::Widgets)
    target_compile_definitions(gandalf PRIVATE GANDALF_GUI=1)
else()
    message(STATUS "Qt6 not found — building CLI-only version")
    add_executable(gandalf main.cpp)
    target_link_libraries(gandalf Eigen3::Eigen)
endif()
