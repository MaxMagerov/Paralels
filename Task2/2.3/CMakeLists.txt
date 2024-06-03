cmake_minimum_required(VERSION 3.22.1)
project(task2.3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fopenmp -O2")

add_executable(individual_region source/individual_region.cpp)

add_executable(static source/single_region.cpp)

add_executable(dynamic source/single_region.cpp)
target_compile_definitions(dynamic PRIVATE SCHEDULE_DYNAMIC)

add_executable(guided source/single_region.cpp)
target_compile_definitions(guided PRIVATE SCHEDULE_GUIDED)

add_executable(auto source/single_region.cpp)
target_compile_definitions(auto PRIVATE SCHEDULE_AUTO)
