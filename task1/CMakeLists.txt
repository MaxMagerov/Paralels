cmake_minimum_required(VERSION 3.27)
project(Lesson1)

add_executable(double main.cpp)

add_executable(float main.cpp)
target_compile_definitions(float PUBLIC FLOAT)