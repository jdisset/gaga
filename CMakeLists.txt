cmake_minimum_required(VERSION 2.8)
project(GAGA)
set(CMAKE_CXX_COMPILER  "g++-5")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Wextra -pedantic -Wno-dollar-in-identifier-extension -fopenmp" ${MPI_COMPILE_FLAGS})

add_subdirectory(examples)

