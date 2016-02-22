cmake_minimum_required(VERSION 2.8)
project(GAGA)
set(CMAKE_CXX_COMPILER "g++-5")
set(CMAKE_CXX_FLAGS "-O3 -g -std=c++14 -Wall -Wextra -Wundef -Wshadow -pedantic -fopenmp")


add_subdirectory(examples)
