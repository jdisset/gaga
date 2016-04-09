cmake_minimum_required(VERSION 2.8)
project(gaga_unit CXX)
set(CMAKE_CXX_FLAGS "-O3 -g -std=c++14 -Wall -Wextra -Wundef -Wshadow -Winit-self -Wconversion -Wformat=2 -pedantic ")
file(GLOB SRC
	"*.hpp"
  "config.cpp"
	"*.cpp"
	"../*.h"
	"../*.hpp"
	"../*.cpp"
	)
add_executable(gaga_unit_test ${SRC})
