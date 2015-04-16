cmake_minimum_required(VERSION 2.8)
project(gaga)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Wextra -pedantic -Wno-dollar-in-identifier-extension" ${MPI_COMPILE_FLAGS})
file(GLOB CORESRC 
	"src/*.h"
	"src/*.hpp"
	"src/*.cpp"
	"src/jsonxx/jsonxx.h"
	"src/jsonxx/jsonxx.cc"
	)

find_package(MPI REQUIRED)
add_executable(gaga ${CORESRC})
target_link_libraries(gaga ${MPI_LIBRARIES})
set_target_properties(gaga
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
	)
