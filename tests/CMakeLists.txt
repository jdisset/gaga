cmake_minimum_required(VERSION 3.1)
project(GAGA-tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

if( NOT CMAKE_BUILD_TYPE )
	set(CMAKE_BUILD_TYPE Release) # default build = release
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -Wextra -pedantic")

# Sanitize modes
set(CMAKE_CXX_FLAGS_ASAN " -O1 -g -fsanitize=address,undefined" CACHE STRING "Enable sanitization" FORCE)
mark_as_advanced( CMAKE_CXX_FLAGS_ASAN)

set(CMAKE_CXX_FLAGS_TSAN " -O1 -g -fsanitize=thread" CACHE STRING "Enable sanitization" FORCE)
mark_as_advanced( CMAKE_CXX_FLAGS_TSAN)


find_package(ZMQ REQUIRED)
find_package(Boost REQUIRED)
find_package(SQLite3 REQUIRED)

add_executable(test unit.cpp)
target_include_directories(test PUBLIC ${ZeroMQ_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${SQLITE3_INCLUDE_DIRS})
target_link_libraries(test PUBLIC ${SQLITE3_LIBRARIES} ${ZeroMQ_LIBRARIES})


