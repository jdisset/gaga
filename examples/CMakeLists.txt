cmake_minimum_required(VERSION 3.1)
project(GAGA-Examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if( NOT CMAKE_BUILD_TYPE )
	set(CMAKE_BUILD_TYPE Release) # default build = release
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -Wextra -pedantic")

# Sanitize mode
set(CMAKE_CXX_FLAGS_SAN " -O1 -g -fsanitize=address,undefined" CACHE STRING "Enable sanitization" FORCE)
mark_as_advanced( CMAKE_CXX_FLAGS_SAN)

find_package(SQLite3)
find_package(Threads)
add_executable(onemax onemax/simple_onemax.cpp)
add_executable(onenovelty onemax/simple_novelty.cpp)

#target_compile_definitions(onemax PRIVATE SQLITE_SAVE)
target_link_libraries (onemax PRIVATE ${SQLITE3_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(onemax PRIVATE ${SQLITE3_INCLUDE_DIRS})

target_compile_definitions(onenovelty PRIVATE SQLITE_SAVE)
target_link_libraries (onenovelty PRIVATE ${SQLITE3_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(onenovelty PRIVATE ${SQLITE3_INCLUDE_DIRS})
