cmake_minimum_required(VERSION 3.1)
project(GAGA-Examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -Wextra -pedantic")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
