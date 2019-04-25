cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

add_compile_options("-Wall" "-Wextra" "-pedantic")

find_package(ZMQ REQUIRED)
find_package(Boost REQUIRED)
find_package (SQLite3 REQUIRED)


add_executable(test unit.cpp)
target_include_directories(test PUBLIC ${ZeroMQ_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${SQLITE3_INCLUDE_DIRS})
target_link_libraries(test PUBLIC ${SQLITE3_LIBRARIES} ${ZeroMQ_LIBRARIES})


