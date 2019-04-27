cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

add_compile_options("-Wall" "-Wextra" "-pedantic")

find_package(ZMQ REQUIRED)



add_executable(server server.cpp)
target_include_directories(server PUBLIC ${ZeroMQ_INCLUDE_DIRS})
target_include_directories(server PUBLIC "../../../../")
target_link_libraries(server PUBLIC ${ZeroMQ_LIBRARIES})


