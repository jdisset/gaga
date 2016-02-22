file(GLOB SRC 
	"*.h"
	"*.hpp"
	"*.cpp"
	"../../gaga/gaga.hpp"
	)
#add_definitions(-DOMP)
add_definitions(-DCLUSTER)
find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_executable(biggestInt ${SRC})
target_link_libraries(biggestInt ${MPI_CXX_LIBRARIES})


