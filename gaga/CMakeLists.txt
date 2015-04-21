file(GLOB CORESRC 
	"*.h"
	"*.hpp"
	"*.cpp"
	"jsonxx/*.h"
	"jsonxx/*.cc"
	)

find_package(MPI REQUIRED)
add_library(gaga ${CORESRC})
target_link_libraries(gaga ${MPI_LIBRARIES})
