file(GLOB SRC 
	"*.h"
	"*.hpp"
	"*.cpp"
	)
add_executable(biggestInt ${SRC})
target_link_libraries(biggestInt gaga)


