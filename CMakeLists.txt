
project(libcv)
cmake_minimum_required(VERSION 2.8)

file(GLOB SRC_FILES "./src/*.c*")
file(GLOB INC_FILES "./include/*.h*")

set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -w -DCV_REAL_TYPE_DOUBLE")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if (CMAKE_BUILD_TYPE MATCHES Debug)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
else(CMAKE_BUILD_TYPE MATCHES Debug)
	message(warning "Invalid Build Type - only Release and Debug supported, and give value is: ${CMAKE_BUILD_TYPE}")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

message(${CMAKE_CXX_FLAGS})

if (WIN32)
	# to be changed so it can compile on msc
	message(FATAL_ERROR "Windows not supported so far")
endif(WIN32)

find_package(Qt4 REQUIRED)

include_directories("/usr/include/qt4")
link_directories("/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu")

# Project variables
set(OUT_LIB_NAME "cv")

# Configuration variables
set(WITH_GUI TRUE CACHE BOOL "Compile GUI module of the libcv.")
set(DOUBLE_REAL TRUE CACHE BOOL "Should the real type be double(true) or float(false).")
set(COMPILE_EXAMPLES TRUE CACHE BOOL "Should the examples be compiled.")

if (NOT ${WITH_GUI})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_IGNORE_GUI")
endif(${WITH_GUI})

add_library(${OUT_LIB_NAME} SHARED ${SRC_FILES})
target_link_libraries(${OUT_LIB_NAME} "png" "lapacke" "lapack" "blas" "jpeg")
target_link_libraries(${OUT_LIB_NAME} Qt4::QtGui Qt4::QtCore)

if(${COMPILE_EXAMPLES})
	add_executable(cv_image_example "./examples/image.cpp")
	add_executable(cv_kdtree_example "./examples/kdtree.cpp")
	add_executable(cv_linalg_example "./examples/linalg.cpp")
	add_executable(cv_matrix_example "./examples/matrix.cpp")
	add_executable(cv_vector_example "./examples/vector.cpp")

	target_link_libraries(cv_image_example ${OUT_LIB_NAME})
	target_link_libraries(cv_kdtree_example ${OUT_LIB_NAME})
	target_link_libraries(cv_linalg_example ${OUT_LIB_NAME})
	target_link_libraries(cv_matrix_example ${OUT_LIB_NAME})
	target_link_libraries(cv_vector_example ${OUT_LIB_NAME})

	set(EXAMPLE_FILES cv_image_example cv_kdtree_example cv_linalg_example cv_matrix_example cv_vector_example)
	install(FILES ${EXAMPLE_FILES} DESTINATION "examples/")
endif(${COMPILE_EXAMPLES})

# setup install
install(TARGETS ${OUT_LIB_NAME} 
                RUNTIME DESTINATION bin
                LIBRARY DESTINATION lib
                ARCHIVE DESTINATION lib/static)
                
#install headers
install(FILES ${INC_FILES} DESTINATION "include/cv")
