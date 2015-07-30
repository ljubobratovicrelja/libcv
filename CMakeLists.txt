
project(libcv)
cmake_minimum_required(VERSION 2.8)

file(GLOB SRC_FILES "./src/*.cpp")
file(GLOB INC_FILES "./include/*.hpp")

set(CMAKE_CXX_FLAGS "-std=c++11 -w -g -O0 -fopenmp")

if (WIN32)
	# to be changed so it can compile on msc
	message(FATAL_ERROR "Windows not supported so far")
endif(NOT WIN32)

find_package(Qt4 REQUIRED)

include_directories("/usr/include/qt4")
link_directories("/usr/lib" "/usr/local/lib")

# Project variables
set(OUT_LIB_NAME "cv")

# Configuration variables
set(COMPILE_GUI TRUE CACHE BOOL "Compile GUI module of the libcv.")
set(DOUBLE_REAL TRUE CACHE BOOL "Should the real type be double(true) or float(false).")


add_library(${OUT_LIB_NAME} SHARED ${SRC_FILES})
target_link_libraries("cv" "png")
target_link_libraries("cv" Qt4::QtGui Qt4::QtCore)


add_executable("test" "./test/test.cpp")
target_link_libraries("test" "cv")

# setup install
install(TARGETS ${OUT_LIB_NAME} 
                RUNTIME DESTINATION bin
                LIBRARY DESTINATION lib
                ARCHIVE DESTINATION lib/static)
                
#install headers
install(FILES ${INC_FILES} DESTINATION "include/cv")
