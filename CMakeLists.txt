
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
else(...)
	message(warning "Invalid Build Type - only Release and Debug supported, and give value is: ${CMAKE_BUILD_TYPE}")
endif(...)

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
set(COMPILE_GUI TRUE CACHE BOOL "Compile GUI module of the libcv.")
set(DOUBLE_REAL TRUE CACHE BOOL "Should the real type be double(true) or float(false).")


add_library(${OUT_LIB_NAME} SHARED ${SRC_FILES})
target_link_libraries("cv" "png" "lapacke" "lapack" "blas" "jpeg")
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
