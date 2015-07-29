
project(libcv)
cmake_minimum_required(VERSION 2.8)

file(GLOB SRC_FILES "./src/*.cpp")

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "-std=c++11 -w -g -O0 -fopenmp")
else (NOT WIN32)
	# to be changed so it can compile on msc
	message(FATAL_ERROR "Windows not supported so far")
endif(NOT WIN32)

find_package(Qt4 REQUIRED)

link_directories("/usr/lib" "/usr/local/lib")
include_directories("/usr/include/qt4")

add_library("cv" SHARED ${SRC_FILES})

target_link_libraries("cv" "png")
target_link_libraries("cv" Qt4::QtGui Qt4::QtCore)

add_executable("test" "./test/test.cpp")

target_link_libraries("test" "cv")
