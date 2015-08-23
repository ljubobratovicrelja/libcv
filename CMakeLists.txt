project(libcv)
cmake_minimum_required(VERSION 2.8)

# Project variables
set(OUT_LIB_NAME "cv")
SET(CMAKE_DEBUG_POSTFIX "d")

# Configuration variables
set(WITH_GUI TRUE CACHE BOOL "Compile GUI module of the libcv.")
set(WITH_JPEG TRUE CACHE BOOL "Compile libcv with jpeg support.")
set(DOUBLE_REAL TRUE CACHE BOOL "Should the real type be double(true) or float(false).")
set(COMPILE_EXAMPLES TRUE CACHE BOOL "Should the examples be compiled.")

file(GLOB SRC_FILES "./src/*.c*")
file(GLOB INC_FILES "./include/*.h*")


if (WIN32)
	set(CMAKE_CXX_FLAGS "/openmp /EHsc /W3")
	set(CMAKE_CXX_FLAGS_DEBUG "/DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "/Ox /DNDEBUG")

	set(ADDITIONAL_LIB_DIRECTORIES "" CACHE PATH "Path to the directory with libraries.")
	set(PNG_LIB "${CMAKE_CURRENT_LIST_DIR}/thirdParty/libpng/lib/libpng16.lib" CACHE FILEPATH "Path to the png library file")
	if (${WITH_JPEG})
		set(JPEG_LIB "jpeg" CACHE FILEPATH "Path to the jpeg library file")
	else (${WITH_JPEG})
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DCV_NO_JPEG")
	endif (${WITH_JPEG})
	set(OPENBLAS_LIB "${CMAKE_CURRENT_LIST_DIR}/thirdParty/blas/lib/libopenblas.dll.a" CACHE FILEPATH "Path to the blas library file")
elseif(NOT WIN32 AND UNIX)	
	set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -Wall")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

	set(ADDITIONAL_LIB_DIRECTORIES "/usr/local/lib" CACHE PATH "Path to the directory with libraries.")
	set(PNG_LIB "png" CACHE FILEPATH "Path to the png library file")
	if (${WITH_JPEG})
		set(JPEG_LIB "jpeg" CACHE FILEPATH "Path to the jpeg library file")
	else (${WITH_JPEG})
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_NO_JPEG")
	endif (${WITH_JPEG})
	set(LAPACK_LIB "lapack" CACHE FILEPATH "Path to the lapack library file")
	set(LAPACKE_LIB "lapacke" CACHE FILEPATH "Path to the lapacke library file")
	set(BLAS_LIB "blas" CACHE FILEPATH "Path to the blas library file")
endif(WIN32)

if (CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RELEASE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES DEBUG)

find_package(Qt4 4.8.6 REQUIRED)

include_directories(${QT_INCLUDE_DIR})
link_directories(${ADDITIONAL_LIB_DIRECTORIES})

if (NOT ${WITH_GUI})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_IGNORE_GUI")
endif(NOT ${WITH_GUI})

if(${DOUBLE_REAL})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_DOUBLE_REAL")
endif(${DOUBLE_REAL})

	
add_library(${OUT_LIB_NAME} SHARED ${SRC_FILES})
	
target_link_libraries(${OUT_LIB_NAME} ${PNG_LIB})
target_link_libraries(${OUT_LIB_NAME} Qt4::QtGui)
target_link_libraries(${OUT_LIB_NAME} Qt4::QtCore)

if (WIN32)
	target_link_libraries(${OUT_LIB_NAME} "imm32.lib")
	target_link_libraries(${OUT_LIB_NAME} "winmm.lib")
	target_link_libraries(${OUT_LIB_NAME} "Ws2_32.lib")
	target_link_libraries(${OUT_LIB_NAME} ${OPENBLAS_LIB})
else (WIN32)
	target_link_libraries(${OUT_LIB_NAME} ${LAPACK_LIB})
	target_link_libraries(${OUT_LIB_NAME} ${LAPACKE_LIB})
	target_link_libraries(${OUT_LIB_NAME} ${BLAS_LIB})
endif(WIN32)

if (${WITH_JPEG})
	target_link_libraries(${OUT_LIB_NAME} ${JPEG_LIB})
endif (${WITH_JPEG})

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
	
	# setup examples
	install(FILES ${EXAMPLE_FILES} DESTINATION "examples/")
endif(${COMPILE_EXAMPLES})

# setup install
install(TARGETS ${OUT_LIB_NAME} 
                RUNTIME DESTINATION bin
                LIBRARY DESTINATION lib
                ARCHIVE DESTINATION lib/static)
                
#install headers
install(FILES ${INC_FILES} DESTINATION "include/cv")
