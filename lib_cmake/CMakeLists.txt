cmake_minimum_required(VERSION 2.8)
project(mobileNet)

#set(inference_VERSION_MAJOR 2)
#set(inference_VERSION_MINOR 1)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

set (SERIAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/serial/src)
set (CAMERA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mindVisionApi/src)

set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/build)
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

message("The runtime libraries are included in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message("The library files are included in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

find_package(CUDA)
find_package(OpenCV REQUIRED)
message(" -- CUDA and Opencv Found ")
message(" -- opencv_version  "${OpenCV_VERSION})


set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};--disable-warnings;
        -O3
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_50,code=compute_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
	-gencode arch=compute_62,code=sm_62
        )


#file(GLOB sources util/*.cu util/*.cpp util/cuda/*.cu  util/cuda/*.cpp  tensorRTplugin/*.cpp  tensorRTplugin/*.cu)
file(GLOB includes tensorRTplugin/*.h util/cuda/*.h util/*.h)

# serial lib
#file(GLOB SERIAL_SRC ${SERIAL_DIR}/*.cc)
#add_library(serial SHARED ${SERIAL_SRC})
#install(TARGETS serial LIBRARY DESTINATION lib)

#mvCamera lib
#file(GLOB CAMERA_SRC ${CAMERA_DIR}/*.cpp)
#add_library(camera SHARED ${CAMERA_SRC})
#target_link_libraries(camera ${CMAKE_CURRENT_SOURCE_DIR}/mindVisionApi/lib/libMVSDK.so)
#install(TARGETS camera LIBRARY DESTINATION lib)

include_directories(${PROJECT_INCLUDE_DIR}/util)
include_directories(${PROJECT_BINARY_DIR}/util)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorRT4/ARM64/include
    ${SERIAL_DIR}
    ${CAMERA_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorRTplugin
    ${CMAKE_CURRENT_SOURCE_DIR}/util
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    )


# Camera Driver
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mindVisionApi/include)

add_executable(mobileNet ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

## OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(mobileNet
	${OpenCV_LIBS}
	${CMAKE_CURRENT_SOURCE_DIR}/mindVisionApi/lib/libMVSDK.so
	${CMAKE_CURRENT_SOURCE_DIR}/lib/libinferLib.so
	${CMAKE_CURRENT_SOURCE_DIR}/lib/libserial.so
	${CMAKE_CURRENT_SOURCE_DIR}/lib/libcamera.so
	${CUDA_LIBRARIES}
)
