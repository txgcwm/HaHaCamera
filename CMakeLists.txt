cmake_minimum_required(VERSION 3.0)
project(cam)

AUX_SOURCE_DIRECTORY(hahacamera DIR_SRCS)

add_executable(cam ${DIR_SRCS})
include_directories(hahacamera)


set(CMAKE_INCLUDE_CURRENT_DIR ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

# TODO:
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
    set(OPENCV_ROOT_DIR /usr/local/Cellar/opencv3/3.0.0)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(OpenCV REQUIRED)
#include_directories(${OPENCV_INCLUDE_DIRS})
#message(STATUS "FIND GLFW_INCLUDE_DIRS " ${OpenCV_INCLUDE_DIRS})



set(ALL_LIBS
        ${OpenCV_LIBS}
)

target_link_libraries(cam ${ALL_LIBS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

install(TARGETS cam RUNTIME DESTINATION bin)
