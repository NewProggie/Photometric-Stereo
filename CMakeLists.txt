CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(Photometric-Stereo)

# required packages
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(VTK REQUIRED)

#required macros and variables
INCLUDE(${VTK_USE_FILE})
ADD_DEFINITIONS(-std=c++11)

# add all sources automagically
FILE(GLOB SOURCES src/*.cpp src/*.h)

ADD_EXECUTABLE(main ${SOURCES})
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${VTK_LIBRARIES})