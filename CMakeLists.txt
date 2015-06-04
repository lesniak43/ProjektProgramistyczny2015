cmake_minimum_required (VERSION 2.6) 

PROJECT(Final)

INCLUDE_DIRECTORIES( . )

FIND_PACKAGE( OpenCV REQUIRED )

ADD_EXECUTABLE( final main.cpp )
ADD_EXECUTABLE( get_data get_data.cpp )
ADD_EXECUTABLE( train train.cpp )
ADD_EXECUTABLE( test_model test_model.cpp )

add_library(utils utils.cpp)
add_library(consts consts.cpp)

TARGET_LINK_LIBRARIES( final  ${OpenCV_LIBS} utils consts)
TARGET_LINK_LIBRARIES( get_data  ${OpenCV_LIBS} utils consts)
TARGET_LINK_LIBRARIES( train  ${OpenCV_LIBS} utils consts)
TARGET_LINK_LIBRARIES( test_model  ${OpenCV_LIBS} utils consts)
