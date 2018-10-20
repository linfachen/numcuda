cmake_minimum_required(VERSION 3.5)

project (numcuda CXX)

# include directories
include_directories("/usr/include/python3.6m")


set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++14 ${CMAKE_CXX_FLAGS}")

file(GLOB DIR_SRCS src/*.cpp)


add_library(numcuda SHARED ${DIR_SRCS})