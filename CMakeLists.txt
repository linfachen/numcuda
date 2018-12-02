cmake_minimum_required(VERSION 3.8)

project (numcuda CXX)

# find python include path
find_package(PythonLibs 3 REQUIRED)

# find CUDA
find_package(CUDA)

# include directories
include_directories("include")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++14 ${CMAKE_CXX_FLAGS}")


file(GLOB DIR_SRCS src/*.cpp)
file(GLOB CU_SRCS src/kernel/*.cu)

cuda_add_library(kernel STATIC  ${CU_SRCS})
# message(${CUDA_INCLUDE_DIRS})
add_library(_numcuda SHARED ${DIR_SRCS})
target_link_libraries(_numcuda kernel)