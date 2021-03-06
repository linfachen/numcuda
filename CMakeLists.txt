cmake_minimum_required(VERSION 3.9)

project (numcuda CXX)

# find python include path
find_package(PythonLibs 3 REQUIRED)

# find CUDA
find_package(CUDA)

set(USE_GDB ON)

if(USE_GDB)
    SET(CMAKE_BUILD_TYPE "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
endif()

# include directories
include_directories("include")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories("/usr/local/lib/python3.6/dist-packages/numpy/core/include")



set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++14 ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ;-gencode arch=compute_70,code=sm_70)

file(GLOB DIR_SRCS src/*.cpp)
file(GLOB CU_SRCS src/kernel/*.cu)


cuda_add_library(kernel STATIC  ${CU_SRCS})
# message(${CUDA_INCLUDE_DIRS})
add_library(_numcuda SHARED ${DIR_SRCS})
target_link_libraries(_numcuda kernel)