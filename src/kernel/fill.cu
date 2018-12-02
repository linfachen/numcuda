#include <stdint.h>
#include <iostream>
#include "util.h" 
#include "cuda_fp16.h"

template<typename T>
__global__ void fill(T *data,T value,int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<size){
        *(data+i) = value;
    } 
}


void fill_value(char *data,double value,int size,Dtype dtype,int thread_size=32)
{
    int block_size = std::ceil(size*1.0/thread_size);
    switch(dtype){
        case Dtype::float16:
            fill<<<block_size,thread_size>>>((__half*)data,__float2half(static_cast<float>(value)),size);
            break;
        case Dtype::float32:
            fill<<<block_size,thread_size>>>((float*)data,static_cast<float>(value),size);
            break;
        case Dtype::float64: 
            fill<<<block_size,thread_size>>>((double*)data,value,size);
            break;
        default:
            std::cerr<<"use wrong type"<<std::endl;           
    }
}


void fill_value(char *data,long long value,int size,Dtype dtype,int thread_size=32)
{
    int block_size = std::ceil(size*1.0/thread_size);
    switch(dtype){
        case Dtype::int8:
        fill<<<block_size,thread_size>>>((int8_t*)data,static_cast<int8_t>(value),size);
        break;
        case Dtype::uint8:
        fill<<<block_size,thread_size>>>((uint8_t*)data,static_cast<uint8_t>(value),size);
        break;
        case Dtype::int16:
        fill<<<block_size,thread_size>>>((int16_t*)data,static_cast<int16_t>(value),size);
        break;
        case Dtype::uint16:
        fill<<<block_size,thread_size>>>((uint16_t*)data,static_cast<uint16_t>(value),size);
        break;
        case Dtype::int32:
        fill<<<block_size,thread_size>>>((int32_t*)data,static_cast<int32_t>(value),size);
        break;
        case Dtype::uint32:
        fill<<<block_size,thread_size>>>((uint32_t*)data,static_cast<uint32_t>(value),size);
        break;
        case Dtype::int64:
        fill<<<block_size,thread_size>>>((int64_t*)data,static_cast<int64_t>(value),size);
        break;
        case Dtype::uint64:
        fill<<<block_size,thread_size>>>((uint64_t*)data,static_cast<uint64_t>(value),size);
        break;
        default:
            std::cerr<<"use wrong type"<<std::endl;   
    }
}