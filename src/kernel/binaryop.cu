#include <stdint.h>
#include <iostream>
#include "util.h" 
#include "cuda_fp16.h"

template<typename T>
__global__ void elt_add(T *a,T *b,T *c,int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<size){
        *(c+i) = *(a+i) + *(b+i);
    } 
}

template<>
__global__ void elt_add(__half *a,__half *b,__half *c,int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<size){
        *(c+i) = __hadd(*(a+i), *(b+i));
    } 
}


void elt_add_op(char *a,char *b,char *c,int size,Dtype dtype,int thread_size=32)
{
    int block_size = std::ceil(size*1.0/thread_size);
    switch(dtype){
        case Dtype::float16:
            elt_add<<<block_size,thread_size>>>((__half*)a,(__half*)b,(__half*)c,size);
            break;
        case Dtype::float32:
            elt_add<<<block_size,thread_size>>>((float*)a,(float*)b,(float*)c,size);
            break;
        case Dtype::float64: 
            elt_add<<<block_size,thread_size>>>((double*)a,(double*)b,(double*)c,size);
            break;
        case Dtype::int8:
            elt_add<<<block_size,thread_size>>>((int8_t*)a,(int8_t*)b,(int8_t*)c,size);
            break;
        case Dtype::uint8:
            elt_add<<<block_size,thread_size>>>((uint8_t*)a,(uint8_t*)b,(uint8_t*)c,size);
            break;
        case Dtype::int16:
            elt_add<<<block_size,thread_size>>>((int16_t*)a,(int16_t*)b,(int16_t*)c,size);
            break;
        case Dtype::uint16:
            elt_add<<<block_size,thread_size>>>((uint16_t*)a,(uint16_t*)b,(uint16_t*)c,size);
            break;
        case Dtype::int32:
            elt_add<<<block_size,thread_size>>>((int32_t*)a,(int32_t*)b,(int32_t*)c,size);
            break;
        case Dtype::uint32:
            elt_add<<<block_size,thread_size>>>((uint32_t*)a,(uint32_t*)b,(uint32_t*)c,size);
            break;
        case Dtype::int64:
            elt_add<<<block_size,thread_size>>>((int64_t*)a,(int64_t*)b,(int64_t*)c,size);
            break;
        case Dtype::uint64:
            elt_add<<<block_size,thread_size>>>((uint64_t*)a,(uint64_t*)b,(uint64_t*)c,size);
            break;
        default:
            std::cerr<<"use wrong type"<<std::endl;                
    }
}


