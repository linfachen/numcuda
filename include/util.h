//
// Created by spinors on 18-12-3.
//
#ifndef NUMCUDA_UTILL_H
#define NUMCUDA_UTILL_H

#include <unordered_map>
#include "numpy/arrayobject.h"

enum class Dtype
{
    //uint1,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
};

size_t elem_size(Dtype type);

extern std::unordered_map<Dtype,NPY_TYPES>   TO_NUMPY_TYPE;
extern std::unordered_map<Dtype,NPY_TYPECHAR>   TO_NUMPY_CHARTYPE;
extern std::unordered_map<NPY_TYPES,Dtype>   TO_NUMCUDA_TYPE;
extern std::unordered_map<NPY_TYPECHAR,Dtype>   NUMPY_CHARTYPE_TO_NUMCUDA_TYPE;



#endif //NUMCUDA_UTILL_H
