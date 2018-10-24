//
// Created by spinors on 18-10-21.
//
#include <Python.h>


#ifndef NUMCUDA_CUDAARRAY_H
#define NUMCUDA_CUDAARRAY_H

enum class Dtype
{
    uint1,
    int8,
    int16,
    int32,
    int64,
    unit8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
};


typedef struct {
    PyObject_HEAD
    size_t buff_size;
    Dtype data_type;
    PyObject * shape; //it always a tupleobject or listobject
    PyObject * strides; //it always a tupleobject or listobject
    char * data;
}PyCudaArray;


PyObject * get_strides_from_shape(PyObject *);





#endif //NUMCUDA_CUDAARRAY_H
