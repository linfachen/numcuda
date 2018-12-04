//
// Created by spinors on 18-10-21.
//
#ifndef NUMCUDA_CUDAARRAY_H
#define NUMCUDA_CUDAARRAY_H

#include <Python.h>
#include "util.h"


typedef struct {
    PyObject_HEAD
    size_t buff_size;
    Dtype data_type;
    int ndim;
    int64_t * shape; //it always a tupleobject or listobject
    int64_t * strides; //it always a tupleobject or listobject
    char * data;
}PyCudaArray;

size_t elem_size(Dtype type);
PyObject * get_strides_from_shape(PyObject *);

#endif //NUMCUDA_CUDAARRAY_H
