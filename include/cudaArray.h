//
// Created by spinors on 18-10-21.
//
#include <Python.h>
#include "util.h"

#ifndef NUMCUDA_CUDAARRAY_H
#define NUMCUDA_CUDAARRAY_H



typedef struct {
    PyObject_HEAD
    size_t buff_size;
    Dtype data_type;
    PyObject * shape; //it always a tupleobject or listobject
    PyObject * strides; //it always a tupleobject or listobject
    char * data;
}PyCudaArray;

size_t elem_size(Dtype type);
PyObject * get_strides_from_shape(PyObject *);





#endif //NUMCUDA_CUDAARRAY_H
