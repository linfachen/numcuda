#include <iostream>
#include "cudaArray.h"
#include "cuda_runtime.h"
#include "cudaArray.h"
#include "kernel.h"

#include "cuda_fp16.h"


//args:shape buff dtype
static PyObject *new_cudaArray(int64_t *shape,int ndim,char *buff,Dtype dtype,bool use_buff=false);


template<typename DTYPE>
void _debug_data(void *data,int num=10)
{
    std::cout<<"********debug_data********"<<std::endl;
    for(int i=0;i<num;i++){
        std::cout<<*((DTYPE*)data+i)<<std::endl;
    }
}


size_t elem_size(Dtype type)
{
	size_t res = 0;
	switch(type){
		case Dtype::int8:
		case Dtype::uint8:
			res = 1;
			break;
		case Dtype::int16:
		case Dtype::uint16:
		case Dtype::float16:
			res = 2;
			break;
		case Dtype::int32:
		case Dtype::uint32:
		case Dtype::float32:
			res = 4;
			break;
		case Dtype::int64:
		case Dtype::uint64:
		case Dtype::float64:
			res = 8;
	}
	return res;
}


//get data size from shape
size_t get_data_size(int64_t * shape,int ndim,Dtype type)
{
	size_t res = elem_size(type);
    for(int i=0;i<ndim;i++){
        res *= *(shape+i);
    }
    return res;
}


//get_strides_from_shape
int64_t * get_strides_from_shape(int64_t * shape,int ndim,Dtype type)
{
    int64_t * res = (int64_t *)malloc(ndim* sizeof(int64_t));
	res[ndim-1] = 1;
	for(int i=2;i<=ndim;i++){
		res[ndim-i] = res[ndim-i+1]*shape[ndim-i+1];
	}
    return res;
}


bool is_same_shape(int64_t * a,int64_t * b,int a_dim,int b_dim)
{
    if(a_dim!=b_dim) return false;
    for(int i=0;i<a_dim;i++){
        if(*(a+i)!=*(b+i)) return false;
    }
    return true;
}


static PyObject *
asnumpy(PyObject *self, PyObject *args)
{
	PyCudaArray *cuda_array = (PyCudaArray *)self;
	PyObject *PyArray;
	if(cuda_array->data!=NULL){
		char * buff = (char *)malloc(cuda_array->buff_size);
		cudaMemcpy(buff,cuda_array->data,cuda_array->buff_size,cudaMemcpyDeviceToHost);
        //_debug_data<float>(buff,5);
		PyArray  = PyArray_SimpleNewFromData(cuda_array->ndim, cuda_array->shape, TO_NUMPY_TYPE[cuda_array->data_type], buff);
	}else{
		PyArray = Py_None;
		Py_INCREF(Py_None);
	}
	return PyArray;
}


/* tp_dealloc */
static void
cudaArray_dealloc(PyCudaArray *self)
{
    std::cout<<"cudaArray_dealloc is called"<<std::endl;
    if(self->data){
        cudaFree(self->data);
        self->data = NULL;
    }
    if(self->shape){
        free(self->shape);
    }
    if(self->strides){
        free(self->strides);
    }
}


/* tp_free */
void cudaArray_free(void *ptr)
{
    PyObject_Free(ptr);
}


//class_method
static PyMethodDef cudaArrayMethod[] = {
		{ "asnumpy", (PyCFunction)asnumpy, METH_NOARGS, "convert cudaArray to numpy array!" },
		{ 0 }
};


static PyObject * cudaArray_add(PyObject * a, PyObject *b)
{
    PyCudaArray *aa = (PyCudaArray *)a;
    PyCudaArray *bb = (PyCudaArray *)b;

    if((aa->data_type==bb->data_type)&&is_same_shape(aa->shape,bb->shape,aa->ndim,bb->ndim)){
        char *buff = NULL;
        cudaMalloc((void **)&buff,aa->buff_size);
        elt_add_op(aa->data,bb->data,buff,aa->buff_size,aa->data_type);
        PyObject *res = new_cudaArray(aa->shape,aa->ndim,buff,aa->data_type,true);
        return res;
    }else{
        std::cerr<<"add only support for same shape"<<std::endl;
        return NULL;
    }
}


static PyNumberMethods cudaArray_as_number = {
        (binaryfunc)cudaArray_add,     /*nb_add*/
        0,//(binaryfunc)cudaArray_sub,     /*nb_subtract*/
        0,//(binaryfunc)cudaArray_mul,     /*nb_multiply*/
        0,                           /*nb_remainder*/
        0,                           /*nb_divmod*/
        0,                           /*nb_power*/
        0,                           /*nb_negative*/
        0,                           /*tp_positive*/
        0,                           /*tp_absolute*/
        0,                           /*tp_bool*/
        0,                           /*nb_invert*/
        0,                           /*nb_lshift*/
        0,                           /*nb_rshift*/
        0,                           /*nb_and*/
        0,                           /*nb_xor*/
        0,                           /*nb_or*/
        0,                           /*nb_int*/
        0,                           /*nb_reserved*/
        0,                           /*nb_float*/
        0,                           /* nb_inplace_add */
        0,                           /* nb_inplace_subtract */
        0,                           /* nb_inplace_multiply */
        0,                           /* nb_inplace_remainder */
        0,                           /* nb_inplace_power */
        0,                           /* nb_inplace_lshift */
        0,                           /* nb_inplace_rshift */
        0,                           /* nb_inplace_and */
        0,                           /* nb_inplace_xor */
        0,                           /* nb_inplace_or */
        0,                           /* nb_floor_divide */
        0,                           /* nb_true_divide */
        0,                           /* nb_inplace_floor_divide */
        0,                           /* nb_inplace_true_divide */
        0,                           /* nb_index */
};



static PyTypeObject PyCudaArray_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"cudaArray",                 /* tp_name */
	sizeof(PyCudaArray),         /* tp_basicsize */
	0,                         /* tp_itemsize */
    (destructor)cudaArray_dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_as_async */
	0,                         /* tp_repr */
    &cudaArray_as_number,       /* tp_as_number */
	0,       					/* tp_as_sequence */
	0,       					/* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT ,  		 /* tp_flags */
	"a Array alloc in device",    /* tp_doc */
	0,          			 /* tp_traverse */
	0,                       /* tp_clear */
	0,                       /* tp_richcompare */
	0,                       /* tp_weaklistoffset */
	0, 						/* tp_iter */
	0,                       /* tp_iternext */
	cudaArrayMethod,           /* tp_methods */
	0,                        /* tp_members */
	0,                       /* tp_getset */
	0,                       /* tp_base */
	0,                       /* tp_dict */
	0,                       /* tp_descr_get */
	0,                       /* tp_descr_set */
	0,                       /* tp_dictoffset */
	0,                       /* tp_init */
	0,                       /* tp_alloc */
	0,              		 /* tp_new */
    cudaArray_free,          /* tp_free */
};

//args:shape buff dtype
static PyObject *
new_cudaArray(int64_t *shape,int ndim,char *buff,Dtype dtype,bool use_buff)
{
	PyCudaArray * res = PyObject_New(PyCudaArray,&PyCudaArray_Type);


    size_t buff_size = get_data_size(shape,ndim,dtype);
	res->buff_size = buff_size;
	res->data_type = dtype;
	res->ndim = ndim;
	res->shape = (int64_t *)malloc(ndim* sizeof(int64_t));
	memcpy(res->shape,shape,ndim* sizeof(int64_t));
	res->strides = get_strides_from_shape(shape,ndim,dtype);
    if(use_buff){
        res->data = buff;
    }else {
        cudaMalloc((void **) &res->data, buff_size);
        if (buff != NULL) {
            //_debug_data<float>(buff,5);
            //if buff!=NULL copy data to device
            cudaMemcpy(res->data, buff, buff_size, cudaMemcpyHostToDevice);
        }
    }
	return (PyObject *)res;
}


//create cudaArray from numpy array
static PyObject *
_array(PyObject *self, PyObject *obj)
{
	PyArrayObject *np_array = (PyArrayObject *)PyTuple_GET_ITEM(obj,0);
	import_array();

	if(PyArray_CheckExact(np_array)) {
		PyObject *res = new_cudaArray(np_array->dimensions,np_array->nd,np_array->data,NUMPY_CHARTYPE_TO_NUMCUDA_TYPE[(NPY_TYPECHAR)(np_array->descr->type)]);
		return res;
	}else{
		return NULL;
	}
}




/* Method table */
static PyMethodDef Numcuda_Methods[] = {
        {"arrays", _array, METH_VARARGS, "create cudaArray from numpy array!" },
        {"asnumpy",asnumpy,METH_VARARGS,"convert cudaArray to numpy array!"},
        { NULL, NULL, 0, NULL }
};



static struct PyModuleDef numcuda_emodule = {
	PyModuleDef_HEAD_INIT,
	"numcuda", /* name of module */
	"a module like numpy use cuda to accelerat computation", /* Doc string (may be NULL) */
	-1, /* Size of per-interpreter state or -1 */
	Numcuda_Methods /* Method table */
};



PyMODINIT_FUNC
PyInit_lib_numcuda(void) {
	PyObject *m;

	//if (PyType_Ready(&PyMydict_Type) < 0)
	//	return NULL;

	if (PyType_Ready(&PyCudaArray_Type) < 0)
		return NULL;


	m = PyModule_Create(&numcuda_emodule);

	Py_INCREF(&PyCudaArray_Type);
	PyModule_AddObject(m, "numcuda", (PyObject *)&PyCudaArray_Type);

	return m;
}