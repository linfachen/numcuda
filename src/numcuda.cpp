#include "cudaArray.h"
#include "cuda_runtime.h"
#include "cudaArray.h"


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
size_t get_data_size(PyObject * shape,Dtype type)
{
	size_t res = elem_size(type);
	if(PyTuple_CheckExact(shape)){
    	int num = PyTuple_GET_SIZE(shape);
    	for(int i=0;i<num;i++){
			PyObject * x = PyTuple_GET_ITEM(shape,i);
    		res *= PyLong_AsLong(x);
    	}
    }
    return res;
}

//get_strides_from_shape
PyObject * get_strides_from_shape(PyObject * shape,Dtype type)
{
	PyObject * res = NULL;
	size_t type_size = elem_size(type);
	if(PyTuple_CheckExact(shape)){
		int num = PyTuple_GET_SIZE(shape);
		res = PyTuple_New(num);
		PyTuple_SET_ITEM(res,num-1,PyLong_FromLong(type_size));
		for(int i=num-2;i>0;i--){
			PyObject * x = PyTuple_GET_ITEM(shape,i+1);
			type_size *= PyLong_AsLong(x);
			PyTuple_SET_ITEM(res,i,PyLong_FromLong(type_size));
		}
	}
    return res;
}

static PyTypeObject PyCudaArray_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"cudaArray",                 /* tp_name */
	sizeof(PyCudaArray),         /* tp_basicsize */
	0,                         /* tp_itemsize */
	0,                         /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_as_async */
	0,                         /* tp_repr */
	0,                        /* tp_as_number */
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
	0,                       /* tp_methods */
	0,                        /* tp_members */
	0,                       /* tp_getset */
	0,                       /* tp_base */
	0,                       /* tp_dict */
	0,                       /* tp_descr_get */
	0,                       /* tp_descr_set */
	0,                       /* tp_dictoffset */
	0,                       /* tp_init */
	0,                       /* tp_alloc */
	0,              			/* tp_new */
	0,             			 /* tp_free */
};

//args:shape buff dtype
static PyObject *
new_cudaArray(PyObject *shape, char *buff,Dtype dtype)
{
	PyCudaArray * res = PyObject_New(PyCudaArray,&PyCudaArray_Type);


    size_t buff_size = get_data_size(shape,dtype);
	res->buff_size = buff_size;
	res->data_type = dtype;
	Py_INCREF(shape);
	res->shape = shape;
	PyObject * strides = get_strides_from_shape(shape,dtype);
	res->strides = strides;

	cudaMalloc((void**)&res->data, buff_size);
	if(buff != NULL){
		//if buff!=NULL copy data to device
		cudaMemcpy(&res->data,buff,buff_size,cudaMemcpyHostToDevice);
	}
	return (PyObject *)res;
}


static PyObject *
asnumpy(PyObject *self,PyObject *cuda_array)
{
    return NULL;
}






/* Method table */
static PyMethodDef Numcuda_Methods[] = {
        //{"array", new_cudaArray, METH_VARARGS, "create a cudaArray!" },
        {"asnumpy",asnumpy,METH_VARARGS,"convert cudaArray to numpy array"},
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