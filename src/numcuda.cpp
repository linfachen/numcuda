#include <Python.h>



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
	char * data;     
}cudaArray;


static PyTypeObject PycudaArray_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"cudaArray",                 /* tp_name */
	sizeof(cudaArray),         /* tp_basicsize */
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





static struct PyModuleDef samplemodule = {
	PyModuleDef_HEAD_INIT,
	"mydict", /* name of module */
	"c++ map module", /* Doc string (may be NULL) */
	-1, /* Size of per-interpreter state or -1 */
	 0 /* Method table */
};



PyMODINIT_FUNC
PyInit_numcuda(void) {
	PyObject *m;

	//if (PyType_Ready(&PyMydict_Type) < 0)
	//	return NULL;

	if (PyType_Ready(&PycudaArray_Type) < 0)
		return NULL;


	m = PyModule_Create(&samplemodule);

	Py_INCREF(&PyMydict_Type);
	PyModule_AddObject(m, "mydict", (PyObject *)&PyMydict_Type);

	return m;
}