#include "../include/siffreader/pyFrameData.hpp"
#define PY_SSIZE_T_CLEAN

static PyObject* framedata_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    PyFrameData* self = (PyFrameData*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->framedatastruct = new FrameData();
    }
    return (PyObject*)self;
};
/*
static int PyObject* framedata_init(PyFrameData* self, PyObject* args){
    
};
*/

static void framedata_dealloc(PyFrameData* self) {
    delete self->framedatastruct;
    Py_TYPE(self)->tp_free((PyObject*)self);
};

PyTypeObject PyFrameDataType = {
    PyVarObject_HEAD_INIT(NULL,0)
    .tp_name = PYFRAMEDATA_TPNAME,
    .tp_basicsize = sizeof(PyFrameData),
    .tp_itemsize = 0,
    
    .tp_dealloc = (destructor)framedata_dealloc,

    .tp_doc = PYFRAMEDATA_DOCSTRING,
    
    .tp_members = pyframedata_members,

    .tp_new = framedata_new,
};
