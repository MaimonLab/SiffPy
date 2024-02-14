// Python object corresponding to FrameData

#ifndef PYFRAMEDATA_HPP
#define PYFRAMEDATA_HPP

#include <Python.h>
#include <string>
#include "structmember.h"
#include <iostream>
#include "framedatastruct.hpp"

#define NO_IMPORT_ARRARY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes
#include <numpy/arrayobject.h>

#define PYFRAMEDATA_OBJECTNAME "FrameData"
#define PYFRAMEDATA_TPNAME "siffreader.FrameData"
#define PYFRAMEDATA_DOCSTRING "FrameData object"

/*
TODO: MAKE THIS A USABLE PYOBJECT
*/
typedef struct _PyFrameData {
    PyObject_VAR_HEAD
    const char *tp_name;
    Py_ssize_t tp_basicsize, tp_itemsize;

    destructor tp_dealloc;

    const char *tp_doc;

    //struct PyMethodDef *tp_methods;
    struct PyMemberDef *tp_members;
    //struct PyGetSetDef *tp_getset;
    newfunc tp_new;

    FrameData* framedatastruct;
} PyFrameData;

static PyObject* pyframedata_new(PyTypeObject* type, PyObject* args, PyObject* kwargs){
    PyFrameData* self;
    self = (PyFrameData*)type->tp_alloc(type, 0);
    return (PyObject*) self;
    // DANGEROUS not to properly initalize the framedatastruct. Will come back to.
};

static void pyframedata_dealloc(PyFrameData* self){
    Py_TYPE(self)->tp_free((PyObject*)self);
};

static PyMemberDef pyframedata_members[] = {
    // Provides Python with access to the C++ struct members of the framedata
    // though the framedata is just a pointer! So... maybe not the right format.
/*    {"image_width", T_UINT, offsetof(PyFrameData, framedatastruct->imageWidth), READONLY, "image width"},
    {"image_length", T_UINT, offsetof(PyFrameData, framedatastruct->imageLength), READONLY, "image length"},
    {"bits_per_sample", T_USHORT, offsetof(PyFrameData, framedatastruct->bitsPerSample), READONLY, "bits per sample"},
    {"compression", T_USHORT, offsetof(PyFrameData, framedatastruct->compression), READONLY, "compression"},
    {"end_of_ifd", T_UINT, offsetof(PyFrameData, framedatastruct->endOfIFD), READONLY, "end of IFD"},
    {"data_strip_address", T_UINT, offsetof(PyFrameData, framedatastruct->dataStripAddress), READONLY, "data strip address"},
    {"x_resolution", T_UINT, offsetof(PyFrameData, framedatastruct->xResolution), READONLY, "x resolution (as rational)"},
    {"y_resolution", T_UINT, offsetof(PyFrameData, framedatastruct->yResolution), READONLY, "y resolution (as rational)"},
    {"resolution_unit", T_USHORT, offsetof(PyFrameData, framedatastruct->resUnit), READONLY, "Unit: 1 = unknown, 2 = inch, 3 = cm"},
    {"frame_meta_data", T_STRING, offsetof(PyFrameData, framedatastruct->frameMetaData), READONLY, "Contains frame-specific metadata"},
*/    {NULL},
};

static PyTypeObject PyFrameDataType = {
    PyVarObject_HEAD_INIT(NULL,0)
    PYFRAMEDATA_TPNAME,        /* tp_name */
    sizeof(PyFrameData),       /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) pyframedata_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    PYFRAMEDATA_DOCSTRING,     /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    pyframedata_members,       /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    pyframedata_new            /* tp_new */
};


#endif