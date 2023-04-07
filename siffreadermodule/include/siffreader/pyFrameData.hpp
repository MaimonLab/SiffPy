// Python object corresponding to FrameData

#ifndef PYFRAMEDATA_HPP
#define PYFRAMEDATA_HPP

#include <Python.h>
#include <string>
#include "structmember.h"
#include <iostream>
#include "../siffreader/framedatastruct.hpp"

#define NO_IMPORT_ARRARY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes
#include <numpy/arrayobject.h>

#define PYFRAMEDATA_OBJECTNAME "FrameData"
#define PYFRAMEDATA_TPNAME "siffreader.FrameData"
#define PYFRAMEDATA_DOCSTRING "FrameData object"

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

static PyMemberDef pyframedata_members[] = {
    // Provides Python with access to the C++ struct members of the framedata
    
    //{"image_width", T_UINT, offsetof(PyFrameData, framedatastruct->imageWidth), READONLY, "image width"}

    
    // {"imageWidth", T_UINT, offsetof(PyFrameData, framedatastruct->imageWidth), 0, "image width"},
    // {"imageLength", T_UINT, offsetof(PyFrameData, framedatastruct->imageLength), 0, "image length"},
    // {"bitsPerSample", T_USHORT, offsetof(PyFrameData, framedatastruct->bitsPerSample), 0, "bits per sample"},
    // {"compression", T_USHORT, offsetof(PyFrameData, framedatastruct->compression), 0, "compression"},
    // {"photometric", T_USHORT, offsetof(PyFrameData, framedatastruct->photometric), 0, "photometric"},
    // {"endOfIFD", T_UINT, offsetof(PyFrameData, framedatastruct->endOfIFD), 0, "end of IFD"},
    // {"dataStripAddress", T_UINT, offsetof(PyFrameData, framedatastruct->dataStripAddress), 0, "data strip address"},

    {NULL},
};

extern PyTypeObject PyFrameDataType;

#endif