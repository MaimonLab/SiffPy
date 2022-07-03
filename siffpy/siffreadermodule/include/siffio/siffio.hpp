/*

Defines a Python class siffio which
handles file I/O for each file type. This
cleans up SiffPy pretty dramatically relative
to earlier versions:

1) Allows multiple files to be opened from one
interpreter

2) Allows each SiffReader object from SiffPy to
operate independently

TODO: IMPLEMENT

*/

#ifndef SIFFIO_HPP
#define SIFFIO_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include "structmember.h"
#include <iostream>
#include "../siffreader/siffreader.hpp"
#include "siffiodocstring.hpp"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes
#include <numpy/arrayobject.h>

typedef struct _SiffIO {
    PyObject_VAR_HEAD
    const char *tp_name; /* For printing, in format "<module>.<name>" */
    Py_ssize_t tp_basicsize, tp_itemsize; /* For allocation */

    /* Methods to implement standard operations */

    destructor tp_dealloc;
    reprfunc tp_repr;


    const char *tp_doc; /* Documentation string */

    /* Attribute descriptor and subclassing stuff */
    struct PyMethodDef *tp_methods;
    struct PyMemberDef *tp_members;
    struct PyGetSetDef *tp_getset;
    newfunc tp_new;

    SiffReader* siffreader; // each SiffIO has a C++ siffreader
    std::string status;
    std::iostream debug_log; // only defined when debug called. TODO
} SiffIO;


static PyMemberDef siffio_members[] = {
    //{Attribute name, attribute type, location in struct, flags, docstring}
//    {"filename", T_OBJECT_EX, offsetof(SiffIO, filename), 0, "Name of open file."},
    {NULL},
};


static PyObject* siffio_open(SiffIO* self, PyObject* args);
static PyObject* siffio_close(SiffIO* self);
static PyObject* siffio_get_file_header(SiffIO* self);
static PyObject* siffio_num_frames(SiffIO* self);

// Frame methods
static PyObject* siffio_get_frames(SiffIO* self, PyObject* args, PyObject *kwargs);
static PyObject* siffio_get_frame_metadata(SiffIO* self, PyObject* args, PyObject *kwargs);
static PyObject* siffio_pool_frames(SiffIO* self, PyObject* args, PyObject *kwargs);

// Flim methods
static PyObject* siffio_flim_map(SiffIO* self, PyObject* args, PyObject* kwargs);

// ROI methods
static PyArrayObject* siffio_sum_roi(SiffIO* self, PyObject* args, PyObject* kwargs);
static PyArrayObject* siffio_sum_roi_flim(SiffIO* self, PyObject* args, PyObject* kwargs);

// Histogram methods
static PyArrayObject* siffio_get_histogram(SiffIO* self, PyObject* args, PyObject* kwargs);

static PyMethodDef siffio_methods[] = {
    // {Method name, (PyCFunction) Function, flags, doc}
    {"open", (PyCFunction) siffio_open, METH_VARARGS, siffio_open_doc},
    {"close", (PyCFunction) siffio_close, METH_NOARGS, siffio_close_doc},
    {"get_file_header", (PyCFunction) siffio_get_file_header, METH_NOARGS, siffio_get_file_header_doc},
    {"num_frames", (PyCFunction) siffio_num_frames, METH_NOARGS, siffio_num_frames_doc},

    // Frame methods
    {"get_frames", (PyCFunction) siffio_get_frames, METH_VARARGS|METH_KEYWORDS, siffio_get_frames_doc},
    {"get_frame_metadata", (PyCFunction) siffio_get_frame_metadata, METH_VARARGS|METH_KEYWORDS, siffio_get_frame_metadata_doc},
    {"pool_frames", (PyCFunction) siffio_pool_frames, METH_VARARGS|METH_KEYWORDS, siffio_pool_frames_doc},
    
    // Flim methods
    {"flim_map", (PyCFunction) siffio_flim_map, METH_VARARGS|METH_KEYWORDS, siffio_flim_map_doc},

    // ROI methods
    {"sum_roi", (PyCFunction) siffio_sum_roi, METH_VARARGS|METH_KEYWORDS, siffio_sum_roi_doc},
    {"sum_roi_flim", (PyCFunction) siffio_sum_roi_flim, METH_VARARGS|METH_KEYWORDS, siffio_sum_roi_flim_doc},
    
    //Histogram methods
    {"get_histogram", (PyCFunction) siffio_get_histogram, METH_VARARGS|METH_KEYWORDS, siffio_get_histogram_doc},
    {NULL},
};

static PyObject* siffio_filename_get(SiffIO* self);
static PyObject* siffio_filename_set(SiffIO* self, PyObject* args);
static PyObject* siffio_status_get(SiffIO*self);
static PyObject* siffio_status_set(SiffIO* self, PyObject* args);

static PyGetSetDef siffio_getset[] = {
    {"filename", (getter) siffio_filename_get, (setter) siffio_filename_set, PyDoc_STR("Retrieves filename from C++ object.")},
    {"status" , (getter) siffio_status_get , (setter) siffio_status_set, PyDoc_STR("Checks the status of the SiffIO's SiffReader")},
    {NULL},
};

extern PyTypeObject SiffIOType;

#endif