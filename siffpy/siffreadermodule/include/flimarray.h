/*

FlimArray

A PyObject class for handling FLIM data.

Wraps two numpy arrays and overloads + and * operators to
appropriately handle the interactions between FLIM data.

You can't just add the lifetimes (or average them), but I'd
like them to be treated like numpy arrays, so I thought it could
be easier to simply make the operators behave correctly, i.e.
summing two FlimArrays x and y results in a new FlimArray z where

z.lifetime = (x.intensity*x.lifetime + y.intensity*y.lifetime)/(x.intensity + y.intensity)

z.intensity = x.intensity + y.intensity

Work in progress...

Currently implemented as a native Python object, but I think there are good reasons
to do it in C (to speed up some types of operations)


*/

#ifndef FLIMARRAY_H
#define FLIMARRAY_H

#define FLIMARRAYDOCSTRING \
"FlimArray -- two wrapped arrays, in the attributes 'lifetime' and 'intensity'\n" \
"Implements addition in a FLIM-compatible format, blending lifetime and intensity appropriately.\n" \
"But behaves like a standard numpy array, in terms of its shape etc. (or I try to make it do so)."

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


// Numpy includes

// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
//#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes
#include <numpy/arrayobject.h>

#define FLIMARRAY_TYPE FlimArrayType

#define PyFLIMArray_Check(op) PyObject_TypeCheck(op, &FlimArrayType)

typedef struct {
    PyObject_HEAD
    PyArrayObject* lifetime; // empirical lifetime, by pixel
    PyArrayObject* intensity; // photon counts, by pixel
    int ndims;
    npy_intp* dimensions; // shape, length ndims
    /*
     * Number of bytes to jump to get to the
     * next element in each dimension
     */
    npy_intp *strides;
} FlimArray;

// Called on deallocation
static void FlimArray_dealloc(FlimArray *self) {
    Py_XDECREF(self->lifetime);
    Py_XDECREF(self->intensity);    
    Py_TYPE(self)->tp_free((PyObject*) self);
}

// Allocate
static PyObject* FlimArray_new(PyTypeObject * type, PyObject *args, PyObject *kwds){
    FlimArray *self;

    return (PyObject*) self;
}

// Initialize
static int FlimArray_init(FlimArray *self, PyObject *args, PyObject *kwds) {

    return 0;
}

// Defines the methods
static PyMethodDef FlimArray_methods[] = {
    // TODO: Fill out!!
    {NULL}  /* Sentinel */
};

// Defines the attributes as accessed by Python
static PyMemberDef FlimArray_members[] = {
    {"intensity"},
    {"lifetime"},
    {"shape"},
    {NULL}
};

static PyTypeObject FlimArrayType = { // This is the --type-- object
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "siffreader.FlimArray",
    .tp_doc = FLIMARRAYDOCSTRING, // TODO do this right
    .tp_basicsize = sizeof(FlimArray),
    .tp_itemsize = 0, // MAYBE WRONG
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = FlimArray_new,
    .tp_init = (initproc) FlimArray_init,
    .tp_dealloc = (destructor) FlimArray_dealloc,
    .tp_methods = FlimArray_methods,
    .tp_members = FlimArray_members,
};

#endif