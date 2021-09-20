/* METHOD DEFS

SCT Sept 17 2021

*/
#include "../include/flimarray.h"

typedef struct {
    PyObject_HEAD
    PyArrayObject* lifetime; // empirical lifetime, by pixel
    PyArrayObject* intensity; // photon counts, by pixel
    int dims[NDIMS];            // always 2-dimensional!!
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

// Defines the attributes
static PyMemberDef FlimArray_members[] = {
    {"intensity"},
    {"lifetime"},
    {"dims"},
    {NULL}
};

static PyTypeObject FlimArrayType = { // This is the --type-- object
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "siffpy.FlimArray",
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