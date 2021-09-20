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


*/

#ifndef FLIMARRAY_H
#define FLIMARRAY_H

#define FLIMARRAYDOCSTRING \
"FlimArray -- two wrapped arrays, in the attributes 'lifetime' and 'intensity'"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


// Numpy includes

// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes
#include <numpy/arrayobject.h>

#define NDIMS 2

#endif