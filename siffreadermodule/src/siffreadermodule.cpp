/**
 * 
 * Interface with Python to read .siff files and meta data 
 * as quickly as possible and return appropriate data types
 * for data analysis. Or... at least that's the goal.
 * 
 * This got bloated and will one day certainly need a refactor.
 * For now, I'm putting off the problem. This was the first time
 * I'd written a python module and started with something simple
 * that didn't require good practices... maybe lesson learned.
 * Maybe not.
 * 
 * SCT March 01 2021
 * 
 * */
//

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// OK THIS GAVE ME SO MUCH HEADACHE SO I'M GONNA COMMENT IT HERE.
// the PyArray_API is defined as STATIC, so every file needs to call
// import_array() on its own. BUT, this can be avoided by doing this:
// in this file, I define a PY_ARRAY_UNIQUE_SYMBOL referring to this
// PyArray_API. Then, in any other file that includes numpy/arrayobject.h
// I define NO_IMPORT_ARRAY and define the unique symbol again.
// for more reading: https://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API
#ifndef NPY_NO_DEPRECATED_API
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>

#include "../include/siffreader/siffreader.hpp"
#include "../include/siffmoduledefin.hpp"
#include "../include/sifftotiff.hpp"
#include "../include/siffio/siffio.hpp"
#include "../include/framedata/pyFrameData.hpp"


//  I'm going to comment this in a way to remind myself how this process works
//  so hopefully I can use this as a reference doc for myself

// Compile with the headers for your python version. Which means, if you're using
// Anaconda, to pay close attention to your anaconda2/envs directory for include files. You can control
// this with the "include_dirs" argument in your setup.py but there's likely a smarter way using
// anaconda interactions directly. I'll update this when I get around to figuring that out.

/*
Warnings and utils
*/

static PyObject* siffreader_suppress_warnings(PyObject* self){
    // Suppresses warnings
    
    PyErr_SetString(
        PyExc_NotImplementedError,
        "Not implemented since the 0.6 rework"
    );

    return NULL;
    //Sf.suppressWarnings(true);
    Py_RETURN_NONE;
}

static PyObject* siffreader_report_warnings(PyObject* self){
    
    PyErr_SetString(
        PyExc_NotImplementedError,
        "Not implemented since the 0.6 rework"
    );
    
    return NULL;
   //Sf.suppressWarnings(false);

    Py_RETURN_NONE;
}

static PyObject* siffreader_debug(PyObject* self, PyObject *args){

    if(!PyArg_ParseTuple(args, "p:debug", &debug_SIFFIO)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Object passed to debug must be convertible to bool."
        );
        return NULL;
    }

    PyErr_SetString(
        PyExc_NotImplementedError,
        "Sorry, haven't implemented the 0.6.0 version debug mode yet"
    ); return NULL;
    
    Py_RETURN_NONE;
}

/**
 *
 *  Converts the open .siff file to a .tiff file and saves it in the location specified
 * 
 *  @param self the `SiffIO` object
 *  @param args the arguments passed to the function (no positional arguments)
 * @param kwargs the keyword arguments passed to the function (`sourcepath`, `savepath`, `mode`)
*/
static PyObject* siffreader_sifftotiff(PyObject *self, PyObject *args, PyObject *kwargs) {
    // PyErr_SetString(
    //     PyExc_NotImplementedError,
    //     "Siff to tiff conversion not implemented yet. "
    //     "Check status in the siffreadermodule src code. "
    // );
    // return NULL;

    char* sourcepath;
    Py_ssize_t sourcepath_len = Py_ssize_t(0);
    
    char* savepath;
    Py_ssize_t savepath_len = Py_ssize_t(0);

    char* mode;
    Py_ssize_t mode_len = Py_ssize_t(0);

    const char* scanimageMode = "scanimage";
    const char* omeMode = "ome";

    static const char *keywords[] = {"sourcepath", "savepath", "mode", NULL};

    if(
        !PyArg_ParseTupleAndKeywords(args, kwargs, "s#|$z#z#:siff_to_tiff", const_cast<char**>(keywords),
        &sourcepath, &sourcepath_len,
        &savepath, &savepath_len,
        &mode, &mode_len
        ))
        {
            return NULL;
    }
    try{
        TiffMode saveMode = SCANIMAGE;
        if (mode_len > 0){
            if (strcmp(mode, scanimageMode) == 0) {
                saveMode = SCANIMAGE;
            }
            else if (strcmp(mode, omeMode) == 0) {
                saveMode = OME;
                // If the extension is not right, fix it.
                if (savepath_len > 0) {
                    std::string savepathStr(savepath);
                    std::string extension = ".ome.tiff";
                    if (savepathStr.substr(savepath_len - extension.length(), extension.length()) != extension) {
                        savepathStr = savepathStr + extension;
                        savepath = const_cast<char*>(savepathStr.c_str());
                    }
                }
            }
            else {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Invalid mode specified. Must be either 'scanimage' or 'ome'."
                );
                return NULL;
            }
        }
        if (savepath_len > 0) {
            siff_to_tiff(sourcepath, savepath, saveMode);
        }
        else {
            siff_to_tiff(sourcepath, saveMode);
        }
    }
    catch(std::exception& e) {
        PyErr_SetString(
            PyExc_RuntimeError,
            e.what()
        );
        return NULL;
    }
    Py_RETURN_NONE;
}


/*

PyMethodDef, PyModuleDef

*/

static PyMethodDef SiffreaderMethods[] = {
// Array of the methods and corresponding docstrings
        {"suppress_warnings", (PyCFunction) siffreader_suppress_warnings, METH_NOARGS, SUPPRESS_WARNINGS_DOCSTRING},
        {"report_warnings", (PyCFunction) siffreader_report_warnings, METH_NOARGS, REPORT_WARNINGS_DOCSTRING},
        {"debug", siffreader_debug, METH_VARARGS, DEBUG_DOCSTRING},
        {"siff_to_tiff",(PyCFunction) siffreader_sifftotiff, METH_VARARGS|METH_KEYWORDS, SIFF_TO_TIFF_DOCSTRING},
        {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef siffreadermodule = {
// Defines the module, created during initialization function below
    PyModuleDef_HEAD_INIT, // must be the first property
    "siffpy.siffreadermodule",   /* name of module, this is what you type for "import XX" */
    PyDoc_STR(MODULE_DOC), /* module documentation, may be NULL*/ 
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SiffreaderMethods // PyMethodDef array of methods for the module
}; // OTHER OPTIONAL ELEMENTS OF ARRAY MIGHT BE ADDED. SOME USEFUL ONES INCLUDE:
// CLEAR (for Python's garbage collector)
// FREE (for deallocation)


PyMODINIT_FUNC PyInit_siffreadermodule(void) {
// THE  ONLY NONSTATIC OBJECT
// executes on initialization
    import_array(); // we use NumPy in here, so I have to run this before creating the module

    PyObject* module;
    
    if (PyType_Ready(&SiffIOType) < 0) return NULL;
    if (PyType_Ready(&PyFrameDataType) < 0) return NULL;
    
    module = PyModule_Create(&siffreadermodule);
    if (module == NULL) return NULL;

    Py_INCREF(&SiffIOType);
    if (
        PyModule_AddObject(
            module,
            SIFFIO_OBJECTNAME,
            (PyObject*) &SiffIOType
        ) < 0
    ) {
        Py_DECREF(&SiffIOType);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&PyFrameDataType);
    if (
        PyModule_AddObject(
            module,
            PYFRAMEDATA_OBJECTNAME,
            (PyObject*) &PyFrameDataType
        ) < 0
    ) {
        Py_DECREF(&PyFrameDataType);
        Py_DECREF(&SiffIOType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}