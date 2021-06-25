/**
 * 
 * Interface with Python to read .siff files and meta data 
 * as quickly as possible and return appropriate data types
 * for data analysis. Or... at least that's the goal.
 * 
 * SCT March 01 2021
 * 
 * */

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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes, some compiler thing I don't really understand
#include <numpy/arrayobject.h>

#include "siffreader.hpp"
#include "siffmoduledefin.hpp"

static SiffReader Sf;

//  I'm going to comment this in a way to remind myself how this process works
//  so hopefully I can use this as a reference doc for myself

// What you need to do: build this file a la the Python C API guide to make a .so object, then put that .so object in your
// Python's lib path. If you use setuptools or distutil tools, it will do both automatically.
// But! Don't forget to put all your .c and .cpp files in the source. Whoops.

// Also important, compile with the headers for your python version. Which means, if you're using
// Anaconda, to pay close attention to your anaconda2/envs directory for include files. You can control
// this with the "include_dirs" argument in your setup.py but there's likely a smarter way using
// anaconda interactions directly. I'll update this when I get around to figuring that out.

static PyObject* siffreader_open(PyObject *self, PyObject *args) {
    // returns a PyObject to Python when called.
    const char *filename;

    if (!PyArg_ParseTuple(args, "s:open", &filename)) // parses PyObject args, "s" says it should be a string, loads into filename
        // PyArg_ParseTuple returns 0 if it fails to parse
        return NULL; // functions as an exception (TODO read more into this)
    
    int ret = Sf.openFile(filename); // Opens the file in the SiffReader
    if (ret < 0) {
        if (ret == -2) {
            PyErr_WarnEx(PyExc_RuntimeWarning, Sf.getErrString(),Py_ssize_t(1));
            Py_RETURN_NONE;
        }
        else PyErr_SetString(PyExc_FileNotFoundError,Sf.getErrString());
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject* siffreader_close(PyObject *self, PyObject *args) {
    // simple: returns the file header data

    if(!PyArg_ParseTuple(args, ":close")) return NULL;
    // if there are any args, it shouldn't go through

    Py_RETURN_NONE;
}

static PyObject* siffreader_get_file_header(PyObject *self, PyObject *args) {
    // simple: returns the file header data

    if(!PyArg_ParseTuple(args, ":get_file_header")) return NULL;
    // if there are any args, it shouldn't go through

    return Sf.readFixedData();
}

static PyObject* siffreader_get_frames(PyObject *self, PyObject *args, PyObject* kw) {
    // 

    PyObject *frames_list = NULL;
    PyObject *type = NULL;
    bool flim = false;
    PyObject* registrationDict = NULL;
    PyObject* discard_bins = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!O!pO!O:get_frames", const_cast<char**>(GET_FRAMES_KEYWORDS),
        &PyList_Type, &frames_list,
        &PyType_Type, &type,
        &flim,
        &PyDict_Type, &registrationDict,
        &discard_bins
        )
    ) {
        PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

    // default mode: get all frames as an intensity profile
    if(!frames_list) {
        return Sf.retrieveFrames((uint64_t *) NULL, 0, flim);
    }

    if(!registrationDict) registrationDict = PyDict_New();
    if(!PyObject_TypeCheck(registrationDict, &PyDict_Type)) {
        Py_DECREF(registrationDict);
        registrationDict = PyDict_New();
    }
    else {
        uint64_t framesArray[PyList_Size(frames_list)];
        for(Py_ssize_t idx = Py_ssize_t(0); idx < PyList_Size(frames_list); idx++) {
            PyObject* item = PyList_GET_ITEM(frames_list, idx);
            if(!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of frame list must be ints");
                return NULL;
            }
            framesArray[idx] = (uint64_t) PyLong_AsUnsignedLongLong(item);

            // if this isn't in the registration dict, shift by (0,0)
            if (!PyDict_Contains(registrationDict, item)) {
                PyDict_SetItem(registrationDict, item, // steals the reference to the value
                    PyTuple_Pack(Py_ssize_t(2), // steals references, makes life easier
                        PyLong_FromLong(0),
                        PyLong_FromLong(0)
                    )
                );
            }
        }
        uint64_t framesN = PyList_Size(frames_list);
        try{
            if (discard_bins) {
                if (!(PyLong_Check(discard_bins) || PyFloat_Check(discard_bins))) {
                    return Sf.retrieveFrames(framesArray, framesN,flim, registrationDict);
                }
                else {
                    uint64_t terminalBin = int(PyLong_AsLongLong(discard_bins));
                    return Sf.retrieveFrames(framesArray, framesN,flim, registrationDict, terminalBin);
                }
            }
            else{
                return Sf.retrieveFrames(framesArray, framesN,flim, registrationDict);
            }
        }
        catch(...) {
            PyErr_SetString(PyExc_RuntimeError, Sf.getErrString());
            return NULL;
        }
    }
}

static PyObject* siffreader_get_frame_metadata(PyObject *self, PyObject *args, PyObject* kw) {
    // Gets the meta data for the frames in kw, or for all the frames.  
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_frames_metadata", 
            const_cast<char**>(GET_FRAMES_METADATA_KEYWORDS), 
            &PyList_Type, &frames_list)
        ) {
        return NULL;
    }

    // default mode: get all frames
    if(!frames_list) {
        return Sf.readMetaData();
    }
    
    else {
        uint64_t framesArray[PyList_Size(frames_list)];
        for(Py_ssize_t idx = Py_ssize_t(0); idx< PyList_Size(frames_list); idx++) {
            // these references are BORROWED, so we don't have to worry about INCREF or DECREF
            PyObject* item = PyList_GET_ITEM(frames_list, idx);
            if(!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements of frame list must be ints");
                return NULL;
            }
            framesArray[idx] = (uint64_t) PyLong_AsUnsignedLongLong(item);
        }
        uint64_t framesN = PyList_Size(frames_list);
        return Sf.readMetaData(framesArray, framesN);
    }
    
}

static PyObject * siffreader_pool_frames(PyObject* self, PyObject *args, PyObject* kw) {
    // Pools frames together, returns list of pooled frames.
    PyObject* listOfFramesListed = NULL;
    PyObject* type = NULL;
    bool flim = false;
    PyObject* registrationDict = NULL;
    PyObject* discard_bins = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "O!|$O!pO!O:pool_frames", 
        const_cast<char**>(POOL_FRAMES_KEYWORDS),
        &PyList_Type, &listOfFramesListed,
        &PyType_Type, &type,
        &flim,
        &PyDict_Type, &registrationDict,
        &discard_bins
        )
    ) {
        return NULL;
    }

    // defaults to 0's
    if(!registrationDict) registrationDict = PyDict_New();
    if(!PyObject_TypeCheck(registrationDict, &PyDict_Type)) {
        // can't help but think there's a DECREF that should go in here.
        registrationDict = PyDict_New();
    }

    // Check that listOfFramesListed is a list of lists, and that the elements of that are ints.
    for (Py_ssize_t idx = Py_ssize_t(0); idx < PyList_Size(listOfFramesListed); idx++) {
        PyObject* item = PyList_GET_ITEM(listOfFramesListed, idx);
        if(!PyList_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All elements of pool_list must be lists themselves");
            return NULL;
        }
        
        // Have been surprised by encountering int overflow here, 65k photons per pixel requires either massive data
        // rates or loooots of pooling. I should do smarter checking but this is a short term solution.
        if (PyList_Size(item) > 10000) {
            PyErr_WarnEx(PyExc_RuntimeWarning, "Pooling a large number of frames! May cause uint16 overflow.",Py_ssize_t(1));
        }

        // check that elements of this list are ints. If so, pass them along to be used by the siffreader functionality. 
        for(Py_ssize_t subidx = Py_ssize_t(0); subidx < PyList_Size(item); subidx++) {
            PyObject* element = PyList_GET_ITEM(item, subidx);
            if(!PyLong_Check(element)) {
                PyErr_SetString(PyExc_TypeError, "All elements of sublists in pool_list must be of type int.");
                return NULL;
            }

            // if this isn't in the registration dict, shift by (0,0)
            if (!PyDict_Contains(registrationDict, element)) {
                PyDict_SetItem(registrationDict, element, // steals the reference to the value
                    PyTuple_Pack(Py_ssize_t(2), // steals references, makes life easier
                        PyLong_FromLong(0),
                        PyLong_FromLong(0)
                    )
                );
            }
        }
    }


    try{
        if (discard_bins) {
            if (!(PyLong_Check(discard_bins) || PyFloat_Check(discard_bins))) {
                return Sf.poolFrames(listOfFramesListed, flim, registrationDict);
            }
            else {
                uint64_t terminalBin = int(PyLong_AsLongLong(discard_bins));
                return Sf.poolFrames(listOfFramesListed, terminalBin, flim, registrationDict);
            }
        }
        else{
            return Sf.poolFrames(listOfFramesListed, flim, registrationDict);
        }
    }
    catch(...) {
        PyErr_SetString(PyExc_RuntimeError, Sf.getErrString());
        return NULL;
    }
}

static PyObject* siffreader_flim_map(PyObject* self, PyObject* args, PyObject* kw) {
    // Takes in a FLIMParams object and frames desired and returns a lifetime map, an intensity distribution,
    // and a chi-squared value for every pixel.

    PyObject* FLIMParams = NULL;
    PyObject* listOfFramesListed = NULL;
    char* conf_measure;
    uint16_t conf_measure_length = 0;
    PyObject* registrationDict = NULL;
    PyObject* discard_bins = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "O|$O!s#O!pO:flim_map", const_cast<char**>(FLIM_MAP_KEYWORDS),
        &FLIMParams,
        &PyList_Type,&listOfFramesListed,
        &conf_measure, &conf_measure_length,
        &PyDict_Type, &registrationDict,
        &discard_bins
        )) {
        return NULL;
    }

    if(!conf_measure_length) conf_measure = (char*) "chi_sq";
    // defaults to 0's
    if(!registrationDict) registrationDict = PyDict_New();
    if(!PyObject_TypeCheck(registrationDict, &PyDict_Type)) {
        // can't help but think there's a DECREF that should go in here.
        registrationDict = PyDict_New();
    }

    // Check that FLIMParams is of type siffutils.flimparams.FLIMParams
    if (strcmp(FLIMParams->ob_type->tp_name,"FLIMParams")){

        PyErr_SetString(PyExc_TypeError, 
            strcat((char*)"Expected params to be of type FLIMParams. Instead is type: ",
                FLIMParams->ob_type->tp_name
            )
        );
        return NULL;
    }

    // check that conf_measure is one of the permitted values
    if (strcmp(conf_measure, "log_p") && strcmp(conf_measure,"chi_sq") && strcmp(conf_measure,"None")) {
        PyErr_SetString(PyExc_TypeError, 
            strcat((char*) "Expected confidence_measure to be one of 'log_p', 'chi_sq', 'None'. Instead is type: ",
                conf_measure
            )
        );
        return NULL;
    }

    if (!listOfFramesListed) { // default behavior, skip type-checking.
        try{
            PyErr_SetString(PyExc_RuntimeError, "All frames default is not yet implemented");
            return NULL;
            //return Sf.flimMap(FLIMParams, listOfFramesListed, conf_measure, registrationDict);
        }
        catch(...) {
            PyErr_SetString(PyExc_RuntimeError, Sf.getErrString());
            return NULL;
        }
    }

    // Check that listOfFramesListed is a list of lists, and that the elements of that are ints.
    for (Py_ssize_t idx = Py_ssize_t(0); idx < PyList_Size(listOfFramesListed); idx++) {
        PyObject* item = PyList_GET_ITEM(listOfFramesListed, idx);
        if(!PyList_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All elements of framelist must be lists themselves");
            return NULL;
        }
        
        // Have been surprised by encountering int overflow here, 65k photons per pixel requires either massive data
        // rates or loooots of pooling. I should do smarter checking but this is a short term solution.
        if (PyList_Size(item) > 10000) {
            PyErr_WarnEx(PyExc_RuntimeWarning, "Pooling a large number of frames! May cause uint16 overflow.",Py_ssize_t(1));
        }

        // check that elements of this list are ints. If so, pass them along to be used by the siffreader functionality. 
        for(Py_ssize_t subidx = Py_ssize_t(0); subidx < PyList_Size(item); subidx++) {
            PyObject* element = PyList_GET_ITEM(item, subidx);
            if(!PyLong_Check(element)) {
                PyErr_SetString(PyExc_TypeError, "All elements of sublists in framelist must be of type int.");
                return NULL;
            }

            // if this isn't in the registration dict, shift by (0,0)
            if (!PyDict_Contains(registrationDict, element)) {
                PyDict_SetItem(registrationDict, element, // steals the reference to the value
                    PyTuple_Pack(Py_ssize_t(2), // steals references, makes life easier
                        PyLong_FromLong(0),
                        PyLong_FromLong(0)
                    )
                );
            }
        }
    }

    try{
        if (discard_bins) {
            if (!(PyLong_Check(discard_bins) || PyFloat_Check(discard_bins))) {
                return Sf.flimMap(FLIMParams, listOfFramesListed, conf_measure, registrationDict);
            }
            else {
                uint64_t terminalBin = int(PyLong_AsLongLong(discard_bins));
                PyErr_SetString(PyExc_NotImplementedError, "Discarding photon counts not yet implemented.");
                return NULL;
                //
                // TODO: IMPLEMENT
            }
        }
        else{
            return Sf.flimMap(FLIMParams, listOfFramesListed, conf_measure, registrationDict);
        }
    }
    catch(...) {
        PyErr_SetString(PyExc_RuntimeError, Sf.getErrString());
        return NULL;
    }
}

static PyArrayObject* siffreader_get_histogram(PyObject* self, PyObject *args, PyObject* kw) {
    PyObject* frames = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_histogram", const_cast<char**>(GET_HISTOGRAM_KEYWORDS), &PyList_Type, &frames)) {
        return NULL;
    }

    try{
        if(!frames) {
            return Sf.getHistogram();
        }
        else {
            uint64_t framesArray[PyList_Size(frames)];

            // Check that they're ints
            for(Py_ssize_t idx = Py_ssize_t(0); idx< PyList_Size(frames); idx++) {
                // these references are BORROWED, so we don't have to worry about INCREF or DECREF
                PyObject* item = PyList_GET_ITEM(frames, idx);
                if(!PyLong_Check(item)) {
                    PyErr_SetString(PyExc_TypeError, "All elements of frame list must be ints");
                    return NULL;
                }
                framesArray[idx] = (uint64_t) PyLong_AsUnsignedLongLong(item);
            }

            uint64_t framesN = PyList_Size(frames);
            return Sf.getHistogram(framesArray, framesN);
        }
    }
    catch(...) {
        PyErr_SetString(PyExc_RuntimeError, Sf.getErrString());
        return NULL;
    }
}

static PyObject* siffreader_suppress_warnings(PyObject* self, PyObject *args){
    // Suppresses warnings

    if(!PyArg_ParseTuple(args, ":suppress_warnings")) return NULL;
    // if there are any args, it shouldn't go through
    
    Sf.suppressWarnings(true);
    Py_RETURN_NONE;
}

static PyObject* siffreader_report_warnings(PyObject* self, PyObject *args){
    // Suppresses warnings
    if(!PyArg_ParseTuple(args, ":report_warnings")) return NULL;
    // if there are any args, it shouldn't go through
    
    Sf.suppressWarnings(false);

    Py_RETURN_NONE;
}

static PyObject* siffreader_num_frames(PyObject* self, PyObject *args){
    // Returns number of frames in file

    if(!PyArg_ParseTuple(args, ":num_frames")) return NULL;
    // if there are any args, it shouldn't go through
    
    uint64_t ret_val = Sf.numFrames();

    if (ret_val < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unopened file");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(ret_val);
}


static PyObject* siffreader_debug(PyObject* self, PyObject *args){
    // Returns number of frames in file

    if(!PyArg_ParseTuple(args, ":debug")) return NULL;
    // if there are any args, it shouldn't go through
    
    Sf.setDebug(true);
    Py_RETURN_NONE;
}

static PyMethodDef SiffreaderMethods[] = {
// Array of the methods and corresponding docstrings
        {"open", siffreader_open, METH_VARARGS,OPEN_DOCSTRING},
        {"close", siffreader_close, METH_VARARGS, CLOSE_DOCSTRING},
        {"get_file_header", siffreader_get_file_header, METH_VARARGS, GET_FILE_HEADER_DOCSTRING},
        // This needs to be cast to a PyCFunction, which by definition has only two arguments unlike our function
        {"get_frames", (PyCFunction) siffreader_get_frames, METH_VARARGS|METH_KEYWORDS, GET_FRAMES_DOCSTRING},
        {"get_frame_metadata", (PyCFunction) siffreader_get_frame_metadata, METH_VARARGS|METH_KEYWORDS, GET_METADATA_DOCSTRING},
        {"pool_frames", (PyCFunction) siffreader_pool_frames, METH_VARARGS|METH_KEYWORDS, POOL_FRAMES_DOCSTRING},
        {"flim_map", (PyCFunction) siffreader_flim_map, METH_VARARGS|METH_KEYWORDS, FLIM_MAP_DOCSTRING},
        {"get_histogram", (PyCFunction) siffreader_get_histogram, METH_VARARGS|METH_KEYWORDS, GET_HISTOGRAM_DOCSTRING},
        {"suppress_warnings", siffreader_suppress_warnings, METH_VARARGS, SUPPRESS_WARNINGS_DOCSTRING},
        {"report_warnings", siffreader_report_warnings, METH_VARARGS, REPORT_WARNINGS_DOCSTRING},
        {"num_frames", siffreader_num_frames, METH_VARARGS, NUM_FRAMES_DOCSTRING},
        {"debug", siffreader_debug, METH_VARARGS, DEBUG_DOCSTRING},
        {NULL, NULL, 0, NULL}        /* Sentinel */

};

static struct PyModuleDef siffreadermodule = {
// Defines the module, created during initialization function below
    PyModuleDef_HEAD_INIT, // must be the first property
    "siffreader",   /* name of module, this is what you type for "import XX" */
    PyDoc_STR(MODULE_DOC), /* module documentation, may be NULL*/ 
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SiffreaderMethods // PyMethodDef array of methods for the module
}; // OTHER OPTIONAL ELEMENTS OF ARRAY MIGHT BE ADDED. SOME USEFUL ONES INCLUDE:
// CLEAR (for Python's garbage collector)
// FREE (for deallocation)


PyMODINIT_FUNC PyInit_siffreader(void) {
// THE  ONLY NONSTATIC OBJECT
// executes on initialization
    import_array(); // we use NumPy in here, so I have to run this before creating the module
    return PyModule_Create(&siffreadermodule);
}