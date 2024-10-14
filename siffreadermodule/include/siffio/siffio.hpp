/*

Defines a Python class siffio which
handles file I/O for each file type. This
cleans up SiffPy pretty dramatically relative
to earlier versions:

1) Allows multiple files to be opened from one
interpreter

2) Allows each SiffReader object from SiffPy to
operate independently

All in the header file, which means every file that
includes this will compile a separate static SiffIO
class (so manipulations to class variables won't communicate
well across any other extension module). This seems
unlikely to happen, but my plan is to come back and implement
this with `PyCapsule`s at some point to resolve that.

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

#ifndef NPY_NO_DEPRECATED_API
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>

// Shared across all SiffIOs
extern bool debug_SIFFIO;

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

    //PyObject* frameDataList;
    SiffReader* siffreader; // each SiffIO has a C++ siffreader
    // status string causes segfault when initialized to something small??
    // getting allocated on stack?? But why would that be a problem????
    std::string status;
    std::iostream debug_log; // only defined when debug called. TODO
} SiffIO;


static PyMemberDef siffio_members[] = {
    //{Attribute name, attribute type, location in struct, flags, docstring}
    //{"frame_data", T_OBJECT_EX, offsetof(SiffIO, frameDataList), 0, "List of frame data."},
    {NULL},
};

#define KWARG_CAST(x) const_cast<char **>(x) // so fucking dumb

/**
 * I repeat this so often that I realized I should just make an inlined func.
*/
inline void populate_frame_list_if_null(PyObject** frames_list, SiffReader *siffreader){
    if (*frames_list != NULL) return;
    *frames_list = PyList_New(siffreader->numFrames());
    for (uint64_t frame_idx = 0; frame_idx < siffreader->numFrames(); frame_idx++) {
        PyList_SET_ITEM(*frames_list, frame_idx, PyLong_FromUnsignedLongLong(frame_idx));
    }
}


/*
 * Called on creation of an instance of a siffio.

 * Creates a blank C++ siffreader
*/

bool debug_SIFFIO = false;

// Allocate the new SiffReader++
static PyObject* siffio_new(PyTypeObject* type, PyObject* args, PyObject* kwargs){
    SiffIO *self;
    self = (SiffIO *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->siffreader = new SiffReader();
        //self->frameDataList = PyList_New(0);
        //self->status = std::string("");
    }
    return (PyObject *) self;
};

// Can init with a filename to open that way.
static int siffio_init(SiffIO* self, PyObject* args){
    
    const char* filename = NULL;

    if(!PyArg_ParseTuple(args, "|s:SiffIO", &filename)){
        return -1;
    }

    // No filename provided, nothing to do.
    if(filename == NULL) return 0;

    int retval = self->siffreader->openFile(filename);
    if (retval < 0) {
        PyErr_SetString(
            PyExc_ValueError,
            (std::string(
            "Invalid file or filename provided.\n"
            "Additional info :\n\n"
            ) + 
            std::string(self->siffreader->getErrString())
            ).c_str()
        );
        return -1;
    }
    return 0;
};

static void siffio_dealloc(SiffIO *self){
    // Called on deallocation -- close the siffreader object.
    self->siffreader->closeFile();
    delete self->siffreader;
    //Py_CLEAR(self->frameDataList);

    Py_TYPE(self)->tp_free((PyObject *) self);
};

static PyObject* siffio_repr(SiffIO *self){
    PyObject* retstring = PyUnicode_FromString(
        "A SiffIO object. Not meant to be touched "
        "outside of a SiffReader. Maybe this will change."
        " - SCT 06/29/2022"
    );
    return retstring;
};

static int siffio_clear(SiffIO *self){
    // If I ever update to the new module spec...
    //Py_CLEAR(self->frameDataList);
    return 0;
};

/******
 * 
 *  FILE IO
 * 
 * */

// Opens a file in the SiffReader++ object.

static PyObject* siffio_open(SiffIO* self, PyObject* args){
    
    const char* filename; 
    
    if (!PyArg_ParseTuple(args, "s:open", &filename)){
        return NULL;
    }
    
    int ret = self->siffreader->openFile(filename);

    if (ret < 0) {
        if (ret == -2) {
            PyErr_WarnEx(
                PyExc_RuntimeWarning,
                self->siffreader->getErrString(),
                Py_ssize_t(1)
            );
            Py_RETURN_NONE;
        }
        else PyErr_SetString(
            PyExc_FileNotFoundError,
            self->siffreader->getErrString()
            );
        return NULL;
    }
    Py_RETURN_NONE;
};

// Closes any open SiffReader++ object.
static PyObject* siffio_close(SiffIO* self){
    try{
        self->siffreader->closeFile();
    }
    catch(std::exception &e) {
        PyErr_SetString(
            PyExc_RuntimeError,
            e.what()
        );
        return NULL;
    }
    Py_RETURN_NONE;
};

static PyObject* siffio_get_file_header(SiffIO *self) {
    // simple: returns the file header data
    // Error handling is in the siffreader method this time.
    return self->siffreader->readFixedData();
};

/*
Returns the number of frames in the file from counting
IFDs

@param self: SiffIO object
*/
static PyObject* siffio_num_frames(SiffIO* self){
    
    uint64_t ret_val = self->siffreader->numFrames();

    if (ret_val < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unopened file");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(ret_val);
}

/**************
 * 
 *  FRAMES METHODS
 * 
 * 
 * *****/

/*
Checks a list with the data for a set of frames
to be analyzed. Makes sure they're siffreader-compatible.
Then loads the correct list into a C++ array.

@param frameList: A list object from Python to be inspected for
validity.

@param framesArray: A pointer to a C++ array to be filled with the
frame numbers from the frameList.

@param framesArrayN: The length of the framesArray

@param nTotalFrames: The total number of frames in the file.

@return 0 on success, -1 on failure. Error is written to the
Python error state.
*/
int check_framelist(
    PyObject* frameList,
    uint64_t* framesArray,
    const size_t framesArrayN,
    const uint64_t nTotalFrames
    ){
    // Checks that the frame list is valid
    // and converts it to a C++ array if the framesArray pointer passed
    // is not NULL.
    Py_ssize_t framesN = PyList_Size(frameList);
    if (framesArrayN != ((size_t)framesN)) {
        PyErr_SetString(PyExc_RuntimeError, "Failure to allocate frames array.");
        return -1;
    }
    for(Py_ssize_t idx = Py_ssize_t(0); idx < framesN; idx++) {
        PyObject* item = PyList_GET_ITEM(frameList, idx);
        if(!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All elements of frame list must be ints");
            return -1;
        }
        uint64_t frameNum = PyLong_AsUnsignedLongLong(item);
        if (frameNum >= nTotalFrames) {
            PyErr_SetString(PyExc_ValueError, "Frame number provided is greater than indices of frames.");
            return -1;
        }
        
        if(framesArrayN != 0) framesArray[idx] = frameNum;
    }
    return 0;
};

/**
 * Checks a registration dictionary for validity. If a requested
 * frame is not in the dictionary, it is added with a registration
 * of (0,0). If a requested frame is in the dictionary but the
 * registration is not a tuple of ints, it is cast to a tuple of ints.

 * @param registrationDict: A dictionary of frame numbers to tuples of
 * integers.

 * @param frames_list: A list of frame numbers to be checked against
 * the registration dictionary.

 * @return 0 on success, -1 on failure. Error is written to the
 * Python error state.
*/
int check_registration(PyObject* registrationDict, PyObject* frames_list){
    
    for(Py_ssize_t idx = Py_ssize_t(0); idx < PyList_Size(frames_list); idx++) {
        // Stick to the PyList objects so we don't have to make and destroy PyLongs            
        PyObject* item = PyList_GET_ITEM(frames_list, idx);

        // if this isn't in the registration dict, shift by (0,0)
        if (!PyDict_Contains(registrationDict, item)) {
            PyDict_SetItem(registrationDict, item, // steals the reference to the value
                PyTuple_Pack(Py_ssize_t(2), // steals references, makes life easier
                    PyLong_FromLong(0),
                    PyLong_FromLong(0)
                )
            );
        }
        
        // Now typecheck the registration dict item
        // If it's not a tuple of PyLongs, replace it
        // with a tuple of PyLongs made by attemping to cast
        // the elements to one and replacing the tuple.
        
        try {
            PyObject* shiftTuple = PyDict_GetItem(registrationDict, item);
            if (!PyTuple_Check(shiftTuple)) {
                PyErr_SetString(
                    PyExc_TypeError,
                    (
                        std::string("Registration dictionary element for frame ") + 
                        std::to_string(PyLong_AsLongLong(item)) +
                        std::string(" is not a tuple.")
                    ).c_str()
                );
                return -1;
            }
            Py_ssize_t tupLen = PyTuple_Size(shiftTuple);
            for(Py_ssize_t tupIdx = 0; tupIdx < tupLen; tupIdx++) {
                PyObject* shiftValue = PyTuple_GetItem(shiftTuple, tupIdx);
                if(!PyLong_Check(shiftValue)) { // if it's not okay, try to cast it
                    PyObject* result = PyObject_CallMethod(shiftValue, "__int__", NULL);
                    if (result == NULL) {
                        PyErr_SetString(
                            PyExc_TypeError,
                            (
                                std::string("Registration dictionary element for frame ") + 
                                std::to_string(PyLong_AsLongLong(item)) +
                                std::string(" cannot be cast to type int.")
                            ).c_str()
                        );
                        return -1;
                    }
                    PyTuple_SetItem(shiftTuple, tupIdx, result);
                }
            }
        }
        catch (...) {
            PyErr_SetString(PyExc_RuntimeError,
                (std::string("Failure to access registration dictionary element for frame ") +
                std::to_string(PyLong_AsLongLong(item))).c_str()
            );
            return -1;
        }
    }
    return 0;
};

/**
 * Returns a list of frames or a numpy array from the opened file

 * @param self: SiffIO object

 * @param args: A list of arguments. Not actually inspected. Only takes kwargs

 * @param kw: A dict of keyword arguments. Looks for "frames", "registration", and "as_array",
 * of type List, Dict, and Bool respectively.
*/
static PyObject* siffio_get_frames(SiffIO *self, PyObject *args, PyObject* kw) {
    // 

    static const char* GET_FRAMES_KWARGS[] = {
        "frames", "registration", "as_array", NULL};

    bool make_array = true;

    PyObject *frames_list = NULL;
    PyObject* registrationDict = NULL;
    PyObject* as_array = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!OO:get_frames", 
        KWARG_CAST(GET_FRAMES_KWARGS),
        &PyList_Type, &frames_list,
        &registrationDict,
        &as_array
        )
    ) {
        //PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    if(as_array != NULL){
        make_array = true;
        if (PyBool_Check(as_array)) {
            make_array = as_array == Py_True;
        }
    }

    uint64_t framesN = PyList_Size(frames_list);
    uint64_t* framesArray = new uint64_t[framesN];

    if (check_framelist(frames_list, framesArray, framesN, self->siffreader->numFrames())<0){
        delete[] framesArray;
        return NULL;
    }

    if (registrationDict == NULL) {
        registrationDict = Py_None;
    }
    bool registrationDictProvided = registrationDict != Py_None;
    if(registrationDict == Py_None) {
        registrationDict = PyDict_New();
    }
    
    if (!PyDict_Check(registrationDict)) {
        PyErr_SetString(PyExc_TypeError, "Registration dictionary must be a dictionary or None");
        return NULL;
    }

    if (check_registration(registrationDict, frames_list) < 0){
        delete[] framesArray;
        return NULL;
    }

    try{
        if (make_array) {

            if(!self->siffreader->dimensionsConsistent(framesArray, framesN)){
                PyErr_SetString(PyExc_TypeError, "Dimensions of requested frames are not consistent");
                delete[] framesArray;
                return NULL;
            }
            PyObject* retArray = (PyObject*) self->siffreader->retrieveFramesAsArray(
                framesArray, framesN, registrationDict
            );
            if (!registrationDictProvided) Py_DECREF(registrationDict);
            delete[] framesArray;
            return retArray;
        }
        if (!registrationDictProvided) Py_DECREF(registrationDict);
        
        PyObject* retList = self->siffreader->retrieveFrames(framesArray, framesN, registrationDict);
        delete[] framesArray;
        return retList;
    }
    catch(...) {
        delete[] framesArray;
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
}

static PyArrayObject* siffio_get_experiment_timestamps(SiffIO* self, PyObject *args, PyObject* kw){
    static const char* GET_EXPERIMENT_TIME_KEYWORDS[] = {"frames", NULL};
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O:get_experiment_timestamps", 
            KWARG_CAST(GET_EXPERIMENT_TIME_KEYWORDS), 
            &frames_list
            )
        ) {
        //PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    // TODO: Test this and make sure it actually works -- for now
    // I'm just always passing in lists anyway..
    if(PyArray_Check(frames_list)){
        frames_list = PyArray_ToList((PyArrayObject*) frames_list);
    };

    const uint64_t framesN = PyList_Size(frames_list);
    uint64_t* framesArray = new uint64_t[framesN];
    if(
        check_framelist(
            frames_list, framesArray, framesN, self->siffreader->numFrames()
        )<0
    ){  
        delete[] framesArray;
        // PyErr_SetString called in check_framelist
        return NULL;
    }

    try{
        PyArrayObject* timestamps = self->siffreader->getExperimentTimestamps(framesArray, framesN);
        delete[] framesArray;
        return timestamps;
    }
    catch(...){
        delete[] framesArray;
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
};

static PyArrayObject* siffio_get_epoch_laser(SiffIO *self, PyObject *args, PyObject *kw){
    // Returns the frame timestamps as epoch timestamps from the laser clock
    
    static const char* GET_EPOCH_LASER_KEYWORDS[] = {"frames", NULL};
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_epoch_timestamps_laser", 
            KWARG_CAST(GET_EPOCH_LASER_KEYWORDS), 
            &PyList_Type, &frames_list)
        ) {
        //PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    const uint64_t framesN = PyList_Size(frames_list);
    uint64_t* framesArray = new uint64_t[framesN];

    DEBUG(
        self->siffreader->toDebugLog("In get_epoch_laser");
        self->siffreader->toDebugLog("About to check framelist");
    )

    if(
        check_framelist(
            frames_list, framesArray, framesN, self->siffreader->numFrames()
        )<0
    ){  
        delete[] framesArray;
        // PyErr_SetString called in check_framelist
        return NULL;
    }

    try{
        PyArrayObject* timestamps = self->siffreader->getEpochTimestampsLaser(framesArray, framesN);
        delete[] framesArray;
        return timestamps;
    }
    catch(...){
        delete[] framesArray;
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
};

static PyArrayObject* siffio_get_epoch_system(SiffIO *self, PyObject *args, PyObject* kw){
    // Returns the frame timestamps as the most recent system clock time call
    
    static const char* GET_EPOCH_LASER_KEYWORDS[] = {"frames", NULL};
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_epoch_timestamps_system", 
            KWARG_CAST(GET_EPOCH_LASER_KEYWORDS), 
            &PyList_Type, &frames_list)
        ) {
        //PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    const uint64_t framesN = PyList_Size(frames_list);
    uint64_t* framesArray = new uint64_t[framesN];
    if(
        check_framelist(
            frames_list, framesArray, framesN, self->siffreader->numFrames()
        )<0
    ){  
        delete[] framesArray;
        // PyErr_SetString called in check_framelist
        return NULL;
    }

    try{
        PyArrayObject* timestamps = self->siffreader->getEpochTimestampsSystem(framesArray, framesN);
        delete[] framesArray;
        return timestamps;
    }
    catch(...){
        delete[] framesArray;
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
};

static PyArrayObject* siffio_epoch_both(SiffIO* self, PyObject *args, PyObject* kw) {
       // Returns the frame timestamps as the most recent system clock time call
    
    static const char* GET_EPOCH_BOTH_KEYWORDS[] = {"frames", NULL};
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_epoch_both", 
            KWARG_CAST(GET_EPOCH_BOTH_KEYWORDS), 
            &PyList_Type, &frames_list)
        ) {
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    const uint64_t framesN = PyList_Size(frames_list);
    uint64_t* framesArray = new uint64_t[framesN];

    DEBUG(
        self->siffreader->toDebugLog("In get_epoch_both");
        self->siffreader->toDebugLog("About to check framelist");
    )

    if(
        check_framelist(
            frames_list, framesArray, framesN, self->siffreader->numFrames()
        )<0
    ){  
        delete[] framesArray;
        // PyErr_SetString called in check_framelist
        return NULL;
    }

    try{
        PyArrayObject* timestamps = self->siffreader->getEpochTimestampsBoth(framesArray, framesN);
        delete[] framesArray;
        return timestamps;
    }
    catch(...){
        delete[] framesArray;
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    } 
};

static PyObject* siffio_get_frame_metadata(SiffIO *self, PyObject *args, PyObject* kw) {
    // Gets the meta data for the frames in kw, or for all the frames.  

    static const char* GET_FRAMES_METADATA_KEYWORDS[] = {"frames", NULL};
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_frames_metadata", 
            KWARG_CAST(GET_FRAMES_METADATA_KEYWORDS), 
            &PyList_Type, &frames_list)
        ) {
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    const uint64_t framesN = PyList_Size(frames_list);
    uint64_t* framesArray = new uint64_t[framesN];
    DEBUG(self->siffreader->toDebugLog("In `get_frame_metadata`, about to check framelist");)
    if(
        check_framelist(
            frames_list, framesArray, framesN, self->siffreader->numFrames()
        )<0
    ){  
        delete[] framesArray;
        // PyErr_SetString called in check_framelist
        return NULL;
    }
    try{
        DEBUG(self->siffreader->toDebugLog("calling readMetaData");)
        PyObject* metadata = self->siffreader->readMetaData(framesArray, framesN);
        delete[] framesArray;
        return metadata;
    }
    catch(...){
        delete[] framesArray;
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
};

static PyObject* siffio_get_appended_text(SiffIO *self, PyObject *args, PyObject* kw){
    static const char* GET_APPENDED_TEXT_KEYWORDS[] = {"frames", NULL};
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_appended_text", 
            KWARG_CAST(GET_APPENDED_TEXT_KEYWORDS), 
            &PyList_Type, &frames_list)
        ) {
        //PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    const uint64_t framesN = PyList_Size(frames_list);
    uint64_t* framesArray = new uint64_t[framesN];
    if(
        check_framelist(
            frames_list, framesArray, framesN, self->siffreader->numFrames()
        )<0
    ){  
        delete[] framesArray;
        // PyErr_SetString called in check_framelist
        return NULL;
    }

    try{
        PyObject* appendedText = self->siffreader->getAppendedText(framesArray, framesN);
        delete[] framesArray;
        return appendedText;
    }
    catch(...){
        delete[] framesArray;
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
};

static PyObject * siffio_pool_frames(SiffIO* self, PyObject *args, PyObject* kw) {
    // Pools frames together, returns list of pooled frames.
    static const char* POOL_FRAMES_KEYWORDS[] = {"pool_lists", "flim", "registration", NULL};

    PyObject* listOfFramesListed = NULL;
    bool flim = false;
    PyObject* registrationDict = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "O!|$O!pO!:pool_frames", 
        KWARG_CAST(POOL_FRAMES_KEYWORDS),
        &PyList_Type, &listOfFramesListed,
        &flim,
        &PyDict_Type, &registrationDict
        )
    ) {
        return NULL;
    }

    bool need_to_decref_regdict = false;
    // defaults to 0's
    if(!registrationDict) {
        registrationDict = PyDict_New();
        need_to_decref_regdict = true;
    }
    if(!PyObject_TypeCheck(registrationDict, &PyDict_Type)) {
        registrationDict = PyDict_New();
        need_to_decref_regdict = true;
    }

    PyErr_SetString(PyExc_NotImplementedError, "Pooling is not yet implemented. Some bug...");
    if(need_to_decref_regdict) Py_DECREF(registrationDict);
    return NULL;
};

/*******
 * 
 * 
 *  FLIM METHODS
 * 
 * 
 * *////


static PyObject* siffio_flim_map(SiffIO* self, PyObject* args, PyObject* kw) {
    // Takes in a FLIMParams object and frames desired and returns a lifetime map, an intensity distribution,
    // and a chi-squared value for every pixel.

    static const char* FLIM_MAP_KEYWORDS[] = {"params", "frames", "confidence_metric", "registration", NULL};

    PyObject* FLIMParams = NULL;
    PyObject* listOfFrames = NULL;
    char* conf_measure;
    Py_ssize_t conf_measure_length = Py_ssize_t(0);
    PyObject* registrationDict = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "O|$O!s#O:flim_map", 
        KWARG_CAST(FLIM_MAP_KEYWORDS),
        &FLIMParams,
        &PyList_Type, &listOfFrames,
        &conf_measure, &conf_measure_length,
        &registrationDict
        )) {
        return NULL;
    }

    populate_frame_list_if_null(&listOfFrames, self->siffreader);

    if(!conf_measure_length) conf_measure = (char*) "chi_sq";
    if (registrationDict == NULL) registrationDict = Py_None;
    // defaults to 0's
    bool need_to_decref_regdict = false;
    if(registrationDict == Py_None) {
        registrationDict = PyDict_New();
        need_to_decref_regdict = true;
    }
    
    if (!PyDict_Check(registrationDict)) {
        PyErr_SetString(PyExc_TypeError, "Registration dictionary must be a dictionary or None");
        return NULL;
    }

    // Check that FLIMParams is of type siffpy.core.flim.flimparams.FLIMParams
    if (strcmp(FLIMParams->ob_type->tp_name,"FLIMParams")){
        if (need_to_decref_regdict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_TypeError, 
            strcat((char*)"Expected params to be of type FLIMParams. Instead is type: ",
                FLIMParams->ob_type->tp_name
            )
        );
        return NULL;
    }

    // check that conf_measure is one of the permitted values
    if (strcmp(conf_measure, "log_p") && strcmp(conf_measure,"chi_sq") && strcmp(conf_measure,"None")) {
        if (need_to_decref_regdict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_TypeError, 
            strcat((char*) "Expected confidence_measure to be one of 'log_p', 'chi_sq', 'None'. Instead is type: ",
                conf_measure
            )
        );
        return NULL;
    }

    uint64_t* frames = new uint64_t[PyList_Size(listOfFrames)];
    check_framelist(listOfFrames, frames, PyList_Size(listOfFrames), self->siffreader->numFrames());
    check_registration(registrationDict, listOfFrames);

    try{
        
        if (!self->siffreader->dimensionsConsistent(frames, PyList_Size(listOfFrames))) {
            if (need_to_decref_regdict) Py_DECREF(registrationDict);
            PyErr_SetString(PyExc_TypeError, "Dimensions of requested frames are not consistent");
            return NULL;
        }

        PyObject* flimTuple(
            self->siffreader->flimTuple(
                FLIMParams,
                frames,
                PyList_Size(listOfFrames),
                conf_measure,
                registrationDict
            )
        );
        if (need_to_decref_regdict) Py_DECREF(registrationDict);
        return flimTuple;
    }
    catch(...) {
        if (need_to_decref_regdict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
}


/***
 * 
 * ROI METHODS
 * 
 * */

/**
 * @brief Returns the summed photon counts within the provided ROI
 * for multiple masks. Expects either a list of masks (whether they're 2d or 3d)
 * or a singular numpy array.
 * 
 * @returns A numpy array of the summed photon counts within the ROI,
 * with shape (`n_masks`, `n_frames`)
*/
static PyArrayObject* siffio_sum_rois(SiffIO* self, PyObject* args, PyObject*kw){

   static const char* SUM_ROIS_KEYWORDS[] = {"masks", "frames", "registration", NULL};

    PyArrayObject* masks;
    PyObject *frames_list = NULL;
    PyObject* registrationDict = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$O!O:sum_rois",
        KWARG_CAST(SUM_ROIS_KEYWORDS),
        &masks,
        &PyList_Type, &frames_list,
        &registrationDict
        )
     ) {
        return NULL;
    }

    if (!registrationDict) {
        registrationDict = Py_None;
    }

    if (PyList_Check(masks)){
        PyErr_SetString(
            PyExc_TypeError,
            "List of masks provided -- not yet implemented in `SiffIO`."
            " Please, for now, use the `SiffReader` `Python` class's"
            " `sum_mask` or `_sum_masks` because it converts the list of"
            " masks to a numpy array efficiently. TODO: Implement conversion"
            " from `List` to `numpy.ndarray` in `SiffIO` C++ code too."
        );
        return NULL;
        // Convert to numpy array.
        // masks = (PyArrayObject*) PyArray_FromAny(
        //     (PyObject*) masks,
        //     PyArray_DescrFromType(NPY_BOOL),
        //     0,
        //     0,
        //     NPY_ARRAY_CARRAY,
        //     NULL
        // );
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    if (registrationDict == NULL) registrationDict = Py_None;

    bool need_to_decref_dict = false;
    if(registrationDict == Py_None) {
        registrationDict = PyDict_New();
        need_to_decref_dict = true;
    }
    if (!PyDict_Check(registrationDict)) {
        PyErr_SetString(PyExc_TypeError, "Registration dictionary must be a dictionary or None");
        return NULL;
    }

    // Check that all elements of the frame list and registration dictionary are valid.
    uint64_t* framesArray = new uint64_t[PyList_Size(frames_list)];

    if(check_framelist(frames_list, framesArray, PyList_Size(frames_list), self->siffreader->numFrames()) < 0){
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }
    if (check_registration(registrationDict, frames_list) < 0) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }

    uint64_t framesN = PyList_Size(frames_list);

    // Okay enough argument checking, we can call the siffreader function
    try{
        PyArrayObject *returnedMask = self->siffreader->sumMasks(
            framesArray,
            framesN,
            (PyArrayObject*) masks,
            registrationDict
        );
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return (PyArrayObject*) PyArray_Transpose(returnedMask, NULL);
        //return returnedMask;

    }
    catch(...) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        delete[] framesArray;
        return NULL;
    }
}; 


/**
 *  
 * @brief Returns the summed photon counts within the provided ROI.
 * Expects a singular ROI mask, whether it's 2d or 3d.
 * 
 * @param self : SiffIO object
 * 
 * @param args : (`mask`, *)
 * 
 * @param kw : `{'frames' : List[int], 'registration' : Dict[int, Tuple[int, int], **}`
 * 
 * @return : A numpy array of the summed photon counts within the ROI,
 * with shape (`n_frames`,)
*/
static PyArrayObject* siffio_sum_roi(SiffIO* self, PyObject* args, PyObject*kw){
   static const char* SUM_ROIS_KEYWORDS[] = {"mask", "frames", "registration", NULL};

    PyArrayObject *mask;
    PyObject *frames_list = NULL;
    PyObject *registrationDict = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$O!O:sum_roi",
        KWARG_CAST(SUM_ROIS_KEYWORDS),
        &mask,
        &PyList_Type, &frames_list,
        &registrationDict
        )
     ) {
            return NULL;
    }

    if (!registrationDict) {
        registrationDict = Py_None;
    }

    if (PyList_Check(mask)) {
        PyErr_WarnEx(
            PyExc_RuntimeWarning,
            "Mask provided is a list -- presumed to correspond to multiple masks"
            " -- will be passed to `sumMasks` instead of `sumMask`.",
            Py_ssize_t(1)
        );
        PyDict_SetItemString(kw, "masks", (PyObject*) mask);
        PyDict_DelItemString(kw, "mask");
        return siffio_sum_rois(self, args, kw);
    }

    if (PyArray_NDIM(mask) != 2 && PyArray_NDIM(mask) != 3) {
        PyErr_SetString(PyExc_ValueError, "Mask must be 2D or 3D");
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);

    if (registrationDict == NULL) registrationDict = Py_None;

    bool need_to_decref_dict = false;
    if(registrationDict == Py_None) {
        registrationDict = PyDict_New();
        need_to_decref_dict = true;
    }
    if (!PyDict_Check(registrationDict)) {
        PyErr_SetString(PyExc_TypeError, "Registration dictionary must be a dictionary or None");
        return NULL;
    }

    // Check that all elements of the frame list and registration dictionary are valid.
    uint64_t* framesArray = new uint64_t[PyList_Size(frames_list)];

    if(check_framelist(frames_list, framesArray, PyList_Size(frames_list), self->siffreader->numFrames()) < 0){
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }
    if (check_registration(registrationDict, frames_list) < 0) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }

    uint64_t framesN = PyList_Size(frames_list);

    // Okay enough argument checking, we can call the siffreader function
    try{
        PyArrayObject *returnedMask = self->siffreader->sumMask(
            framesArray,
            framesN,
            (PyArrayObject*) mask,
            registrationDict
        );
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return returnedMask;

    }
    catch(...) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        delete[] framesArray;
        return NULL;
    }
};

/**
 * @brief Multi-ROI version of `siffio_sum_roi_flim`.
 * Expects a list of masks, whether they're 2d or 3d, or a numpy array
 * for which this code will interpret the slowest dimension as the mask
 * dimension.
 * 
 * @returns A numpy array of the summed photon counts within each ROI,
 * with shape (`n_masks`, `n_frames`)
*/
static PyArrayObject *siffio_sum_rois_flim(SiffIO *self, PyObject *args, PyObject *kw){
    static const char* SUM_ROI_FLIM_KEYWORDS[] = {"masks", "params", "frames", "registration", NULL};

    PyArrayObject* masks;
    PyObject* FLIMParams;
    PyObject *frames_list = NULL;
    PyObject* registrationDict = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|$O!O:sum_rois_flim", 
        KWARG_CAST(SUM_ROI_FLIM_KEYWORDS),
        &masks,
        &FLIMParams,
        &PyList_Type, &frames_list,
        &registrationDict
        )
     ) {
            //PyErr_SetString(PyExc_ValueError, "Error parsing input arguments.");
            return NULL;
    }

    if (!registrationDict) {
        registrationDict = Py_None;
    }

    if (PyList_Check(masks)){
        PyErr_SetString(
            PyExc_TypeError,
            "List of masks provided -- not yet implemented in `SiffIO`."
            " Please, for now, use the `SiffReader` `Python` class's"
            " `sum_mask` or `_sum_masks` because it converts the list of"
            " masks to a numpy array efficiently. TODO: Implement conversion"
            " from `List` to `numpy.ndarray` in `SiffIO` C++ code too."
        );
        return NULL;
        // Convert to numpy array.
        // masks = (PyArrayObject*) PyArray_FromAny(
        //     (PyObject*) masks,
        //     PyArray_DescrFromType(NPY_BOOL),
        //     0,
        //     0,
        //     NPY_ARRAY_CARRAY,
        //     NULL
        // );
    }

    if(PyArray_TYPE(masks) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "mask array must be of type bool.");
        return NULL;
    }

    // Check that FLIMParams is of type siffpy.core.flim.flimparams.FLIMParams
    if (strcmp(FLIMParams->ob_type->tp_name,"FLIMParams")){
        PyErr_SetString(PyExc_TypeError, 
            strcat((char*)"Expected params to be of type FLIMParams. Instead is type: ",
                FLIMParams->ob_type->tp_name
            )
        );
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);
    
    if (registrationDict == NULL) registrationDict = Py_None;

    bool need_to_decref_dict = false;
    if(registrationDict == Py_None) {
        registrationDict = PyDict_New();
        need_to_decref_dict = true;
    }
    if (!PyDict_Check(registrationDict)) {
        PyErr_SetString(PyExc_TypeError, "Registration dictionary must be a dictionary or None");
        return NULL;
    }

    DEBUG(
        self->siffreader->toDebugLog("In `siffio_sum_rois_flim`, about to check framelist");
    )

    // Check that all elements of the frame list and registration dictionary are valid.
    uint64_t* framesArray = new uint64_t[PyList_Size(frames_list)];
    
    if(check_framelist(frames_list, framesArray, PyList_Size(frames_list), self->siffreader->numFrames()) < 0){
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }
    if (check_registration(registrationDict, frames_list) < 0) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }   
    uint64_t framesN = PyList_Size(frames_list);

    // Okay enough argument checking, we can call the siffreader function

    try{
        DEBUG(
            self->siffreader->toDebugLog("In `siffio_sum_rois_flim`, about to call sumFLIMMasks");
        )
        PyArrayObject* FLIMMask(
            self->siffreader->sumFLIMMasks(
                framesArray,
                framesN,
                FLIMParams,
                (PyArrayObject*) masks,
                registrationDict
            )
        );
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return (PyArrayObject*) PyArray_Transpose(FLIMMask, NULL);
    }
    catch(...) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        delete[] framesArray;
        return NULL;
    }
};

static PyArrayObject* siffio_sum_roi_flim(SiffIO *self, PyObject *args, PyObject* kw){
    /*
    Returns the summed photon counts within the provided ROI

    Args : 
        mask : numpy array ROI mask, dtype 'bool'
        FLIMParams : a `siffpy.FLIMParams` object
    Kwargs : 
        frames : list[int]
            - If none provided, uses all frames.
        registration : dict
            - If none provided, uses no shift
    */

   static const char* SUM_ROI_FLIM_KEYWORDS[] = {"mask", "params", "frames", "registration", NULL};

    PyArrayObject* mask;
    PyObject* FLIMParams;
    PyObject *frames_list = NULL;
    PyObject* registrationDict = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|$O!O:sum_roi_flim", 
        KWARG_CAST(SUM_ROI_FLIM_KEYWORDS),
        &mask,
        &FLIMParams,
        &PyList_Type, &frames_list,
        &registrationDict
        )
     ) {
            return NULL;
    }

    if (!registrationDict) {
        registrationDict = Py_None;
    }

    if (PyList_Check(mask)) {
        PyErr_WarnEx(
            PyExc_RuntimeWarning,
            "Mask provided is a list -- presumed to correspond to multiple masks"
            " -- will be passed to `sumMasks` instead of `sumMask`.",
            Py_ssize_t(1)
        );
        PyDict_SetItemString(kw, "masks", (PyObject*) mask);
        PyDict_DelItemString(kw, "mask");
        return siffio_sum_rois_flim(self, args, kw);
    }

    if (!PyArray_Check(mask)) {
        PyErr_SetString(PyExc_TypeError, "`mask` must be a `numpy` array");
        return NULL;
    }

    if(PyArray_TYPE(mask) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "`mask` array must be of type `bool`.");
        return NULL;
    }

    // Check that FLIMParams is of type siffpy.core.flim.flimparams.FLIMParams
    if (strcmp(FLIMParams->ob_type->tp_name,"FLIMParams")){
        PyErr_SetString(PyExc_TypeError, 
            strcat((char*)"Expected params to be of type `FLIMParams`. Instead is type: ",
                FLIMParams->ob_type->tp_name
            )
        );
        return NULL;
    }

    populate_frame_list_if_null(&frames_list, self->siffreader);
    
    if (registrationDict == NULL) registrationDict = Py_None;

    bool need_to_decref_dict = false;
    if(registrationDict == Py_None) {
        registrationDict = PyDict_New();
        need_to_decref_dict = true;
    }
    if (!PyDict_Check(registrationDict)) {
        PyErr_SetString(PyExc_TypeError, "Registration dictionary must be a dictionary or None");
        return NULL;
    }

    // Check that all elements of the frame list and registration dictionary are valid.
    uint64_t* framesArray = new uint64_t[PyList_Size(frames_list)];
    if(check_framelist(frames_list, framesArray, PyList_Size(frames_list), self->siffreader->numFrames()) < 0){
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }
    if (check_registration(registrationDict, frames_list) < 0) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return NULL;
    }   
    uint64_t framesN = PyList_Size(frames_list);

    // Okay enough argument checking, we can call the siffreader function

    try{
        PyArrayObject* FLIMMask(
            self->siffreader->sumFLIMMask(
                framesArray,
                framesN,
                FLIMParams,
                (PyArrayObject*) mask,
                registrationDict
            )
        );
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        delete[] framesArray;
        return FLIMMask;
    }
    catch(...) {
        if (need_to_decref_dict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        delete[] framesArray;
        return NULL;
    }
};

/******************
 * 
 * 
 * HISTOGRAM METHODS
 * 
 * 
 * ***/////

 static PyArrayObject* siffio_get_histogram(SiffIO* self, PyObject *args, PyObject* kw) {

    static const char* GET_HISTOGRAM_KEYWORDS[] = {"frames", "mask", NULL};

    PyObject* frames = NULL;
    PyArrayObject* mask = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!O!:get_histogram",
        KWARG_CAST(GET_HISTOGRAM_KEYWORDS),
        &PyList_Type, &frames,
        &PyArray_Type, &mask
        )) {
        return NULL;
    }

    try{
        if (mask) {
            if (PyArray_TYPE(mask) != NPY_BOOL) {
                PyErr_SetString(PyExc_TypeError, "Mask must be of type bool");
                return NULL;
            }

            PyErr_SetString(
                PyExc_NotImplementedError,
                "Masked histograms are not yet implemented."
            );
            return NULL;
        }
        if(!frames) {
            return self->siffreader->getHistogram(NULL, 0);
        }
    }
    catch(...) {
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
    uint64_t* framesArray = new uint64_t[PyList_Size(frames)];
    try{

        check_framelist(frames, framesArray, PyList_Size(frames), self->siffreader->numFrames());

        uint64_t framesN = PyList_Size(frames);
        PyArrayObject* histo;
        if (mask) {
            histo = self->siffreader->getHistogram(mask, framesArray, framesN);
        }
        else{
            histo = self->siffreader->getHistogram(framesArray, framesN);
        }
        delete[] framesArray;
        return histo;
    }
    catch(...) {
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        delete[] framesArray;
        return NULL;
    }
}


/*
static PyObject* siffio_open(SiffIO* self, PyObject* args);
static PyObject* siffio_close(SiffIO* self);
static PyObject* siffio_get_file_header(SiffIO* self);
static PyObject* siffio_num_frames(SiffIO* self);

// Time methods

static PyArrayObject* siffio_get_experiment_timestamps(SiffIO* self, PyObject* args, PyObject* kwargs);
static PyArrayObject* siffio_get_epoch_laser(SiffIO* self, PyObject* args, PyObject* kwargs);
static PyArrayObject* siffio_get_epoch_system(SiffIO* self, PyObject* args, PyObject* kwargs);
static PyArrayObject* siffio_epoch_both(SiffIO* self, PyObject* args, PyObject* kwargs);

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
*/

/**
 * 
 * The `SiffIO` type's methods, stored as an array of `PyMethodDef` structs.
*/
static PyMethodDef siffio_methods[] = {
    // {Method name, (PyCFunction) Function, flags, doc}
    {"open", (PyCFunction) siffio_open, METH_VARARGS, siffio_open_doc},
    {"close", (PyCFunction) siffio_close, METH_NOARGS, siffio_close_doc},
    {"get_file_header", (PyCFunction) siffio_get_file_header, METH_NOARGS, siffio_get_file_header_doc},
    {"num_frames", (PyCFunction) siffio_num_frames, METH_NOARGS, siffio_num_frames_doc},

    // Time methods
    {"get_experiment_timestamps", (PyCFunction) siffio_get_experiment_timestamps, METH_VARARGS|METH_KEYWORDS, siffio_get_experiment_timestamps_doc},
    {"get_epoch_timestamps_laser", (PyCFunction) siffio_get_epoch_laser, METH_VARARGS|METH_KEYWORDS, siffio_get_epoch_timestamps_laser_doc},
    {"get_epoch_timestamps_system", (PyCFunction) siffio_get_epoch_system, METH_VARARGS|METH_KEYWORDS, siffio_get_epoch_timestamps_system_doc},
    {"get_epoch_both", (PyCFunction) siffio_epoch_both, METH_VARARGS|METH_KEYWORDS, siffio_get_epoch_timestamps_both_doc},

    // Frame methods
    {"get_frames", (PyCFunction) siffio_get_frames, METH_VARARGS|METH_KEYWORDS, siffio_get_frames_doc},
    {"get_frame_metadata", (PyCFunction) siffio_get_frame_metadata, METH_VARARGS|METH_KEYWORDS, siffio_get_frame_metadata_doc},
    {"get_appended_text", (PyCFunction) siffio_get_appended_text, METH_VARARGS | METH_KEYWORDS, siffio_get_appended_text_doc},
    {"pool_frames", (PyCFunction) siffio_pool_frames, METH_VARARGS|METH_KEYWORDS, siffio_pool_frames_doc},
    
    // Flim methods
    {"flim_map", (PyCFunction) siffio_flim_map, METH_VARARGS|METH_KEYWORDS, siffio_flim_map_doc},

    // ROI methods
    {"sum_roi", (PyCFunction) siffio_sum_roi, METH_VARARGS|METH_KEYWORDS, siffio_sum_roi_doc},
    {"sum_rois", (PyCFunction) siffio_sum_rois, METH_VARARGS|METH_KEYWORDS, siffio_sum_rois_doc},
    {"sum_roi_flim", (PyCFunction) siffio_sum_roi_flim, METH_VARARGS|METH_KEYWORDS, siffio_sum_roi_flim_doc},
    {"sum_rois_flim", (PyCFunction) siffio_sum_rois_flim, METH_VARARGS|METH_KEYWORDS, siffio_sum_rois_flim_doc},
    
    //Histogram methods
    {"get_histogram", (PyCFunction) siffio_get_histogram, METH_VARARGS|METH_KEYWORDS, siffio_get_histogram_doc},
    {NULL},
};

static PyObject* siffio_filename_get(SiffIO* self){
    //PyUnicode_FromString()
    if (self->siffreader->isOpen()){
            return PyUnicode_FromString(self->siffreader->filename.c_str());
        }
    Py_INCREF(Py_None);
    return Py_None;
};

static int siffio_filename_set(SiffIO* self, PyObject* args){
    PyErr_SetString(
        PyExc_ValueError,
        "Cannot set filename -- can only open a new file."
    );
    return -1;
}

static PyObject* siffio_status_get(SiffIO* self){
    std::string retstr;
    retstr += "Filename: " + std::string(self->siffreader->filename) + "\n";
    retstr += "Error string: " + std::string(self->siffreader->getErrString());
    return PyUnicode_FromString(retstr.c_str());
};

static int siffio_status_set(SiffIO* self, PyObject* args){
    PyErr_SetString(
        PyExc_ValueError,
        "Cannot set status."
    );
    return -1;
};

static PyObject* siffio_debug_get(SiffIO* self){
    if(debug_SIFFIO) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
};

static int siffio_debug_set(SiffIO* self, PyObject* args){
    
    if(!PyBool_Check(args)){
        PyErr_SetString(
            PyExc_ValueError,
            "Must set with a bool type."
        );
        return -1;
    }

    debug_SIFFIO = (args == Py_True);

    return 0;
};
/*
static PyObject* siffio_filename_get(SiffIO* self);
static int siffio_filename_set(SiffIO* self, PyObject* args);
static PyObject* siffio_status_get(SiffIO* self);
static int siffio_status_set(SiffIO* self, PyObject* args);
static PyObject* siffio_debug_get(SiffIO* self);
static int siffio_debug_set(SiffIO* self, PyObject* args);
*/
static PyGetSetDef siffio_getset[] = {
    {"filename", (getter) siffio_filename_get, (setter) siffio_filename_set, PyDoc_STR("Retrieves filename from C++ object.")},
    {"status" , (getter) siffio_status_get , (setter) siffio_status_set, PyDoc_STR("Checks the status of the SiffIO's SiffReader")},
    {"debug" , (getter) siffio_debug_get, (setter) siffio_debug_set, PyDoc_STR("Determines whether or not the SiffIO object will store a debug log.")},
    {NULL},
};

/******************
 * 
 * SIFFIO TYPE
 * 
 * */

PyTypeObject SiffIOType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    SIFFIO_TPNAME,        /* tp_name */
    sizeof(SiffIO),       /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) siffio_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc) siffio_repr,    /* tp_repr */
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
    PyDoc_STR(SIFFIO_DOCSTRING),     /* tp_doc */
    0,                         /* tp_traverse */
    (inquiry) siffio_clear,    /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    siffio_methods,            /* tp_methods */
    siffio_members,            /* tp_members */
    siffio_getset,             /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) siffio_init,    /* tp_init */
    0,                         /* tp_alloc */
    (newfunc) siffio_new       /* tp_new */
};

//extern PyTypeObject SiffIOType;

#endif