/*

Defines a Python class SiffIO which
handles file I/O for each file type. This
cleans up SiffPy pretty dramatically relative
to earlier versions:

1) Allows multiple files to be opened from one
interpreter

2) Allows each SiffReader object from SiffPy to
operate independently

*/

#include "../include/siffio/siffio.hpp"
#define PY_SSIZE_T_CLEAN
#define KWARG_CAST(x) const_cast<char **>(x) // so fucking dumb

/*
Called on creation of an instance of a siffio.

Creates a blank C++ siffreader
*/

bool debug_SIFFIO = false;

// Allocate the new SiffReader++
static PyObject* siffio_new(PyTypeObject* type, PyObject* args, PyObject* kwargs){
    SiffIO *self;
    self = (SiffIO *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->siffreader = new SiffReader();
        self->frameDataList = PyList_New(0);
        self->status = std::string("");
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
    Py_CLEAR(self->frameDataList);

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
    Py_CLEAR(self->frameDataList);
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

static PyObject* siffio_num_frames(SiffIO* self){
    // Returns number of frames in file

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
    if (framesArrayN != framesN) {
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

int check_registration(PyObject* registrationDict, PyObject* frames_list){
    // Check to make sure the dict contains all the frames requested
    // and if not, set their registration to (0,0)
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

static PyObject* siffio_get_frames(SiffIO *self, PyObject *args, PyObject* kw) {
    // 

    static const char* GET_FRAMES_KWARGS[] = {
        "frames", "registration", "as_array", NULL};

    bool make_array = false;

    PyObject *frames_list = NULL;
    PyObject* registrationDict = NULL;
    PyObject* as_array = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!O!O:get_frames", 
        KWARG_CAST(GET_FRAMES_KWARGS),
        &PyList_Type, &frames_list,
        &PyDict_Type, &registrationDict,
        &as_array
        )
    ) {
        PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

    if(frames_list==NULL) {
        PyErr_SetString(PyExc_TypeError, "Must provide a list of frames");
        return NULL;
    }

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
    bool registrationDictProvided = registrationDict != NULL;
    if (registrationDict == NULL) {
        registrationDict = PyDict_New();
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

static PyObject* siffio_get_frame_metadata(SiffIO *self, PyObject *args, PyObject* kw) {
    // Gets the meta data for the frames in kw, or for all the frames.  

    static const char* GET_FRAMES_METADATA_KEYWORDS[] = {"frames", NULL};
    PyObject *frames_list = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_frames_metadata", 
            KWARG_CAST(GET_FRAMES_METADATA_KEYWORDS), 
            &PyList_Type, &frames_list)
        ) {
        PyErr_SetString(PyExc_TypeError,"Error in parsing input arguments");
        return NULL;
    }

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
    return NULL;

    // Check that listOfFramesListed is a list of lists, and that the elements of that are ints.
    for (Py_ssize_t idx = Py_ssize_t(0); idx < PyList_Size(listOfFramesListed); idx++) {
        PyObject* item = PyList_GET_ITEM(listOfFramesListed, idx);
        if(!PyList_Check(item)) {
            if (need_to_decref_regdict) Py_DECREF(registrationDict);
            PyErr_SetString(PyExc_TypeError, "All elements of pool_list must be lists themselves");
            return NULL;
        }
        
        // Have been surprised by encountering int overflow here, 65k photons per pixel requires either massive data
        // rates or loooots of pooling. I should do smarter checking but this is a short term solution.
        if (PyList_Size(item) > 10000) {
            PyErr_WarnEx(PyExc_RuntimeWarning, "Pooling a large number of frames! May cause uint16 overflow.",Py_ssize_t(1));
        }

        check_framelist(item, NULL, 0, self->siffreader->numFrames());

        // Now typecheck the registration dict item
        // If it's not a tuple of PyLongs, replace it
        // with a tuple of PyLongs made by attemping to cast
        // the elements to one and replacing the tuple.
        
        check_registration(registrationDict, item);
    }

    try{
        PyObject* pool(
            (PyObject*) self->siffreader->poolFrames(
                listOfFramesListed,
                flim,
                registrationDict
            )
        );
        if (need_to_decref_regdict) Py_DECREF(registrationDict);
        return pool;
    }

    catch(...) {
        if (need_to_decref_regdict) Py_DECREF(registrationDict);
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        return NULL;
    }
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
    if(!PyArg_ParseTupleAndKeywords(args, kw, "O|$O!s#O!:flim_map", 
        KWARG_CAST(FLIM_MAP_KEYWORDS),
        &FLIMParams,
        &PyList_Type, &listOfFrames,
        &conf_measure, &conf_measure_length,
        &PyDict_Type, &registrationDict
        )) {
        return NULL;
    }

    if(!conf_measure_length) conf_measure = (char*) "chi_sq";
    // defaults to 0's
    bool need_to_decref_regdict = false;
    if(!registrationDict) {
        registrationDict = PyDict_New(); // TODO: DECREF ME!!!!
        need_to_decref_regdict = true;
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

    uint64_t frames[PyList_Size(listOfFrames)];
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
/*

ROI methods

*/

static PyArrayObject* siffio_sum_roi(SiffIO* self, PyObject* args, PyObject*kw){
    /*
    Returns the summed photon counts within the provided ROI

    Args : numpy array ROI mask, dtype 'bool'
    Kwargs : 
        frames : list[int]
            - If none provided, uses all frames.
        registration : dict
            - If none provided, uses no shift
    */
   static const char* SUM_ROIS_KEYWORDS[] = {"mask", "frames", "registration", NULL};

    PyArrayObject* mask;
    PyObject *frames_list = NULL;
    PyObject* registrationDict = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$O!O!:sum_roi",
        KWARG_CAST(SUM_ROIS_KEYWORDS),
        &mask,
        &PyList_Type, &frames_list,
        &PyDict_Type, &registrationDict
        )
     ) {
            PyErr_SetString(PyExc_ValueError, "Error parsing input arguments.");
            return NULL;
    }

    if(!frames_list) { // populate with all frames. BAD to do if there are flyback frames
        Py_ssize_t nFrames = self->siffreader->numFrames();
        frames_list = PyList_New(nFrames);
        for (Py_ssize_t frame_idx = Py_ssize_t(0); frame_idx < nFrames ; frame_idx++){
            PyList_SET_ITEM(frames_list,frame_idx,PyLong_FromSsize_t(frame_idx));
        }
    }
    
    bool need_to_decref_dict = false;
    if(registrationDict == NULL) {
        registrationDict = PyDict_New();
        need_to_decref_dict = true;
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
        PyArrayObject* returnedMask = self->siffreader->sumMask(framesArray, framesN, (PyArrayObject*) mask, registrationDict);
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

static PyArrayObject* siffio_sum_roi_flim(SiffIO* self, PyObject* args, PyObject*kw){
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

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|$O!O!:sum_roi", 
        KWARG_CAST(SUM_ROI_FLIM_KEYWORDS),
        &mask,
        &FLIMParams,
        &PyList_Type, &frames_list,
        &PyDict_Type, &registrationDict
        )
     ) {
            PyErr_SetString(PyExc_ValueError, "Error parsing input arguments.");
            return NULL;
    }

    if(PyArray_TYPE(mask) != NPY_BOOL) {
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

    if(!frames_list) { // populate with all frames.
        Py_ssize_t nFrames = self->siffreader->numFrames();
        frames_list = PyList_New(nFrames);
        for (Py_ssize_t frame_idx = Py_ssize_t(0); frame_idx < nFrames ; frame_idx++){
            PyList_SET_ITEM(frames_list,frame_idx,PyLong_FromSsize_t(frame_idx));
        }
    }
    
    bool need_to_decref_dict = false;
    if(registrationDict==NULL) {
        registrationDict = PyDict_New();
        need_to_decref_dict = true;
    }

    // Check that all elements of the frame list and registration dictionary are valid.
    uint64_t* framesArray = new uint64_t[PyList_Size(frames_list)];
    for(Py_ssize_t idx = Py_ssize_t(0); idx < PyList_Size(frames_list); idx++) {
        PyObject* item = PyList_GET_ITEM(frames_list, idx);
        if(!PyLong_Check(item)) {
            if (need_to_decref_dict) Py_DECREF(registrationDict);
            PyErr_SetString(PyExc_TypeError, "All elements of frame list must be ints");
            delete[] framesArray;
            return NULL;
        }
        uint64_t frameNum  = PyLong_AsUnsignedLongLong(item);
        if (frameNum >= self->siffreader->numFrames()) {
            if (need_to_decref_dict) Py_DECREF(registrationDict);
            PyErr_SetString(PyExc_ValueError, "Frame number provided is greater than indices of frames.\nRemember they are zero indexed!");
            delete[] framesArray;
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
        
        // Now typecheck the registration dict item
        // If it's not a tuple of PyLongs, replace it
        // with a tuple of PyLongs made by attemping to cast
        // the elements to one and replacing the tuple.
        
        try {
            PyObject* shiftTuple = PyDict_GetItem(registrationDict, item);
            if (!PyTuple_Check(shiftTuple)) {
                if (need_to_decref_dict) Py_DECREF(registrationDict);
                PyErr_SetString(
                    PyExc_TypeError,
                    (
                        std::string("Registration dictionary element for frame ") + 
                        std::to_string(PyLong_AsLongLong(item)) +
                        std::string(" is not a tuple.")
                    ).c_str()
                );
                delete[] framesArray;
                return NULL;
            }
            Py_ssize_t tupLen = PyTuple_Size(shiftTuple);
            for(Py_ssize_t tupIdx = 0; tupIdx < tupLen; tupIdx++) {
                PyObject* shiftValue = PyTuple_GetItem(shiftTuple, tupIdx);
                if(!PyLong_Check(shiftValue)) { // if it's not okay, try to cast it
                    PyObject* result = PyObject_CallMethod(shiftValue, "__int__", NULL);
                    if (result == NULL) {
                        if (need_to_decref_dict) Py_DECREF(registrationDict);
                        PyErr_SetString(
                            PyExc_TypeError,
                            (
                                std::string("Registration dictionary element for frame ") + 
                                std::to_string(PyLong_AsLongLong(item)) +
                                std::string(" cannot be cast to type int.")
                            ).c_str()
                        );
                        delete[] framesArray;
                        return NULL;
                    }
                    PyTuple_SetItem(shiftTuple, tupIdx, result);
                }
            }
        }
        catch (...) {
            if (need_to_decref_dict) Py_DECREF(registrationDict);
            PyErr_SetString(PyExc_RuntimeError,
                (std::string("Failure to access registration dictionary element for frame ") +
                std::to_string(PyLong_AsLongLong(item))).c_str()
            );
            delete[] framesArray;
            return NULL;
        }
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

    static const char* GET_HISTOGRAM_KEYWORDS[] = {"frames", NULL};

    PyObject* frames = NULL;

    // | indicates optional args, $ indicates all following args are keyword ONLY
    if(!PyArg_ParseTupleAndKeywords(args, kw, "|$O!:get_histogram", 
        KWARG_CAST(GET_HISTOGRAM_KEYWORDS), &PyList_Type, &frames)) {
        return NULL;
    }

    try{
        if(!frames) {
            return self->siffreader->getHistogram();
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
        PyArrayObject* histo = self->siffreader->getHistogram(framesArray, framesN);
        delete[] framesArray;
        return histo;
    }
    catch(...) {
        PyErr_SetString(PyExc_RuntimeError, self->siffreader->getErrString());
        delete[] framesArray;
        return NULL;
    }
}


/**********************
 * 
 * 
 *  GETTER AND SETTER METHODS
 * 
 * 
 * */

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

/******************
 * 
 * SIFFIO TYPE
 * 
 * */

PyTypeObject SiffIOType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = SIFFIO_TPNAME,
    .tp_doc = PyDoc_STR(SIFFIO_DOCSTRING),
    .tp_basicsize = sizeof(SiffIO),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = (newfunc) siffio_new,
    .tp_init = (initproc) siffio_init,
    .tp_dealloc = (destructor) siffio_dealloc,
    .tp_clear = (inquiry) siffio_clear,
//    .tp_getattr = (getattrfunc) siffio_getattr,
    .tp_repr = (reprfunc) siffio_repr,
    .tp_members = siffio_members,
    .tp_methods = siffio_methods,
    .tp_getset  = siffio_getset,

};