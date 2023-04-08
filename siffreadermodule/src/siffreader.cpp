// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API
#define PY_SSIZE_T_CLEAN
#include "../include/siffreader/siffreader.hpp"

#include "../include/siffreader/sifdefin.hpp"
#include "../include/siffreader/siffreaderinline.hpp"
#include <algorithm>

// Appends the error to errstring and rethrows()
#define REPORT_ERR(x) \
    catch(std::exception &e) {\
        errstring = std::string(x) + e.what();\
        throw e;\
    }

// Appends the error to errstring, executes y, and rethrows
#define REPORT_ERR_EXEC(x,y) \
    catch(std::exception &e) {\
        y; \
        errstring = std::string(x) + e.what();\
        throw e;\
    }

///////////////////////
////// FILE I/O ///////
///////////////////////
//
// BIG TIME TODO: MAKE THIS CROSS-ENDIAN COMPATIBLE.

SiffReader::SiffReader(){
    suppress_errors = false; 
    suppress_warnings = false;
    debug = false;
    debug_clock = std::chrono::high_resolution_clock();
}

int SiffReader::openFile(const char* _filename) {
    // Opens a .siff file for further analysis / use.
    try{
        // First make sure we can open it at all.
        if(siff.is_open()) {
            if (filename == _filename) {
                errstring = std::string("\n\nThis file is already open. Was that an accident?\n");
                return -2;
            }
            siff.close();
            reset();
        }
        if (!siff.good()) {siff.clear();}
        siff.open(_filename, std::ios::binary | std::ios::in);
        if (!siff.good()) {siff.clear();}
        if (!(siff.is_open())) throw std::runtime_error("Could not open putative .siff file. Check that path exists.\nAttempted path: "+std::string(_filename));
        
        // Get the file size:
        siff.seekg(0, siff.end);
        params.fileSize = siff.tellg();
        siff.seekg(0, siff.beg);
        
        // Now check that it's a siff file.
        // Gotta know the endianness
        char *endian = new char[2];
        siff.read(endian, sizeof(char)*2);   

        // strcmp == 0 if they match.
        if (
            (strcmp(endian,BIGENDIAN) !=0)
            &&
            (strcmp(endian,LITTLEENDIAN) != 0)
        ) {
            throw std::runtime_error(
                std::string("Could not deduce endian. May not be .siff/.tiff file.")+
                " First two bytes (should be II or MM): "+
                std::string(endian)
            );
        }
        params.little = (strcmp(endian,LITTLEENDIAN) == 0); // true if little, false if big.

        delete[] endian;

        // temporary solution: if endian-ness doesn't match, give up.
        uint16_t i = 1; // the uint16_t 1 is 0x01 in big endian, 0x10 in little endian
        char* c = (char*)&i;
        // dereferencing c will be 1 if the least significant byte is first. 0 if not.
        if(! (((bool)*c) == params.little) ) throw std::runtime_error("ENDIANS DON'T MATCH AND I HAVEN'T FIXED THAT YET.");
        
        // Check the magic numbers
        uint16_t tiffid;
        siff.read((char*)&tiffid, sizeof(uint16_t));

        if(!((tiffid == BIGTIFFID) || (tiffid == TIFFID))) throw std::runtime_error("Could not verify that file is a true .tiff or .siff based on magic numbers.");
        params.bigtiff = (tiffid == BIGTIFFID);

        // TEMPORARY!!! TODO: BETTER CHECK FOR SIFFNESS in the file itself maybe?
        params.issiff = (strcmp(".siff",strrchr(_filename, '.')) == 0);

        if (params.bigtiff) {
            //  here the headers diverge a bit
            uint16_t offset_size;
            siff.read((char*)&offset_size, sizeof(uint16_t));
            params.bytesPerPointer = offset_size; // sure to be 8 byte.
            params.bytesPerNumTags = 8;
            siff.read((char*)&offset_size, sizeof(uint16_t)); // these are always 0.
            if(offset_size) throw std::runtime_error("File is not a valid BIGTIFF or .SIFF.");
            
            uint64_t firstIFD;
            siff.read((char*)&firstIFD,params.bytesPerPointer);
            params.firstIFDAddress = firstIFD;
            params.bytesPerTag = 20;
        }
        else {
            // regular ol' tiff
            uint32_t firstIFD;
            siff.read((char*)&firstIFD, sizeof(uint32_t));
            params.firstIFDAddress = (uint64_t) firstIFD;
            params.bytesPerTag = 12;
            params.bytesPerNumTags = 2;
        }
    
        // Now do the ScanImage-specific checks!
        uint32_t magic;
        siff.read((char*)&magic, sizeof(uint32_t));
        
        uint32_t si;
        siff.read((char*)&si, sizeof(uint32_t));
        
        if( !( (magic == MAGICNUMBER) && (si == SI2019) ) ) throw std::runtime_error("File is a .tiff, but was not produced by ScanImage");

        // ScanImage data stuff
        uint32_t NVFD;
        siff.read((char*)&NVFD, sizeof(uint32_t));
        params.NVFD_length = NVFD; // in bytes

        uint32_t ROI;
        siff.read((char*)&ROI, sizeof(uint32_t));
        params.ROI_string_length = ROI; // in bytes

        
        char headerstring[params.NVFD_length];
        siff.read(headerstring, params.NVFD_length);
        params.headerstring = std::string(headerstring);

        char roistring[params.ROI_string_length];
        siff.read(roistring, params.ROI_string_length);
        params.ROI_string = std::string(roistring);
              
        // Finally, keep track of the filename. We're happy.
        filename = _filename;
        params.suppress_warnings = suppress_warnings;

        discernFrames(); // updates params.numFrames with the number of frames in the siff
        
        return 0;
    }
    catch(std::exception& e){
        if (siff.is_open()) siff.close();
        errstring = std::string("Could not open file: ") + e.what(); 
        return -1;
    }
};

bool SiffReader::isOpen(){
    return siff.is_open();
}
//
void SiffReader::discernFrames() {
    // Runs through all the frames to find their IFD, calculates the total number of frames,
    // and stores all the IFDs for quick lookup.
    uint64_t nextIFD = params.firstIFDAddress;
    uint64_t currIFD = 0;
    uint64_t numTags; // number of tags in this directory before the real metadata

    siff.seekg(nextIFD, std::ios::beg); // go there first
    while(nextIFD>0 && nextIFD<params.fileSize && !siff.eof()) {
        currIFD = nextIFD;

        params.allIFDs.push_back(currIFD);
        frameDatas.push_back(getTagData(currIFD, params, siff));

        siff.seekg(currIFD, std::ios::beg); // go back to the beginning of the IFD
        siff.read((char*)&numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.

        siff.seekg(numTags*params.bytesPerTag,std::ios::cur); // skip the tags

        siff.read((char*)&nextIFD, params.bytesPerPointer);

        // go to the next IFD
        siff.seekg(nextIFD, std::ios::beg); // go there first
    }
    while(params.allIFDs.back() == 0) {
        params.allIFDs.pop_back(); // this is for the case that the last IFD is 0ULL
    }
    params.numFrames = params.allIFDs.size();
    //siff.clear(); // get rid of failbits
}

bool SiffReader::dimensionsConsistent(uint64_t frames[], uint64_t framesN){
    // Checks that all the frames have the same dimensions.
    // Returns true if they are consistent, false otherwise.
    if (framesN == 0) return true;
    //return false;
    uint64_t firstFrame = frames[0];
    uint64_t firstFrameWidth = frameDatas[firstFrame].imageWidth;
    uint64_t firstFrameHeight = frameDatas[firstFrame].imageLength;
    for (uint64_t i=1; i<framesN; i++) {
        if (frameDatas[frames[i]].imageWidth != firstFrameWidth) return false;
        if (frameDatas[frames[i]].imageLength != firstFrameHeight) return false;
    }
    return true;
}
 

void SiffReader::closeFile(){
    if (siff.is_open()) siff.close();
    params = SiffParams();
    filename = std::string();
}


////////////////////////////
///// GET FILE DATA ////////
////////////////////////////

PyArrayObject* SiffReader::sumMask(uint64_t frames[], uint64_t framesN, PyArrayObject* mask, PyObject* registrationDict){
    // Sums all pixel elements of the desired frames within the mask and returns a 1d PyArrayObject.
    try{
        if (!siff.is_open()) throw std::runtime_error("No open file.");
        //if (!params.issiff) throw std::runtime_error("File is not of type .siff! No FLIM data to collect.");
        siff.clear();

        // 1 dimensional
        npy_intp dims[1];
        dims[0] = framesN;
        PyArrayObject* summedArray = (PyArrayObject*) PyArray_ZEROS(
            1, // 1d array
            dims, // length equal to number of frames
            NPY_UINT64, // count data, and compressed enough to not care about saving space
            0 // C order, i.e. last index increases fastest
        ); // Or should I make a sparse array? Maybe make that an option? TODO.

        uint64_t* data_ptr = (uint64_t *) PyArray_DATA(summedArray);

        if (PyArray_TYPE(mask) != NPY_BOOL) throw std::runtime_error("Mask is not of dtype 'bool'");

        for(size_t frame_idx = 0; frame_idx < framesN; frame_idx++){

            FrameData frameData = getTagData(params.allIFDs[frames[frame_idx]], params, siff);

            PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(frames[frame_idx]));
            
            data_ptr[frame_idx] = sumFrameMask(
                frameData,
                params,
                mask,
                shift_tuple,
                siff
            );
        }

        return summedArray;

    }
    REPORT_ERR("Error in sum_roi mask method: ");
}

PyArrayObject* SiffReader::sumFLIMMask(uint64_t frames[], uint64_t framesN, PyObject* FLIMParams, PyArrayObject* mask, PyObject* registrationDict) {
    // sums empirical lifetime inside the provided mask

    // Boilerplate-y, some bad code practice in here (for it to be so similar to the non-FLIM version).
    try{
        if (!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();

        // 1 dimensional
        npy_intp dims[1];
        dims[0] = framesN;
        PyArrayObject* summedArray = (PyArrayObject*) PyArray_ZEROS(
            1, // 1d array
            dims, // length equal to number of frames
            NPY_DOUBLE, // empirical lifetime (albeit in units of bins), needs sub-bin resolution
            0 // C order, i.e. last index increases fastest
        ); // Or should I make a sparse array? Maybe make that an option? TODO.

        double_t* data_ptr = (double_t *) PyArray_DATA(summedArray);

        // Offset for empirical tau
        double_t tau_o = PyFloat_AsDouble(PyObject_GetAttrString(FLIMParams,"T_O"));

        if (PyArray_TYPE(mask) != NPY_BOOL) throw std::runtime_error("Mask is not of dtype 'bool'");

        for(size_t frame_idx = 0; frame_idx < framesN; frame_idx++){

            FrameData frameData = getTagData(params.allIFDs[frames[frame_idx]], params, siff);

            PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(frames[frame_idx]));
            
            data_ptr[frame_idx] = sumFrameFLIMMask(
                frameData,
                params,
                tau_o,
                mask,
                shift_tuple,
                siff
            );
        }
        return summedArray;
    }
    REPORT_ERR("Error in sum_roi_flim mask method: ")
};

// TODO!!!!!
PyArrayObject* SiffReader::roiMask(uint64_t frames[], uint64_t framesN, bool flim, PyArrayObject* mask, PyObject* registrationDict) {
    // Returns a 1d numpy array of only the pixels within the mask of the mask object

    

    try{
        throw std::runtime_error("Roi mask method not yet implemented.");
    }
    REPORT_ERR("Error in roiMask method: ");
}


// VANILLA
PyObject* SiffReader::retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim) {
    // By default, retrieves ALL frames, returns as a list of numpy arrays including the arrival times.
    // TODO: Implement frame selection, automatically detect .tiffs to make flim=false, implement
    // the variable type of output
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();
        // create the list into which we shall stuff the numpy arrays
        PyObject* numpyArrayList = PyList_New(Py_ssize_t(0));
        PyObject* shift_tuple = PyTuple_Pack(Py_ssize_t(2), // steals references, makes life easier
                        PyLong_FromLong(0),
                        PyLong_FromLong(0)
                    );
        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                singleFrameRetrieval(params.allIFDs[frames[i]], numpyArrayList, flim, shift_tuple);
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                singleFrameRetrieval(params.allIFDs[i], numpyArrayList, flim, shift_tuple);
            }
        }
        return numpyArrayList;
    }
    REPORT_ERR("Error parsing frames: ");
}

PyArrayObject* SiffReader::retrieveFramesAsArray(
    uint64_t frames[],
    uint64_t framesN,
    bool flim,
    PyObject* registrationDict
) {
    // Build a PyArray object from the framelist
    // with length equal to the number of frames
    // and dtype uint64_t
    PyArray_Descr* desc(PyArray_DescrFromType(NPY_UINT64));

    const int ND = 3;
    // TEMP

    npy_intp retdims[ND];
    retdims[0] = framesN;
    retdims[1] = frameDatas[frames[0]].imageLength;
    retdims[2] = frameDatas[frames[0]].imageWidth;

    PyArrayObject* retArray((PyArrayObject*)PyArray_ZEROS(
        ND,
        retdims,
        NPY_UINT16,
        0
    ));

    uint16_t *data_ptr = (uint16_t *) PyArray_DATA(retArray);
    size_t sizeOfFrame = retdims[1] * retdims[2];
    uint64_t frameIdx;
    FrameData frameData;
    PyObject* shift_tuple;
    // Now parse the frames into the array
    for (uint64_t listIdx = 0; listIdx < framesN; listIdx++){
        frameIdx = frames[listIdx];
        frameData = getTagData(params.allIFDs[frameIdx], params, siff);
        shift_tuple = PyDict_GetItem(
            registrationDict,
            PyLong_FromUnsignedLongLong(frameIdx)
        );

        loadArrayWithData(
            &data_ptr[listIdx * sizeOfFrame],
            &PyArray_DIMS(retArray)[1], // y, x only
            params,
            frameData,
            siff,
            flim,
            shift_tuple
        );
        Py_DECREF(shift_tuple);
    }


    return retArray;
};

PyArrayObject* SiffReader::retrieveFramesAsArray(
    uint64_t frames[],
    uint64_t framesN,
    bool flim,
    PyObject* registrationDict,
    uint64_t terminalBin
) {
    throw std::runtime_error("NOT IMPLEMENTED");
};

// REGISTRATION DICT
PyObject* SiffReader::retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim, PyObject* registrationDict) {
    // By default, retrieves ALL frames, returns as a list of numpy arrays including the arrival times.
    // TODO: Implement frame selection, automatically detect .tiffs to make flim=false, implement
    // the variable type of output
    if (registrationDict == NULL){
        return retrieveFrames(frames, framesN, flim);
    }
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();
        // create the list into which we shall stuff the numpy arrays
        PyObject* numpyArrayList = PyList_New(Py_ssize_t(0));
        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(frames[i]));
                singleFrameRetrieval(params.allIFDs[frames[i]], numpyArrayList, flim, shift_tuple);
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(i));
                singleFrameRetrieval(params.allIFDs[i], numpyArrayList, flim, shift_tuple);
            }
        }
        return numpyArrayList;
    }
    REPORT_ERR("Error parsing frames: ")
}

// TERMINAL BIN
PyObject* SiffReader::retrieveFrames(uint64_t frames[], 
    uint64_t framesN, bool flim, PyObject* registrationDict, uint64_t terminalBin) {
    // By default, retrieves ALL frames, returns as a list of numpy arrays including the arrival times.
    // TODO: Implement frame selection, automatically detect .tiffs to make flim=false, implement
    // the variable type of output
    if (terminalBin == NULL){
        return retrieveFrames(frames, framesN, flim, registrationDict);
    }
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        if(!params.issiff) return retrieveFrames(frames, framesN, flim, registrationDict);
        siff.clear();
        // create the list into which we shall stuff the numpy arrays
        PyObject* numpyArrayList = PyList_New(Py_ssize_t(0));
        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(frames[i]));
                singleFrameRetrieval(params.allIFDs[frames[i]], numpyArrayList, flim, terminalBin, shift_tuple);
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(i));
                singleFrameRetrieval(params.allIFDs[i], numpyArrayList, flim, terminalBin, shift_tuple);
            }
        }
        //throw std::runtime_error("Terminal bin not yet implemented!");
        return numpyArrayList;
    }
    REPORT_ERR("Error parsing frames: ");
}

PyObject* SiffReader::poolFrames(PyObject* listOfLists, bool flim, PyObject* registrationDict) {
    // Pools all frames into one summed element by reading one at a time
    // and then appending them together. Considerably less memory intensive
    // than reading them all into NumPy arrays first and THEN pooling.
    // Permits multiple lists of lists, returning a single numpy array
    // for each sublist.
    try{
        PyObject* returnedList = PyList_New(Py_ssize_t(0));
        // iterate through each list of frame indices
        for(Py_ssize_t idx(0); idx < PyList_Size(listOfLists); idx++) {
            // one merged numpy array for all of them. TODO: size checking!!
            // need to ensure they all have compatible dimensions -- for now
            // I just assume it, but as this expands to support mROI...

            // already ensured these were all PyLongs
            PyObject* listOfFrames = PyList_GetItem(listOfLists, idx);
            
            if(PyList_Size(listOfFrames)==0) { // empty list, you silly goose.
                PyList_Append(returnedList,Py_None);
                Py_DECREF(Py_None);
            }
            // get the first frame requested.

            PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyList_GetItem(listOfFrames,0));
            PyArrayObject* firstFrame = 
                frameAsNumpy(params.allIFDs[PyLong_AsLongLong(PyList_GetItem(listOfFrames,0))], flim, shift_tuple);

            // fuse in more if they asked for it.
            if(PyList_Size(listOfFrames) > Py_ssize_t(1)) {
                // more than one frame in the list
                for(Py_ssize_t frameIdx(1); frameIdx < PyList_Size(listOfFrames); frameIdx++) {
                    PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyList_GetItem(listOfFrames,frameIdx)); 
                    fuseFrames(
                        firstFrame, // template fused onto
                        params.allIFDs[ PyLong_AsLongLong(PyList_GetItem(listOfFrames,frameIdx)) ], // next frame to add
                        flim, // what style to fuse it onto
                        shift_tuple // registration shift
                    ); 
                }
            }
            
            PyList_Append(returnedList, (PyObject*) firstFrame); // ADDS a reference
            Py_DECREF(firstFrame); // prevent memory leaks on this object
        }
        return returnedList;
    }
    REPORT_ERR("Error in pool frames: ")
}

PyObject* SiffReader::poolFrames(PyObject* listOfLists, uint64_t terminalBins,
    bool flim, PyObject* registrationDict) {
    // Pools all frames into one summed element by reading one at a time
    // and then appending them together. Considerably less memory intensive
    // than reading them all into NumPy arrays first and THEN pooling.
    // Permits multiple lists of lists, returning a single numpy array
    // for each sublist.
    
    if (!params.issiff) return poolFrames(listOfLists, flim, registrationDict);

    try{
        PyObject* returnedList = PyList_New(Py_ssize_t(0));
        // iterate through each list of frame indices
        for(Py_ssize_t idx(0); idx < PyList_Size(listOfLists); idx++) {
            // one merged numpy array for all of them. TODO: size checking!!
            // need to ensure they all have compatible dimensions -- for now
            // I just assume it, but as this expands to support mROI...

            // already ensured these were all PyLongs
            PyObject* listOfFrames = PyList_GetItem(listOfLists, idx);
            
            if(PyList_Size(listOfFrames)==0) { // empty list, you silly goose.
                PyList_Append(returnedList,Py_None);
                Py_DECREF(Py_None);
            }
            // get the first frame requested.

            PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyList_GetItem(listOfFrames,0));
            PyArrayObject* firstFrame = 
                frameAsNumpy(params.allIFDs[PyLong_AsLongLong(PyList_GetItem(listOfFrames,0))], terminalBins, flim, shift_tuple);

            // fuse in more if they asked for it.
            if(PyList_Size(listOfFrames) > Py_ssize_t(1)) {
                // more than one frame in the list
                for(Py_ssize_t frameIdx(1); frameIdx < PyList_Size(listOfFrames); frameIdx++) {
                    PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyList_GetItem(listOfFrames,frameIdx)); 
                    fuseFrames(
                        firstFrame, // template fused onto
                        params.allIFDs[ PyLong_AsLongLong(PyList_GetItem(listOfFrames,frameIdx)) ], // next frame to add
                        flim, // what style to fuse it onto
                        shift_tuple // registration shift
                    ); 
                }
            }
            
            PyList_Append(returnedList, (PyObject*) firstFrame); // ADDS a reference
            Py_DECREF(firstFrame); // prevent memory leaks on this object
        }
        return returnedList;
    }
    REPORT_ERR("Error in pool frames: ");
}

PyObject* SiffReader::flimMap(PyObject* FLIMParams, PyObject* listOfLists, PyObject* registrationDict) {
    // For no confidence measure
    try{
        PyObject* T_O = PyObject_GetAttrString(FLIMParams, "T_O");
        double_t tauo = PyFloat_AS_DOUBLE(T_O);
        Py_DECREF(T_O);
        if ((tauo == -1.0) && PyErr_Occurred()) {
            throw std::runtime_error("Purported FLIMParams object has no attribute 'T_O'.");
        }

        PyObject* TupleOutList = PyList_New(Py_ssize_t(0));

        // iterate through each list of frame indices
        for(Py_ssize_t idx(0); idx < PyList_Size(listOfLists); idx++) {
            // one merged numpy array for all of them. TODO: size checking!!
            // need to ensure they all have compatible dimensions -- for now
            // I just assume it, but as this expands to support mROI...

            // already ensured these were all PyLongs
            PyObject* listOfFrames = PyList_GetItem(listOfLists, idx);
            
            if(PyList_Size(listOfFrames)==0) { // empty list, you silly goose.
                PyList_Append(TupleOutList,Py_None);
                Py_DECREF(Py_None);
            }
            
            PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyList_GetItem(listOfFrames,0));
            
            PyObject* flimTup = makeFlimTuple(
                    params.allIFDs[ PyLong_AsLongLong(PyList_GetItem(listOfFrames,0)) ],
                    shift_tuple
                    );

            for(Py_ssize_t frameIdx(1); frameIdx < PyList_Size(listOfFrames); frameIdx++) {
                PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyList_GetItem(listOfFrames,frameIdx)); 
                fuseIntoFlimTuple(flimTup,
                    params.allIFDs[ PyLong_AsLongLong(PyList_GetItem(listOfFrames,frameIdx)) ],
                    shift_tuple);
            }

            normalizeAndOffsetFlimTuple(flimTup, tauo); // operation can only be done after all arrivals merged

            PyList_Append(TupleOutList, flimTup); // ADDS a reference
            Py_DECREF(flimTup); // prevent memory leaks on this object

        }
        return TupleOutList;
    }
    REPORT_ERR("Error in flimMap: ");
}

PyObject* SiffReader::flimMap(PyObject* FLIMParams, PyObject* listOfLists, const char* conf_measure, PyObject* registrationDict) {
    // For the case in which a confidence measure is requested.
    try{
        if (strcmp(conf_measure,"None")==0) return flimMap(FLIMParams, listOfLists, registrationDict); // much faster not to waste time with conf
        // check the tauo offset value before we waste time evaluating things
        PyObject* T_O = PyObject_GetAttrString(FLIMParams, "T_O");
        double_t tauo = PyFloat_AS_DOUBLE(T_O);
        Py_DECREF(T_O);
        if ((tauo == -1.0) && PyErr_Occurred()) {
            throw std::runtime_error("Purported FLIMParams object has no attribute 'T_O'.");
        }

        PyObject* TupleOutList = PyList_New(Py_ssize_t(0));
        
        if (!listOfLists) { // default behavior, all frames
            // TODO: IMPLEMENT
            throw std::runtime_error("List of frames must be provided (all frame default not implemented yet.");
        }

        for(Py_ssize_t idx(0); idx < PyList_Size(listOfLists); idx++) {
            // one merged numpy array for all of them. TODO: size checking!!
            // need to ensure they all have compatible dimensions -- for now
            // I just assume it, but as this expands to support mROI...

            // already ensured these were all PyLongs
            PyObject* listOfFrames = PyList_GetItem(listOfLists, idx);
            
            if(PyList_Size(listOfFrames)==0) { // empty list, you silly goose.
                PyList_Append(TupleOutList,Py_None);
                Py_DECREF(Py_None);
            }
            FrameData firstFrameData = getTagData(params.allIFDs[PyLong_AsLongLong(PyList_GetItem(listOfFrames,0))], params, siff);

            // Have to get all the reads together in one place.
            // Opting to do this with one uint64_t vector regardless of
            // compression format.
            std::vector<uint64_t> photonReadsTogether(0);
            photonReadsTogether.reserve(
                10* PyList_Size(listOfFrames) * firstFrameData.imageLength * firstFrameData.imageWidth
            ); // make space for 10 photons per pixel. Likely unnecessarily large but since
            // this is done sequentially, probably will never be more than 2GB of RAM eaten up
            // by the whole process.
            
            for(Py_ssize_t frameIdx(0); frameIdx < PyList_Size(listOfFrames); frameIdx++) {
                PyObject* element = PyList_GetItem(listOfFrames, frameIdx);
                if(!PyDict_Contains(registrationDict, element)) {
                    PyDict_SetItem(registrationDict, element, // steals the reference to the value
                        PyTuple_Pack(Py_ssize_t(2), // steals references, makes life easier
                            PyLong_FromLong(0),
                            PyLong_FromLong(0)
                        )
                    );
                } 
                PyObject* shift_tuple = PyDict_GetItem(registrationDict, element);
                
                fuseReadVector( // simply appends this frame's read vector onto the old ones.
                    photonReadsTogether,
                    params.allIFDs[PyLong_AsLongLong(element)],
                    shift_tuple
                );
            }
            PyObject* flimMap = readVectorToNumpyTuple(photonReadsTogether,
                firstFrameData, FLIMParams, conf_measure
            );
            PyList_Append(TupleOutList, flimMap); // ADDS a reference
            Py_DECREF(flimMap); // prevent memory leaks on this object            
        }
        //
        return TupleOutList;
    }
    REPORT_ERR("Error in flimMap");
}


PyArrayObject* SiffReader::getHistogram(uint64_t frames[], uint64_t framesN) {
    // NOTE! THIS USES UINT64_T BECAUSE YOU'RE LIKELY TO GET >65k PHOTONS PER BIN
    // By default, retrieves ALL frames, returns a single numpy array of the arrival times
    // create the 1-d numpy array.
    uint16_t tau_dim = 1024; // hardcoded for now. TODO: Implement this measure in SiffWriter
    npy_intp dims[1];
    dims[0] = tau_dim;
    PyArrayObject* numpyArray = (PyArrayObject*) PyArray_ZEROS(
        1,
        dims,
        NPY_UINT64, 
        0 // C order, i.e. last index increases fastest
    );
//
//
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        if(!params.issiff) throw std::runtime_error("Not a .siff -- no arrival time data.");
        siff.clear();
        
        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                singleFrameHistogram(params.allIFDs[frames[i]], numpyArray);
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                singleFrameHistogram(params.allIFDs[i], numpyArray);
            }
        }
        return numpyArray;
    }
    REPORT_ERR("Error collecting histogram: ");
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
///////////////////////// META-DATA ////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////



PyObject* SiffReader::readMetaData(uint64_t frames[],uint64_t framesN){
    // get metadata enumerated in frames

    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();
        // create the list into which we shall stuff the numpy arrays
        PyObject* metaDictList = PyList_New(Py_ssize_t(0));
        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                singleFrameMetaData(params.allIFDs[frames[i]], metaDictList);
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                singleFrameMetaData(params.allIFDs[i], metaDictList);
            }
        }
        
        return metaDictList;
    }
    catch(std::exception& e){
        errstring = std::string("Error parsing frames: ") + e.what();
        PyErr_SetString(PyExc_RuntimeError, errstring.c_str());
        return NULL;
    }

}

/////////////////////////
///// HEADER DATA ///////
/////////////////////////

PyObject* SiffReader::readFixedData(){
    // returns the data in the primary ScanImage header, if it's been opened
    if (!siff.is_open()) {
        errstring = "No file open";
        PyErr_SetString(
            PyExc_AssertionError,
            "No open file in SiffReader C++ class."
        );
        return NULL;
    }

    PyObject* headerDict = PyDict_New();
    PyDict_SetItemString(headerDict, "Filename", Py_BuildValue("s", filename.c_str()));
    PyDict_SetItemString(headerDict, "BigTiff", Py_BuildValue("O", params.bigtiff ? Py_True : Py_False));
    PyDict_SetItemString(headerDict, "IsSiff",Py_BuildValue("O", params.issiff ? Py_True : Py_False));
    PyDict_SetItemString(headerDict, "Number of frames", Py_BuildValue("n", params.numFrames));
    PyDict_SetItemString(headerDict, "Non-varying frame data", Py_BuildValue("s#",params.headerstring.c_str(),Py_ssize_clean_t(params.NVFD_length)));
    PyDict_SetItemString(headerDict, "ROI string", Py_BuildValue("s#",params.ROI_string.c_str(),Py_ssize_clean_t(params.ROI_string_length)));
    if (debug) {
        PyDict_SetItemString(headerDict, "IFD pointers", Py_BuildValue("O",VectorToList(params.allIFDs)));
    }
    
    return headerDict;
}

///////////////////////////
////// GET/SET STYLE //////
///////////////////////////


std::string SiffReader::getNVFD() {
    return params.headerstring;
}

std::string SiffReader::getROIstring() {
    return params.ROI_string;
}

const char* SiffReader::getErrString(){
    return errstring.c_str();
}


//////////////////////////////////
//////// INTERNAL FUNCTIONS //////
//////////////////////////////////

void SiffReader::singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, PyObject* shift_tuple){
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Then appends that IFD to a list of numpy arrays
    FrameData frameData = getTagData(thisIFD, params, siff);
    // create a new numpy array of dimensions:
    // (y, x, tau)
    //const int ND = 2;
    const int ND = 2 + (params.issiff && flim); // number of dimensions
    npy_intp dims[ND];
    uint16_t tau_dim = 1024; // hardcoded for now. TODO: Implement this measure in SiffWriter

    dims[0] = frameData.imageLength;
    dims[1] = frameData.imageWidth;
    if (params.issiff & flim) dims[2] = tau_dim;

    PyArrayObject* numpyArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_UINT16, 
        0 // C order, i.e. last index increases fastest
    ); // Or should I make a sparse array? Maybe make that an option? TODO.

    loadArrayWithData(
        (uint16_t*) PyArray_DATA(numpyArray),
        PyArray_DIMS(numpyArray),
        params,
        frameData,
        siff,
        flim,
        shift_tuple
    );
    int ret = PyList_Append(numpyArrayList, (PyObject*) numpyArray);
    Py_DECREF(numpyArray);
    
    if (ret<0) throw std::runtime_error("Failure to append frame array to list");
}


void SiffReader::singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, 
    uint64_t terminalBin, PyObject* shift_tuple){
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Then appends that IFD to a list of numpy arrays
    if(!params.issiff) return singleFrameRetrieval(thisIFD, numpyArrayList, flim, shift_tuple);
    
    FrameData frameData = getTagData(thisIFD, params, siff);
    // create a new numpy array of dimensions:
    // (y, x, tau)
    //const int ND = 2;
    const int ND = 2 + (params.issiff && flim); // number of dimensions
    npy_intp dims[ND];
    uint16_t tau_dim = 1024; // hardcoded for now. TODO: Implement this measure in SiffWriter

    dims[0] = frameData.imageLength;
    dims[1] = frameData.imageWidth;
    if (params.issiff & flim) dims[2] = tau_dim;

    PyArrayObject* numpyArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_UINT16, 
        0 // C order, i.e. last index increases fastest
    ); // Or should I make a sparse array? Maybe make that an option? TODO.

    loadArrayWithData(
        (uint16_t*) PyArray_DATA(numpyArray),
        PyArray_DIMS(numpyArray),
        params,
        frameData,
        siff,
        flim,
        terminalBin,
        shift_tuple
    );
    //
    int ret = PyList_Append(numpyArrayList, (PyObject*) numpyArray);
    Py_DECREF(numpyArray);
    
    if (ret<0) throw std::runtime_error("Failure to append frame array to list");
}

PyArrayObject* SiffReader::frameAsNumpy(uint64_t thisIFD, bool flim, PyObject* shift_tuple) {
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Identical to above, except returns numpyArray instead of appending it to a list

    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

    FrameData frameData = getTagData(thisIFD, params, siff);
    
    // create a new numpy array of dimensions:
    // (y, x, tau)
    //const int ND = 2;
    const int ND = 2 + (params.issiff && flim); // number of dimensions
    npy_intp dims[ND];
    uint16_t tau_dim = 1024; // hardcoded for now. TODO: Implement this measure in SiffWriter

    dims[0] = frameData.imageLength;
    dims[1] = frameData.imageWidth;
    if (params.issiff & flim) dims[2] = tau_dim;

    PyArrayObject* numpyArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_UINT16, 
        0 // C order, i.e. last index increases fastest
    );

    loadArrayWithData(
        (uint16_t*) PyArray_DATA(numpyArray),
        PyArray_DIMS(numpyArray),
        params,
        frameData,
        siff,
        flim,
        shift_tuple
    );
    
    return numpyArray;
}

PyArrayObject* SiffReader::frameAsNumpy(uint64_t thisIFD, uint64_t terminalBins, bool flim, PyObject* shift_tuple) {
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Identical to above, except returns numpyArray instead of appending it to a list

    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

    FrameData frameData = getTagData(thisIFD, params, siff);
    
    // create a new numpy array of dimensions:
    // (y, x, tau)
    //const int ND = 2;
    const int ND = 2 + (params.issiff && flim); // number of dimensions
    npy_intp dims[ND];
    uint16_t tau_dim = 1024; // hardcoded for now. TODO: Implement this measure in SiffWriter

    dims[0] = frameData.imageLength;
    dims[1] = frameData.imageWidth;
    if (params.issiff & flim) dims[2] = tau_dim;

    PyArrayObject* numpyArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_UINT16, 
        0 // C order, i.e. last index increases fastest
    );

    loadArrayWithData(
        (uint16_t*) PyArray_DATA(numpyArray),
        PyArray_DIMS(numpyArray),
        params,
        frameData,
        siff,
        flim,
        terminalBins,
        shift_tuple
    );
    
    return numpyArray;
}

void SiffReader::singleFrameMetaData(uint64_t thisIFD, PyObject* metaDictList){
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Then appends that IFD to a list of numpy arrays
    // siff.clear();
    siff.seekg(thisIFD, siff.beg); // go there first
    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Siff unable to open frame. Likely error in preceding processing.");
    
    uint64_t numTags; // number of tags in this directory before the real metadata
    siff.read((char*) &numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.
    FrameData frameData;

    if (debug) {
        frameData.tagList = PyList_New(Py_ssize_t(0));
    }

    char thisTag[params.bytesPerTag];
    // tag parsing TODO: SHOULD BE INLINED SOMEWHERE ELSE
    for(uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
        siff.read(thisTag, params.bytesPerTag);
        // append info to frameData, defined in framedatastruct.hpp
        
        uint16_t tagID = ((uint8_t) thisTag[(1-params.little)]) + (thisTag[params.little] << 8 );
        // figure out the number of bytes needed to read the data correctly
        uint16_t datatype = (uint8_t) thisTag[(3-params.little)] + (((uint8_t) thisTag[2+params.little]) << 8);
        uint16_t contentChars = datatypeToCharCount(datatype); // defined in sifdefin

        if (tagID == IMAGEDESCRIPTION) contentChars = 8; // UGH this is to correct a mistake I made early on
        // TODO: DO THIS RIGHT. I ALREADY KNOW THEY ALL ONLY USE A SINGLE TAG VALUE HERE BUT I SHOULD
        // MAKE THIS WORK FOR _ALL_ TIFFS
        
        // convert to a single value
        
        uint64_t contentVals = 0;
        // 8 + 4*bigtiff corresponds to the 4 bytes of identifier + the 4 bytes for number of values in tag (or 8 if bigtiff)
        for(int16_t charnum = (contentChars-1); 0<=charnum; charnum--) {
            contentVals <<= 8;
            contentVals += (thisTag[charnum + 8 + 4*params.bigtiff] & 0xFF); // gotta be honest... I don't understand why the  & 0xFF is necessary. Cut me some slack I learned C 2 months ago.
        }

        // now correct the typing if it's wrong
        switch(tagID){
            case IMAGEWIDTH:
                frameData.imageWidth = contentVals;
                break;
            case IMAGELENGTH:
                frameData.imageLength = contentVals;
                break;
            case BITSPERSAMPLE:
                frameData.bitsPerSample = (uint16_t) contentVals;
                if (params.issiff) frameData.bitsPerSample = 64; // this is a given... for now.
                break;
            case COMPRESSION:
                frameData.compression = (uint16_t) contentVals;
                break;
            case PHOTOMETRIC_INTERPRETATION:
                frameData.photometric = (uint16_t) contentVals;
                break;
            case IMAGEDESCRIPTION:
                frameData.endOfIFD = contentVals;
                break;
            case STRIPOFFSETS:
                frameData.dataStripAddress = contentVals;
                break;
            case ORIENTATION:
                frameData.orientation = (uint16_t) contentVals;
                break;
            case SAMPLESPERPIXEL:
                frameData.samplesPerPixel = (uint16_t) contentVals;
                break;
            case ROWSPERSTRIP:
                frameData.rowsPerStrip = contentVals;
                break;
            case STRIPBYTECOUNTS:
                frameData.stripByteCounts = contentVals;
                break;
            case XRESOLUTION:
                frameData.xResolution = contentVals;
                break;
            case YRESOLUTION:
                frameData.yResolution = contentVals;
                break;
            case PLANARCONFIGURATION:
                frameData.planarConfig = (uint16_t) contentVals;
                break;
            case RESOLUTIONUNIT:
                frameData.resUnit = (uint16_t) contentVals;
                break;
            case SOFTWAREPACKAGE:
                frameData.NVFD_address = contentVals;
                break;
            case ARTIST:
                frameData.ROI_address = contentVals;
                break;
            case SAMPLEFORMAT:
                frameData.sampleFormat = (uint16_t) contentVals;
                break;
            case SIFFTAG:
                frameData.siffCompress = (bool) contentVals;
                break;
            default:
                if (params.suppress_warnings) break;
                PyErr_WarnEx(PyExc_RuntimeWarning,
                    (std::string("INVALID TIFF TAG DETECTED: ") + std::to_string(tagID) +  "\n" +
                    std::string("To suppress this warning in the future, call siffreader.suppress_warnings()")).c_str(),
                    Py_ssize_t(1)
                );
                //throw std::runtime_error(std::string("INVALID TIFF TAG DETECTED: ") + std::to_string(tagID));
        }
        siff.clear();
        
        if (debug) {
            PyObject* tempVal = Py_BuildValue("y#",thisTag, Py_ssize_t(params.bytesPerTag));
            PyList_Append(frameData.tagList, tempVal);
            Py_DECREF(tempVal); // Append adds a reference
        }
    }


    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Failure to discern description string");
    if(!debug && (frameData.dataStripAddress < frameData.endOfIFD)) throw std::runtime_error("Negative description length -- error parsing tags?");
    
    uint64_t description_length = frameData.siffCompress ?
        frameData.dataStripAddress - frameData.endOfIFD - frameData.imageLength*frameData.imageWidth*sizeof(uint16_t)
        :
        frameData.dataStripAddress - frameData.endOfIFD;

    frameData.stringlength = description_length;
    siff.seekg(frameData.endOfIFD, siff.beg);
    char metaString[description_length];
    siff.read(metaString, description_length);
    frameData.frameMetaData = std::string(metaString);
    PyObject* frameDict = frameDataToDict(frameData);
    PyList_Append(metaDictList, frameDict); // append adds a reference
    Py_DECREF(frameDict);
}//

void SiffReader::singleFrameHistogram(uint64_t thisIFD, PyArrayObject* numpyArray){
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.

    FrameData frameData = getTagData(thisIFD, params, siff);
//
    addArrivalsToArray(numpyArray, params, frameData, siff);
}

PyObject* SiffReader::makeFlimTuple(uint64_t thisIFD, PyObject* shift_tuple){
    // Makes a tuple of SUMMED arrival times (in units of bins) and intensity

    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

    FrameData frameData = getTagData(thisIFD, params, siff);
    
    const int ND = 2; // number of dimensions
    npy_intp dims[ND];

    dims[0] = frameData.imageLength;
    dims[1] = frameData.imageWidth;

    PyArrayObject* lifetimeArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_FLOAT64, 
        0 // C order, i.e. last index increases fastest
    );

    PyArrayObject* intensityArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_UINT16, 
        0 // C order, i.e. last index increases fastest
    );

    loadArrayWithSummedArrivalTimes(lifetimeArray, intensityArray, params, frameData, siff, shift_tuple);
    //
    return PyTuple_Pack(Py_ssize_t(2), lifetimeArray, intensityArray);

}

void SiffReader::fuseIntoFlimTuple(PyObject* FlimTup, uint64_t nextIFD, PyObject* shift_tuple){
    
    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

    FrameData frameData = getTagData(nextIFD, params, siff);

    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Failure to navigate to data in frame.");

    PyArrayObject* lifetimeArray = (PyArrayObject*) PyTuple_GetItem(FlimTup, Py_ssize_t(0));
    PyArrayObject* intensityArray = (PyArrayObject*) PyTuple_GetItem(FlimTup, Py_ssize_t(1));

    loadArrayWithSummedArrivalTimes(lifetimeArray, intensityArray, params, frameData, siff, shift_tuple);
    //
}

void SiffReader::fuseFrames(PyArrayObject* fuseFrame, uint64_t nextIFD, bool flim, PyObject* shift_tuple) {
    
    FrameData frameData = getTagData(nextIFD, params, siff);

    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Failure to navigate to data in frame.");

    loadArrayWithData(
        (uint16_t*) PyArray_DATA(fuseFrame), 
        PyArray_DIMS(fuseFrame),
        params,
        frameData,
        siff,
        flim,
        shift_tuple
    );
}

void SiffReader::fuseFrames(PyArrayObject* fuseFrame, uint64_t nextIFD, bool flim, uint64_t terminalBins, PyObject* shift_tuple) {
    
    FrameData frameData = getTagData(nextIFD, params, siff);

    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Failure to navigate to data in frame.");

    loadArrayWithData(
        (uint16_t*) PyArray_DATA(fuseFrame),
        PyArray_DIMS(fuseFrame),
        params,
        frameData,
        siff,
        flim,
        terminalBins,
        shift_tuple
    );
}

void SiffReader::fuseReadVector(std::vector<uint64_t>& photonReadsTogether, uint64_t nextIFD, PyObject* shift_tuple) {
    FrameData frameData = getTagData(nextIFD, params, siff);

    std::vector<uint64_t> frameReads = // this frame's photon counts
        frameData.siffCompress ? 
            compressedReadsToVec(frameData, siff, shift_tuple) :
            uncompressedReadsToVec(frameData, siff, shift_tuple);

    photonReadsTogether.insert(photonReadsTogether.end(), frameReads.begin(), frameReads.end());
}

void SiffReader::reset() {
    
    if (siff.is_open()) siff.close();
    params = SiffParams();
    filename = std::string();
}
