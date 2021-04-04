// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#include "siffreader.hpp"

#include "../include/sifdefin.hpp"
#include "../include/framedatastruct.hpp"
#include "../include/siffreaderinline.hpp"
#include <algorithm>

///////////////////////
////// FILE I/O ///////
///////////////////////

// BIG TIME TODO: MAKE THIS CROSS-ENDIAN COMPATIBLE.
int SiffReader::openFile(const char* _filename) {
    // Opens a .siff file for further analysis / use.
    
    try{
        // First make sure we can open it at all.
        if(siff.is_open()) {
            if (filename == _filename) {
                errstring = std::string("This file is already open. Was that an accident?");
                return -2;
            }
            siff.close();
        }
        siff.open(_filename, std::ios::binary | std::ios::in);
        if (!(siff.is_open())) throw std::runtime_error("Could not open putative .siff file. Check that path exists.\nAttempted path: "+std::string(_filename));
        // Now check that it's a siff file.
        
        // Gotta know the endianness
        char * endian = new char[2];
        siff.read(endian, sizeof(char)*2);   

        // strcmp == 0 if they match.
        if ((strcmp(endian,BIGENDIAN) !=0) && (strcmp(endian,LITTLEENDIAN) != 0)) throw std::runtime_error("Could not deduce endian. May not be .siff/.tiff file. First two bytes (should be II or MM): "+std::string(endian));
        params.little = (strcmp(endian,LITTLEENDIAN) == 0); // true if little, false if big.

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


void SiffReader::discernFrames() {
    // Runs through all the frames to find their IFD, calculates the total number of frames,
    // and stores all the IFDs for quick lookup.
    uint64_t nextIFD = params.firstIFDAddress;
    uint64_t currIFD = 0;
    while(nextIFD && !(currIFD == nextIFD)) {
        // iterates through all the IFDs until it gets to the end
        siff.seekg(nextIFD); // go there first
        if(!(siff.good()||suppress_warnings)) throw std::runtime_error("Failed to reach IDF during load. Possible corrupt frame?");
        currIFD = nextIFD;
        params.allIFDs.push_back(currIFD);

        uint64_t numTags; // number of tags in this directory before the real metadata
        siff.read((char*)&numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.

        siff.seekg(numTags*params.bytesPerTag,std::ios::cur); // skip the tags

        siff.read((char*)&nextIFD, params.bytesPerPointer);
    }
    params.allIFDs.pop_back(); // this is for the case that the last IFD is not 0ULL
    params.numFrames = params.allIFDs.size();
    siff.clear(); // get rid of failbits
}


void SiffReader::closeFile(){
    if (siff.is_open()) siff.close();
    params = SiffParams();
    filename = std::string();
}


////////////////////////////
///// GET FILE DATA ////////
////////////////////////////


PyObject* SiffReader::retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim) {
    // By default, retrieves ALL frames, returns as a list of numpy arrays including the arrival times.
    // TODO: Implement frame selection, automatically detect .tiffs to make flim=false, implement
    // the variable type of output
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();
        // create the list into which we shall stuff the numpy arrays
        PyObject* numpyArrayList = PyList_New(Py_ssize_t(0));

        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                singleFrameRetrieval(params.allIFDs[frames[i]], numpyArrayList, flim);
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                singleFrameRetrieval(params.allIFDs[i], numpyArrayList, flim);
            }
        }
        return numpyArrayList;
    }
    catch(std::exception& e){
        errstring = std::string("Error parsing frames: ") + e.what();
        throw e;
    }
}

PyObject* SiffReader::poolFrames(PyObject* listOfLists, bool flim) {
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
                Py_INCREF(Py_None);
                PyList_Append(returnedList,Py_None);
            }
            // get the first frame requested.
            PyArrayObject* firstFrame = frameAsNumpy(params.allIFDs[PyLong_AsLongLong(PyList_GetItem(listOfFrames,0))], flim);
            
            // fuse in more if they asked for it.
            if(PyList_Size(listOfFrames) > Py_ssize_t(1)) {
                // more than one frame in the list
                for(Py_ssize_t frameIdx(1); frameIdx < PyList_Size(listOfFrames); frameIdx++) {
                    fuseFrames(
                        firstFrame, // template fused onto
                        params.allIFDs[ PyLong_AsLongLong(PyList_GetItem(listOfFrames,frameIdx)) ], // next frame to add
                        flim // what style to fuse it onto
                    ); 
                }
            }
            
            PyList_Append(returnedList, (PyObject*) firstFrame);
        }
        return returnedList;
    }
    catch(std::exception& e) {
        errstring = std::string("Error in pool frames: ") + e.what();
        throw e;
    }
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
        return NULL;
    }

    PyObject* headerDict = PyDict_New();
    PyDict_SetItemString(headerDict, "Filename", Py_BuildValue("s", filename.c_str()));
    PyDict_SetItemString(headerDict, "BigTiff", Py_BuildValue("O", params.bigtiff ? Py_True : Py_False));
    PyDict_SetItemString(headerDict, "IsSiff",Py_BuildValue("O", params.issiff ? Py_True : Py_False));
    PyDict_SetItemString(headerDict, "Number of frames", Py_BuildValue("n", params.numFrames));
    PyDict_SetItemString(headerDict, "Non-varying frame data", Py_BuildValue("s#",params.headerstring.c_str(),params.NVFD_length));
    PyDict_SetItemString(headerDict, "ROI string", Py_BuildValue("s#",params.ROI_string.c_str(),params.ROI_string_length));
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

void SiffReader::singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim){
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

    loadArrayWithData(numpyArray, params, frameData, siff, flim);
    
    int ret = PyList_Append(numpyArrayList, (PyObject*) numpyArray);
    
    if (ret<0) std::runtime_error("Failure to append frame array to list");
}

PyArrayObject* SiffReader::frameAsNumpy(uint64_t thisIFD, bool flim) {
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Identical to above, except returns numpyArray instead of appending it to a list

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

    loadArrayWithData(numpyArray, params, frameData, siff, flim);
    
    return numpyArray;
}

void SiffReader::singleFrameMetaData(uint64_t thisIFD, PyObject* metaDictList){
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Then appends that IFD to a list of numpy arrays
    siff.clear();
    siff.seekg(thisIFD, std::ios::beg); // go there first
    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Siff unable to open frame. Likely error in preceding processing.");

    uint64_t numTags; // number of tags in this directory before the real metadata
    siff.read((char*) &numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.
    FrameData frameData;
   
    if (debug) {
        frameData.tagList = PyList_New(Py_ssize_t(0));
    }
    char tagBuffer[params.bytesPerTag];
    for(uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
        siff.read(tagBuffer, params.bytesPerTag);
        // append info to frameData, defined in framedatastruct.hpp
        parseTags(tagBuffer,frameData,params);
        PyList_Append(frameData.tagList, Py_BuildValue("y#",tagBuffer, Py_ssize_t(params.bytesPerTag)));
    }
    
    siff.clear();
    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Failure to discern description string");
    
    if(!debug && (frameData.dataStripAddress < frameData.endOfIFD)) throw std::runtime_error("Negative description length -- error parsing tags?");
    uint64_t description_length = frameData.dataStripAddress - frameData.endOfIFD;
    
    frameData.stringlength = description_length;
    siff.seekg(frameData.endOfIFD, std::ios::beg);
    siff.clear();
    char metaString[description_length];
    siff.read(metaString, description_length);
    siff.clear();
    frameData.frameMetaData = std::string(metaString);
    PyList_Append(metaDictList, frameDataToDict(frameData));
}

void SiffReader::fuseFrames(PyArrayObject* fuseFrame, uint64_t nextIFD, bool flim) {
    
    uint16_t* data_ptr = (uint16_t*) PyArray_DATA(fuseFrame);
    npy_intp* dims = PyArray_DIMS(fuseFrame);
    
    FrameData frameData = getTagData(nextIFD, params, siff);

    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || suppress_errors)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;

    loadArrayWithData(fuseFrame, params, frameData, siff, flim);
}