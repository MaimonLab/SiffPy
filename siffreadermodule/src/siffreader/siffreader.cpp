// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API
#define PY_SSIZE_T_CLEAN
#include "../../include/siffreader/siffreader.hpp"
#include "../../include/siffreader/siffreaderinline.hpp"

#include "../../include/framedata/sifdefin.hpp"
#include <algorithm>

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

        // Gotta know the endian
        char *endian = new char[2];
        siff.read(endian, sizeof(char)*2);   

        // strcmp == 0 if they match.
        if ((strcmp(endian,BIGENDIAN)*strcmp(endian,LITTLEENDIAN)) != 0){
            throw std::runtime_error(
                std::string("Could not deduce endian. May not be .siff/.tiff file.")+
                " First two bytes (should be II or MM): "+
                std::string(endian)
            );
        }
        params.little = (strcmp(endian,LITTLEENDIAN) == 0); // true if little, false if big.

        delete[] endian;

        // temporary solution: if endian isn't little, give up.
        uint16_t i = 1; // the uint16_t 1 is 0x01 in big endian, 0x10 in little endian
        char* c = (char*)&i;
        // dereferencing c will be 1 if the least significant byte is first. 0 if not.
        if( (bool)*c != params.little ) throw std::runtime_error("ENDIANS DON'T MATCH AND I HAVEN'T FIXED THAT YET.");
        
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
            
            siff.read((char*)&(params.firstIFDAddress),params.bytesPerPointer);
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

        siff.read((char*)&(params.NVFD_length), sizeof(uint32_t));

        siff.read((char*)&(params.ROI_string_length), sizeof(uint32_t));

        
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

// frameDataList has to be a list
void SiffReader::packFrameDataList(PyObject* frameDataList){
    if (PyList_Size(frameDataList) != numFrames()) {
        PyErr_SetString(PyExc_ValueError, "FrameDataList must be the same length as the number of frames in the SiffReader.");
        return;
    }
    //for (Py_ssize_t k = 0; k < numFrames(); k++) {
    //    PyFrameData* frameData = (PyFrameData*) PyObject_CallObject((PyObject*)&PyFrameDataType, NULL);
    //    frameData->framedatastruct = &frameDatas[k];
    //    PyList_SetItem(frameDataList, k, (PyObject*) frameData);
    //}
}

bool SiffReader::dimensionsConsistent(const uint64_t frames[], const uint64_t framesN){
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

void SiffReader::reset() {
    if (siff.is_open()) siff.close();
    params = SiffParams();
    filename = std::string();
}

void SiffReader::suppressWarnings(bool suppress){
    suppress_warnings = suppress;
    params.suppress_warnings = suppress;
}

uint64_t SiffReader::numFrames(){
    return siff.is_open() ? params.allIFDs.size() : -1;
}

void SiffReader::setDebug(bool debug_bool){
    debug = debug_bool;
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

            const FrameData frameData = getTagData(params.allIFDs[frames[frame_idx]], params, siff);

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

            const FrameData frameData = getTagData(params.allIFDs[frames[frame_idx]], params, siff);

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

void SiffReader::singleFrameMetaData(uint64_t thisIFD, PyObject* metaDictList){
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Then appends that IFD to a list of numpy arrays
    // siff.clear();
    FrameData frameData = getTagData(
        thisIFD,
        params,
        siff
    );


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

void SiffReader::fuseReadVector(std::vector<uint64_t>& photonReadsTogether, uint64_t nextIFD, PyObject* shift_tuple) {
    const FrameData frameData = getTagData(nextIFD, params, siff);

    std::vector<uint64_t> frameReads = // this frame's photon counts
        frameData.siffCompress ? 
            compressedReadsToVec(frameData, siff, shift_tuple) :
            uncompressedReadsToVec(frameData, siff, shift_tuple);

    photonReadsTogether.insert(photonReadsTogether.end(), frameReads.begin(), frameReads.end());
}
