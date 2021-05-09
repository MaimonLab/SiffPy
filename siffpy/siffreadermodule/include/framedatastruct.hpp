#ifndef FRAMEDATASTRUCT_HPP
#define FRAMEDATASTRUCT_HPP

#include <stdlib.h>
#include <string>
#include "sifdefin.hpp"

struct FrameData{
    uint64_t imageWidth;
    uint64_t imageLength;
    uint16_t bitsPerSample;
    uint16_t compression;
    uint16_t photometric;
    uint64_t endOfIFD;
    uint64_t dataStripAddress;
    uint16_t orientation;
    uint16_t samplesPerPixel;
    uint64_t rowsPerStrip;
    uint64_t stripByteCounts;
    uint64_t xResolution;
    uint64_t yResolution;
    uint16_t planarConfig;
    uint16_t resUnit;
    uint64_t NVFD_address;
    uint64_t ROI_address;
    uint16_t sampleFormat;
    std::string frameMetaData; // only populated when retrieving metadata
    bool siffCompress = false; // only present in .siff files and only in newer versions

    uint64_t stringlength;
    PyObject* tagList;
    // TODO: ADD A TIMESTAMP OR MAYBE OTHER NICE THINGS
};

void parseTags(const char* thisTag, FrameData& frameData, SiffParams params) {
    // Takes an array of tag chars and parses them to assign the appropriate value to frameData properties
    // Format of a tag:
    // 2 bytes: tag identifier (in sifdefin.hpp)
    // 2 bytes: datatype (also in sifdefin.hpp)
    // 4 bytes (if TIFF), 8 bytes (if BigTiff/Siff): number of values in tag
    // 4 bytes (if TIFF), 8 bytes (if BigTiff/Siff): each individual value OR an offset pointer to the value(s)

    // LSB + MSB << 8
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
    return;
    
}

PyObject* frameDataToDict(FrameData& frameData){
    PyObject* dataDict = PyDict_New();

    PyDict_SetItemString(dataDict, "Width", Py_BuildValue("K", frameData.imageWidth));
    PyDict_SetItemString(dataDict, "Length", Py_BuildValue("K", frameData.imageLength));
    PyDict_SetItemString(dataDict, "endOfIFD", Py_BuildValue("K", frameData.endOfIFD));
    PyDict_SetItemString(dataDict, "dataStripAddress",Py_BuildValue("K", frameData.dataStripAddress));
    PyDict_SetItemString(dataDict, "stringLength", Py_BuildValue("K", frameData.stringlength));
    PyDict_SetItemString(dataDict, "X Resolution", Py_BuildValue("K", frameData.xResolution));
    PyDict_SetItemString(dataDict, "YResolution", Py_BuildValue("K", frameData.yResolution));
    PyDict_SetItemString(dataDict, "Bytecount", Py_BuildValue("K", frameData.stripByteCounts));
    PyDict_SetItemString(dataDict, "Frame metadata", Py_BuildValue("s#", frameData.frameMetaData.c_str(), frameData.stringlength));
    //PyDict_SetItemString(dataDict, "Tag bytes", Py_BuildValue("O",frameData.tagList));
    PyDict_SetItemString(dataDict, "Siff compression", Py_BuildValue("O",frameData.siffCompress ? Py_True : Py_False));
    return dataDict;
}

template <class T>
PyObject* VectorToList(std::vector<T> vec) {
    PyObject* list = PyList_New(vec.size());
    for(int idx = 0; idx < vec.size(); idx++){
        PyList_SetItem(list, idx, Py_BuildValue("K",vec[idx]));
    }
    return list;
}

#endif