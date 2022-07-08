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
    for(uint64_t idx = 0; idx < vec.size(); idx++){
        PyList_SetItem(list, idx, Py_BuildValue("K",vec[idx]));
    }
    return list;
}

#endif