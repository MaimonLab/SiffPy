#ifndef FRAMEDATASTRUCT_HPP
#define FRAMEDATASTRUCT_HPP

#include <Python.h>
#include <stdlib.h>
#include <string>
#include "sifdefin.hpp"
#include "../siffparams/siffparams.hpp"

typedef struct FrameData{
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
} FrameData ;

const FrameData getTagData(const uint64_t IFD, SiffParams& params, std::ifstream& siff);

PyObject* frameDataToDict(FrameData& frameData);

template <class T>
PyObject* VectorToList(std::vector<T> vec) {
    PyObject* list = PyList_New(vec.size());
    for(uint64_t idx = 0; idx < vec.size(); idx++){
        PyList_SetItem(list, idx, Py_BuildValue("K",vec[idx]));
    }
    return list;
};

#endif