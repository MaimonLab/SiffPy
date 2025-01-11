#ifndef FRAMEDATASTRUCT_HPP
#define FRAMEDATASTRUCT_HPP

#ifndef PY_SSIZE_T_CLEAN
    #define PY_SSIZE_T_CLEAN
#endif

#include <Python.h>
#include <cstdint>
#include <stdlib.h>
#include <string>
#include "sifdefin.hpp"
#include "../siffparams/siffparams.hpp"
#include "../debug.hpp"

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

/*
Reads the header data for a given frame from the input file.

@param IFD
    The IFD pointer whose contents are to be read

@param params
    The `SiffParams` struct containing the main file parameters of the file
    being read from.

@param siff
    The input file from which the header data should be read. Must be
    positioned at the IFD that is to correspond to the given `IFD`.

@return A `FrameData` struct containing the header data for the given frame.
*/
const FrameData getTagData(const uint64_t IFD, const SiffParams& params, std::ifstream& siff);

const FrameData getTagData(const uint64_t IFD, const SiffParams& params, std::ifstream& siff, std::ofstream& logstream);


/*
Writes the header data for a given frame to the output file.

@param frameData
    The `FrameData` whose contents should be translated to the head
    of the output file.

@param outfile
    The output file to which the header data should be written. Must be
    positioned at the IFD that is to correspond to the given `FrameData`.

@param params
    The `SiffParams` struct containing the main file parameters of the file
    being read from.
*/
void writeFrameDataAsTiff(const FrameData& frameData, std::ofstream& outfile, const SiffParams& params);

/*
Returns a string containing the OME-XML specification given a frame's metadata
and the file's metadata

@param frameData
    The `FrameData` struct containing the metadata for the frame to be
    converted to OME-XML.

@param params
    The `SiffParams` struct containing the main file parameters of the file
    being read from.

@return A string containing the OME-XML specification for the given frame to
be copied into the IMAGEDESCRIPTION tag of the output file.
*/
const std::string toOMEXML(const FrameData& frameData, const SiffParams& params);

/**
 * Converts a `FrameData` struct to a Python dictionary.
 *
 * @param frameData
 *  The `FrameData` struct to be converted to a Python dictionary.
 *
 * @return A Python dictionary containing the contents of the given `FrameData`.
 * All keys are `str`
*/
PyObject* frameDataToDict(FrameData& frameData);

/**
 * Converts a `FrameData` struct to a Python dictionary. Logs
 * the conversion to a given output stream.
 * 
 * @param frameData
 * The `FrameData` struct to be converted to a Python dictionary.
 * 
 * @param logstream
 * The output stream to which the conversion should be logged.
 * 
 * @return A Python dictionary containing the contents of the given `FrameData`.
 * All keys are `str`
*/
PyObject* frameDataToDict(FrameData& frameData, std::ofstream& logstream);

template <class T>
PyObject* VectorToList(std::vector<T> vec) {
    PyObject* list = PyList_New(vec.size());
    for(uint64_t idx = 0; idx < vec.size(); idx++){
        PyList_SetItem(list, idx, Py_BuildValue("K",vec[idx]));
    }
    return list;
};

#endif