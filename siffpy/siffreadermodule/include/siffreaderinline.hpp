#ifndef SIFFREADERINLINE_HPP
#define SIFFREADERINLINE_HPP


// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes

#include "siffParams.hpp"
#include <numpy/arrayobject.h>

#include "framedatastruct.hpp"
#include "siffParams.hpp"
#include <stdio.h>
#include <fstream>

// for parsing the uint64 format
#define YMASK ((uint64_t) 1 << 63) - ((uint64_t) 1 << 48) + ((uint64_t) 1 << 63) // that last bit'll getcha
#define XMASK ((uint64_t) 1 << 48) - ((uint64_t) 1 << 32)
#define TAUMASK (((uint64_t) 1<<32) - 1)

#define U64TOY(photon) (uint64_t) ((photon & YMASK) >> 48)
#define U64TOX(photon) (uint64_t) ((photon & XMASK) >> 32)
#define U64TOTAU(photon) (uint64_t) (photon & TAUMASK)

///////////////////////////////////
////// INLINE FUNCTIONS ///////////
///////////////////////////////////

inline FrameData getTagData(uint64_t IFD, SiffParams& params, std::ifstream& siff){
    // return a FrameData structure that has parsed all the tag information at location:
    // IFD

    siff.clear();
    siff.seekg(IFD); // go there first

    if (!(siff.good())) throw std::runtime_error("Failure to find frame.");

    uint64_t numTags; // number of tags in this directory before the real metadata
    siff.read((char*)&numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.
    siff.clear();
    FrameData frameData;
    for(uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
        char thisTag[params.bytesPerTag];
        siff.read(thisTag, params.bytesPerTag);
        // append info to frameData, defined in framedatastruct.hpp
        parseTags(thisTag,frameData,params);
    }

    if (frameData.dataStripAddress<frameData.endOfIFD) throw std::runtime_error("Invalid data strip address read.");    
    siff.clear();
    return frameData;
}

inline void loadArrayWithData(PyArrayObject* numpyArray, SiffParams& params, FrameData& frameData, std::ifstream& siff, bool flim){
    // TODO: put the bool flags into the SiffParams struct.
    uint16_t* data_ptr = (uint16_t *) PyArray_DATA(numpyArray);
    npy_intp* dims = PyArray_DIMS(numpyArray);
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    if (params.issiff) {
        uint64_t frameReads[samplesThisFrame];
        siff.read((char*)&frameReads,frameData.stripByteCounts);
        siff.clear();
        if(flim) {
            for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
                // increment the appropriate numpy element
                data_ptr[
                    (U64TOY(frameReads[photon])*dims[1]*dims[2])
                        + (std::min<uint16_t>(U64TOX(frameReads[photon]),dims[1]-1)*dims[2]) // THIS SHOULD NOT BE HERE. 
                        //ONLY NECESSARY DUE TO AN EARLY ERROR IN CODE. TODO REMOVE THIS STD::MIN CALL!!
                            + U64TOTAU(frameReads[photon])
                ]++;
            }
        }
        else{
            // if we're just doing intensity, the pointer element to increment should be different
            for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
                    // increment the appropriate numpy element
                data_ptr[U64TOY(frameReads[photon])*dims[1] + U64TOX(frameReads[photon])]++;
            }
        }
    }
    else {
        // pretty simple -- it's already formatted right.
        if(!(samplesThisFrame == (dims[0]*dims[1]))) throw std::runtime_error("Frame data dimensions don't match frame metadata");
        // now that we're sure, just write the data into the data pointer!
        // willing to eat a small slowdown for compactness, instead of reading
        // directly into the numpy array ptr, we'll add to the ptr.
        // this lets us use the same function for fusing dating into an array
        // but has the cost of storing an evanescent frame buffer.
        uint16_t frameReads[samplesThisFrame];
        siff.read((char*)&frameReads,frameData.stripByteCounts);
        siff.clear();
        for(uint64_t px = 0; px < samplesThisFrame; px++) {
            data_ptr[px] += frameReads[px];
        }
    }
}

#endif