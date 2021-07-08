#ifndef SIFFREADERINLINE_HPP
#define SIFFREADERINLINE_HPP


// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes

#include "siffParams.hpp"
#include <numpy/arrayobject.h>
#include "exp_math.hpp"

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


inline void readCompressed(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    uint16_t* data_ptr, bool flim, npy_intp* dims, PyObject* shift_tuple) // uses siffCompress
    {
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));
    
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
    if(!flim) { // easy, this data's already stored in the start.

        uint16_t frameReads[samplesThisFrame];
        siff.read((char*)&frameReads,pixelsInImage * sizeof(uint16_t));
        siff.clear();

        for(uint64_t px = 0; px < samplesThisFrame; px++) {
            uint64_t shifted_px = ( (((uint64_t)(px / dims[0])) + y_shift) % dims[0]) * dims[1]; // y_val
            shifted_px += (px % dims[0] + x_shift) % dims[1];
            data_ptr[shifted_px] += frameReads[px];
        }

        return;
    }
    
    // first get the number of photons for each pixel
    uint16_t photonCounts[frameData.imageLength * frameData.imageWidth];
    siff.read((char*)&photonCounts, pixelsInImage * sizeof(uint16_t));

    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (uint64_t px = 0; px < pixelsInImage; px++) {
        uint16_t photonsThisPx = photonCounts[px];
        uint16_t pxReads[photonsThisPx];
        siff.read((char*)&pxReads, photonsThisPx*sizeof(uint16_t));

        for (uint16_t readNum = 0; readNum < photonsThisPx; readNum++) {
            data_ptr[
                ((px / frameData.imageWidth + y_shift) % dims[0]) * dims[1] * dims[2] // y index
                    +((px % frameData.imageWidth + x_shift) % dims[1]) *dims[2]       // x index
                        + pxReads[readNum]                      // tau index
            ]++;
        }
        
    }
}

inline void readCompressed(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    uint16_t* data_ptr, bool flim, npy_intp* dims, PyObject* shift_tuple, uint64_t terminalBin) // uses siffCompress
    { // TODO : FINSIH ME!!!
    
    throw std::runtime_error("Not yet implemented (compressed reads with a terminal bin.");
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));
    
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
    if(!flim) { // easy, this data's already stored in the start.

        uint16_t frameReads[samplesThisFrame];
        siff.read((char*)&frameReads,pixelsInImage * sizeof(uint16_t));
        siff.clear();

        for(uint64_t px = 0; px < samplesThisFrame; px++) {
            uint64_t shifted_px = ( (((uint64_t)(px / dims[0])) + y_shift) % dims[0]) * dims[1]; // y_val
            shifted_px += (px % dims[0] + x_shift) % dims[1];
            data_ptr[shifted_px] += frameReads[px];
        }

        return;
    }
    
    // first get the number of photons for each pixel
    uint16_t photonCounts[frameData.imageLength * frameData.imageWidth];
    siff.read((char*)&photonCounts, pixelsInImage * sizeof(uint16_t));

    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (uint64_t px = 0; px < pixelsInImage; px++) {
        uint16_t photonsThisPx = photonCounts[px];
        uint16_t pxReads[photonsThisPx];
        siff.read((char*)&pxReads, photonsThisPx*sizeof(uint16_t));

        for (uint16_t readNum = 0; readNum < photonsThisPx; readNum++) {
            data_ptr[
                ((px / frameData.imageWidth + y_shift) % dims[0]) * dims[1] * dims[2] // y index
                    +((px % frameData.imageWidth + x_shift) % dims[1]) *dims[2]       // x index
                        + pxReads[readNum]                      // tau index
            ]++;
        }
        
    }
}

inline void readCompressedForArrivals(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    double_t* lifetime_ptr, uint16_t* intensity_ptr, npy_intp* dims, PyObject* shift_tuple) // uses siffCompress
    {
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));
    
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
    
    // first get the number of photons for each pixel
    uint16_t photonCounts[frameData.imageLength * frameData.imageWidth];
    siff.read((char*)&photonCounts, pixelsInImage * sizeof(uint16_t));

    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (uint64_t px = 0; px < pixelsInImage; px++) {
        uint16_t photonsThisPx = photonCounts[px];

        uint64_t shifted_px = ( (((uint64_t)(px / dims[0])) + y_shift) % dims[0]) * dims[1]; // y_val
        shifted_px += (px % dims[0] + x_shift) % dims[1];

        intensity_ptr[shifted_px] += photonsThisPx;

        uint16_t pxReads[photonsThisPx];
        siff.read((char*)&pxReads, photonsThisPx*sizeof(uint16_t));

        for (uint16_t readNum = 0; readNum < photonsThisPx; readNum++) {
            lifetime_ptr[shifted_px] += pxReads[readNum];
        }
    }
}


inline void readRaw(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    uint16_t* data_ptr, bool flim, npy_intp* dims, PyObject* shift_tuple) // every read is a uint64 of y, x, tau
    {
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    uint64_t frameReads[samplesThisFrame];
    siff.read((char*)&frameReads,frameData.stripByteCounts);
    siff.clear();
    if(flim) {
        for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
            // increment the appropriate numpy element
            data_ptr[
                (((U64TOY(frameReads[photon]) + y_shift) % dims[0])*dims[1]*dims[2])
                    + (((U64TOX(frameReads[photon]) + x_shift) % dims[1]) * dims[2])
                        + U64TOTAU(frameReads[photon])
            ]++;
        }
    }
    else{
        // if we're just doing intensity, the pointer element to increment should be different
        for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
                // increment the appropriate numpy element
            data_ptr[((U64TOY(frameReads[photon]) + y_shift) % dims[0])*dims[1] + 
                ((U64TOX(frameReads[photon]) + x_shift)%dims[1])]++;
        }
    }
}

inline void readRaw(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    uint16_t* data_ptr, bool flim, npy_intp* dims, PyObject* shift_tuple, uint64_t terminalBin) // every read is a uint64 of y, x, tau
    {
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    uint64_t frameReads[samplesThisFrame];
    siff.read((char*)&frameReads,frameData.stripByteCounts);
    siff.clear();
    if(flim) {
        for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
            // increment the appropriate numpy element
            if (U64TOTAU(frameReads[photon]) > terminalBin) continue;
            
            data_ptr[
                (((U64TOY(frameReads[photon]) + y_shift) % dims[0])*dims[1]*dims[2])
                    + (((U64TOX(frameReads[photon]) + x_shift) % dims[1]) * dims[2])
                        + U64TOTAU(frameReads[photon])
            ]++;
        }
    }
    else{
        // if we're just doing intensity, the pointer element to increment should be different
        for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
            if (U64TOTAU(frameReads[photon]) > terminalBin) continue;
                // increment the appropriate numpy element
            data_ptr[((U64TOY(frameReads[photon]) + y_shift) % dims[0])*dims[1] + 
                ((U64TOX(frameReads[photon]) + x_shift)%dims[1])]++;
        }
    }
}


inline void readRawForArrivals(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    double_t* lifetime_ptr, uint16_t* intensity_ptr, npy_intp* dims, PyObject* shift_tuple) // every read is a uint64 of y, x, tau
    {
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    uint64_t frameReads[samplesThisFrame];
    siff.read((char*)&frameReads,frameData.stripByteCounts);
    siff.clear();

    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increase this pixel's value by ARRIVAL_TIME
        lifetime_ptr[((U64TOY(frameReads[photon]) + y_shift) % dims[0])*dims[1] + 
            ((U64TOX(frameReads[photon]) + x_shift)%dims[1])] += U64TOTAU(frameReads[photon]);
        // increase this pixel's value by 1 for intensity increase
        intensity_ptr[((U64TOY(frameReads[photon]) + y_shift) % dims[0])*dims[1] + 
            ((U64TOX(frameReads[photon]) + x_shift)%dims[1])]++;
    }

}

inline void loadArrayWithData(PyArrayObject* numpyArray,
    SiffParams& params, FrameData& frameData, std::ifstream& siff,
    bool flim, PyObject* shift_tuple = NULL) {
    
    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

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
        frameData.siffCompress ?
            readCompressed(samplesThisFrame, frameData, siff, data_ptr, flim, dims, shift_tuple) : 
            readRaw(samplesThisFrame, frameData, siff, data_ptr, flim, dims, shift_tuple);
    }
    else {
        // pretty simple -- it's already formatted right.
        if(!(samplesThisFrame == uint64_t(dims[0]*dims[1]))) throw std::runtime_error("Frame data dimensions don't match frame metadata");
        // now that we're sure, just write the data into the data pointer!
        // willing to eat a small slowdown for compactness, instead of reading
        // directly into the numpy array ptr, we'll add to the ptr.
        // this lets us use the same function for fusing dating into an array
        // but has the cost of storing an evanescent frame buffer.
        uint16_t frameReads[samplesThisFrame];
        siff.read((char*)&frameReads,frameData.stripByteCounts);
        siff.clear();

        // figure out the rigid deformation for registration
        uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
        uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

        for(uint64_t px = 0; px < samplesThisFrame; px++) {
            uint64_t shifted_px = (( ((uint64_t)(px / dims[0])) + y_shift) % dims[0]) * dims[1]; // y_val
            shifted_px += (px % dims[0] + x_shift) % dims[1];
            data_ptr[shifted_px] += frameReads[px];
        }
    }
};

inline void loadArrayWithData(PyArrayObject* numpyArray,
    SiffParams& params, FrameData& frameData, std::ifstream& siff,
    bool flim, uint64_t terminalBin, PyObject* shift_tuple = NULL) {
    
    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

    // TODO: put the bool flags into the SiffParams struct.
    uint16_t* data_ptr = (uint16_t *) PyArray_DATA(numpyArray);
    npy_intp* dims = PyArray_DIMS(numpyArray);
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    frameData.siffCompress ?
            readCompressed(samplesThisFrame, frameData, siff, data_ptr, flim, dims, shift_tuple, terminalBin) : 
            readRaw(samplesThisFrame, frameData, siff, data_ptr, flim, dims, shift_tuple, terminalBin);
    // it's only ever called on siffs so I don't need the if/else
};

inline void loadArrayWithSummedArrivalTimes(
        PyArrayObject* lifetimeArray,
        PyArrayObject* intensityArray,
        SiffParams& params,
        FrameData& frameData,
        std::ifstream& siff,
        PyObject* shift_tuple = NULL
    )
    {
    
    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

    // TODO: put the bool flags into the SiffParams struct.
    double_t* lifetime_ptr = (double_t *) PyArray_DATA(lifetimeArray);
    uint16_t* intensity_ptr = (uint16_t *) PyArray_DATA(intensityArray);
    npy_intp* dims = PyArray_DIMS(lifetimeArray);
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    if (params.issiff) {
        frameData.siffCompress ?
            readCompressedForArrivals(samplesThisFrame, frameData, siff, lifetime_ptr, intensity_ptr, dims, shift_tuple) : 
            readRawForArrivals(samplesThisFrame, frameData, siff, lifetime_ptr, intensity_ptr, dims, shift_tuple);
    }
    else {
        // ye shant call this function on a tiff!
        throw std::runtime_error("Image files of type .tiff do not contain arrival time data.");
    }
};

/*************************************
 * 
 *  HISTOGRAMMING FUNCTIONALITY
 * 
 * *********************************/


inline void readCompressedArrivals(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    uint64_t* data_ptr) // uses siffCompress
    {
    // Can ignore the preceding array listing photons per pixel.
    siff.seekg(frameData.dataStripAddress);

    uint16_t readsThisFrame[samplesThisFrame];
    siff.read((char*)&readsThisFrame,frameData.stripByteCounts);
    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (uint64_t photon = 0; photon<samplesThisFrame; photon++) {
        data_ptr[readsThisFrame[photon]]++; // not even needing masking
    }
}

inline void readRawArrivals(uint64_t samplesThisFrame, FrameData& frameData, std::ifstream& siff,
    uint64_t* data_ptr) // every read is a uint64 of y, x, tau
    {
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress);
    siff.clear();
    uint64_t frameReads[samplesThisFrame];
    siff.read((char*)&frameReads,frameData.stripByteCounts);
    siff.clear();
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment the appropriate numpy element
        data_ptr[U64TOTAU(frameReads[photon])]++;
    }
}


inline void addArrivalsToArray(PyArrayObject* numpyArray, SiffParams& params, FrameData& frameData, std::ifstream& siff) {
    // Takes a numpy array and adds arrival times of a frame to it.
    uint64_t* data_ptr = (uint64_t *) PyArray_DATA(numpyArray);
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    frameData.siffCompress ?
        readCompressedArrivals(samplesThisFrame, frameData, siff, data_ptr) : 
        readRawArrivals(samplesThisFrame, frameData, siff, data_ptr);

}

inline std::vector<uint64_t> compressedReadsToVec(FrameData& frameData, std::ifstream& siff, PyObject* shift_tuple) {
    // decompresses them. Inelegant but I just wanted a single format for all frame types.
    // Might come back and adjust this implementation so that everything actually uses the compressed
    // format -- there are reasons it'd be better for the flim_map function anyways.
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));

    // first get the number of photons for each pixel
    uint16_t photonCounts[frameData.imageLength * frameData.imageWidth];
    siff.read((char*)&photonCounts, pixelsInImage * sizeof(uint16_t));
    
    siff.clear();

    siff.seekg(frameData.dataStripAddress);

    uint64_t samplesThisFrame = 8*frameData.stripByteCounts/frameData.bitsPerSample;
    
    uint16_t frameReads[samplesThisFrame];
    siff.read((char*)&frameReads,frameData.stripByteCounts);

    uint64_t ydim = frameData.imageLength;
    uint64_t xdim = frameData.imageWidth;

    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    std::vector<uint64_t> readsVector(0);
    readsVector.reserve(samplesThisFrame);
    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (uint64_t px = 0; px < pixelsInImage; px++) {
        uint16_t photonsThisPx = photonCounts[px];
        uint16_t pxReads[photonsThisPx];
        siff.read((char*)&pxReads, photonsThisPx*sizeof(uint16_t));

        for (uint16_t readNum = 0; readNum < photonsThisPx; readNum++) {
            readsVector.push_back(
                pxReads[readNum] +                   // tau index
                    (((px+x_shift)%xdim +(((px/xdim + y_shift)%ydim) << 16)) << 32) // x and y indices
            );
        }
        
    }
    return readsVector;
};

inline std::vector<uint64_t> uncompressedReadsToVec(FrameData& frameData, std::ifstream& siff, PyObject* shift_tuple) {
    siff.clear();
    siff.seekg(frameData.dataStripAddress);
    siff.clear();
    
    uint64_t samplesThisFrame = 8*frameData.stripByteCounts/frameData.bitsPerSample;
    uint64_t frameReads[samplesThisFrame];
    siff.read((char*)&frameReads,frameData.stripByteCounts);
    std::vector<uint64_t> retVec(0);
    retVec.reserve(samplesThisFrame);

    uint64_t ydim = frameData.imageLength;
    uint64_t xdim = frameData.imageWidth;

    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        uint64_t thisRead = frameReads[photon];
        
        retVec.push_back(U64TOTAU(thisRead) +
                ((((U64TOX(thisRead) + x_shift) % xdim) +
                (((U64TOY(thisRead) + y_shift) % ydim) << 16)) <<32)
        );
    }

    return retVec;
};


inline void log_p(std::vector<uint64_t>& reads, double_t tauo, npy_intp* dims,
    double_t* lifetime_ptr, uint16_t* intensity_ptr, double_t* conf_ptr,
    std::vector<double_t> arrival_p)
    { // fill arrays using a log_p confidence measure

    for (uint64_t readNum = 0; readNum < reads.size(); readNum++) {
        uint64_t read = reads[readNum];
        uint64_t this_px = U64TOY(read)*dims[1] + U64TOX(read);
        lifetime_ptr[this_px] += U64TOTAU(read);
        intensity_ptr[this_px]++;
        double_t logp = log(arrival_p[U64TOTAU(read)]); // easy!
        
        if ( !(isinf(logp) || isnan(logp)) ) conf_ptr[this_px] += logp;
    }

    for (int64_t px = 0; px<dims[0]*dims[1]; px++) {
        lifetime_ptr[px] /= intensity_ptr[px];
        lifetime_ptr[px] -= tauo;
    }

}

inline void chi_sq(std::vector<uint64_t>& reads, double_t tauo, npy_intp* dims,
    double_t* lifetime_ptr, uint16_t* intensity_ptr, double_t* conf_ptr,
    std::vector<double_t> arrival_p)
    { // fill arrays using a chi_sq confidence measure
    
    // we'll need the intensity pointer first anyways, so let's just get that
    for (uint64_t readNum = 0; readNum < reads.size(); readNum++) {
        uint64_t read = reads[readNum];
        uint64_t this_px = U64TOY(read)*dims[1] + U64TOX(read); // frame has already been registered by here
        uint16_t arrival = U64TOTAU(read);
        
        lifetime_ptr[this_px] += arrival;
        intensity_ptr[this_px]++;
    }

    // Lifetime by pixel
    for (int64_t px = 0; px<dims[0]*dims[1]; px++) {
        lifetime_ptr[px] /= intensity_ptr[px]; // benefit of nanning if intensity is 0 for free
        lifetime_ptr[px] -= tauo;
        conf_ptr[px] /= intensity_ptr[px]; // nan the bad ones
        conf_ptr[px] += intensity_ptr[px]; // for the iterative procedure, start as if all observed are 0.
    }

    // Now re-read the arrival time, knowing the expected value for each histogram bin per pixel
    // to compute the chi-squared statistic

    // surprised to find this is the fastest way, simply because it avoids all the pre- and re-
    // allocation of memory to track the arrival times of each pixel. It's not exactly blazing
    // fast, and the more frames averaged (or the longer the vector) the smaller the gains.
    // anyways, worth revisiting if I need to find more speed gains.
    std::sort(reads.begin(), reads.end());
    uint16_t obs = 0;
    uint64_t lastRead = reads[0];

    for (uint64_t readNum = 1; readNum < reads.size(); readNum++) {
        uint64_t read = reads[readNum];
        uint64_t this_px = U64TOY(read)*dims[1] + U64TOX(read);
        uint16_t arrival = U64TOTAU(read);
        obs = obs*(read==lastRead) + 1;
        double_t expected = intensity_ptr[this_px]*arrival_p[arrival];
        conf_ptr[this_px] += ((2*obs-1)/expected) - 2; // dChi-sq/dObserved
    }
    // still debating whether there's a better, faster way
}

inline PyObject* readVectorToNumpyTuple(std::vector<uint64_t>& photonReadsTogether, 
    FrameData& firstFrameData, PyObject* FLIMParams, const char* conf_measure) {
    // Takes the read vector and data about the frame shape + FLIM parameters
    // returns a tuple: 
    //      pixel-wise empirical lifetime map
    //      pixel-wise intensity
    //      pixel-wise confidence measure (chi-sq, log_p, etc.)

    PyObject* T_O = PyObject_GetAttrString(FLIMParams, "T_O");
    double_t tauo = PyFloat_AS_DOUBLE(T_O);
    Py_DECREF(T_O);
    if ((tauo == -1.0) && PyErr_Occurred()) {
        throw std::runtime_error("Purported FLIMParams object has no attribute 'T_O'.");
    }


    // Create the arrays first

    const int ND = 2; // number of dimensions
    npy_intp dims[ND];

    dims[0] = firstFrameData.imageLength;
    dims[1] = firstFrameData.imageWidth;

    PyArrayObject* lifetimeArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_FLOAT64, 
        0
    );

    double_t* lifetime_ptr = (double_t*) PyArray_DATA(lifetimeArray);

    PyArrayObject* intensityArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_UINT16, 
        0
    );

    uint16_t* intensity_ptr = (uint16_t*) PyArray_DATA(intensityArray);

    PyArrayObject* confArray = (PyArrayObject*) PyArray_ZEROS(
        ND,
        dims,
        NPY_FLOAT64, 
        0
    );

    double_t* conf_ptr = (double_t*) PyArray_DATA(confArray);

    // compute the likelihood, under the hypothesis of FLIMParams, of a photon
    // arriving in any particular bin.
    std::vector<double_t> arrival_p = compute_arrival_p(FLIMParams);

    if(strcmp(conf_measure, "log_p") == 0) {
        log_p(photonReadsTogether, tauo, dims, lifetime_ptr, intensity_ptr, conf_ptr, arrival_p);
    }
    
    if(strcmp(conf_measure, "chi_sq") == 0) {
        chi_sq(photonReadsTogether, tauo, dims, lifetime_ptr, intensity_ptr, conf_ptr, arrival_p);
    }
    
    PyObject* outTuple = Py_BuildValue("(OOO)", lifetimeArray, intensityArray, confArray);
    Py_DECREF(lifetimeArray);
    Py_DECREF(intensityArray);
    Py_DECREF(confArray);

    return outTuple;
}

inline void normalizeAndOffsetFlimTuple(PyObject* FlimTup, double_t tauo){
    // divide the summed lifetime bins by the number of photons, subtract the offset
    PyArrayObject* lifetimeArray = (PyArrayObject*) PyTuple_GetItem(FlimTup, Py_ssize_t(0));
    double_t* lifetime_ptr = (double_t *) PyArray_DATA(lifetimeArray);
    npy_intp* dims = PyArray_DIMS(lifetimeArray);

    PyArrayObject* intensityArray = (PyArrayObject*) PyTuple_GetItem(FlimTup, Py_ssize_t(1));
    uint16_t* intensity_ptr = (uint16_t *) PyArray_DATA(intensityArray);
    // Lifetime by pixel
    for (int64_t px = 0; px<dims[0]*dims[1]; px++) {
        lifetime_ptr[px] /= intensity_ptr[px]; // benefit of nanning if intensity is 0 for free
        lifetime_ptr[px] -= tauo;
    }
}

#endif