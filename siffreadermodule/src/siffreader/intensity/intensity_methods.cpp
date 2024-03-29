#include "../../../include/siffreader/siffreader.hpp"
#include "../../../include/framedata/sifdefin.hpp"

/*

Some bad style in all of this. Lot of code repetition, I should
be getting better at my C++ style by now..

*/

inline void readCompressed(
    const uint64_t& samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    uint16_t* data_ptr
){
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    // Goes to the region where the photon count framedata is stored
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
    siff.read((char*)data_ptr, pixelsInImage * sizeof(uint16_t));
    siff.clear();
}

inline void readCompressed(
    const uint64_t& samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    uint16_t* data_ptr,
    const npy_intp* &dims,
    PyObject* shift_tuple
    ) // uses siffCompress
    {
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(0)))
    ;
    uint64_t x_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(1))
    );
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    // Goes to the region where the photon count framedata is stored
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
   
    uint16_t* frameReads = new uint16_t[pixelsInImage];
    siff.read((char*)frameReads, pixelsInImage * sizeof(uint16_t));
    siff.clear();

    for(uint64_t px = 0; px < pixelsInImage; px++) {
        data_ptr[
            PIXEL_SHIFT(px, y_shift, x_shift, dims[0], dims[1])
        ] += frameReads[px];
    }
    delete[] frameReads;
}

inline void readRaw(
    const uint64_t& samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    uint16_t* data_ptr
){
    //const size_t ydim = frameData.imageLength;
    const size_t xdim = frameData.imageWidth;
    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*)frameReads,frameData.stripByteCounts);
    siff.clear();

    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        data_ptr[
            (((frameReads[photon] & YMASK) >> 48)*xdim) +
            ((frameReads[photon] & XMASK) >> 32)
        ]++; // increment the appropriate numpy element
    }
    delete[] frameReads;    
};

inline void readRaw(
    const uint64_t& samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    uint16_t* data_ptr,
    const npy_intp* &dims,
    PyObject* &shift_tuple
    ) // every read is a uint64 of y, x, tau
    {

    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*)frameReads,frameData.stripByteCounts);
    siff.clear();

    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment the appropriate numpy element
        data_ptr[
            READ_TO_PX(
                frameReads[photon],
                y_shift, 
                x_shift,
                dims[0],
                dims[1]
            )
        ]++;
    }
    delete[] frameReads;
}

void loadArrayWithData(
    uint16_t* data_ptr,
    const SiffParams& params,
    const FrameData& frameData,
    std::ifstream& siff
) {
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    if (params.issiff) {
        frameData.siffCompress ?
            readCompressed(samplesThisFrame, frameData, siff, data_ptr) : 
            readRaw(samplesThisFrame, frameData, siff, data_ptr);
    }
    else { // read in the tiff
        siff.read((char*)data_ptr,frameData.stripByteCounts);
        siff.clear();
    } 
};

void loadArrayWithData(
    uint16_t* data_ptr,
    const npy_intp* dims,
    const SiffParams& params,
    const FrameData& frameData,
    std::ifstream& siff,
    PyObject* shift_tuple
    ) {
    
    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0L), PyLong_FromLong(0L));

    // TODO: put the bool flags into the SiffParams struct.
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    if (params.issiff) {
        frameData.siffCompress ?
            readCompressed(samplesThisFrame, frameData, siff, data_ptr, dims, shift_tuple) : 
            readRaw(samplesThisFrame, frameData, siff, data_ptr, dims, shift_tuple);
    }
    else { // read in the tiff
        // pretty simple -- it's already formatted right.
        if(!(samplesThisFrame == uint64_t(dims[0]*dims[1]))) throw std::runtime_error("Frame data dimensions don't match frame metadata");
        // now that we're sure, just write the data into the data pointer!
        // First we read a copy because we'll need to shift it
        uint16_t* frameReads = new uint16_t[samplesThisFrame];
        siff.read((char*)frameReads,frameData.stripByteCounts);
        siff.clear();

        // figure out the rigid deformation for registration
        uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
        uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

        for(uint64_t px = 0; px < samplesThisFrame; px++) {
            data_ptr[
                PIXEL_SHIFT(px, y_shift, x_shift, dims[0], dims[1])
             ] += frameReads[px];
        }

        delete[] frameReads;
    }
};

void SiffReader::singleFrameAddToList(
    const uint64_t thisIFD,
    PyObject* numpyArrayList,
    const bool flim,
    PyObject* shift_tuple
    ) const {
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    // Then appends that IFD to a list of numpy arrays
    
    const FrameData frameData = getTagData(thisIFD, params, siff);
    // create a new numpy array of dimensions:
    // (y, x)
    const int ND = 2; // number of dimensions
    npy_intp dims[ND];

    dims[0] = frameData.imageLength;
    dims[1] = frameData.imageWidth;

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
        shift_tuple
    );

    int ret = PyList_Append(numpyArrayList, (PyObject*) numpyArray);
    Py_DECREF(numpyArray);
    
    if (ret<0) throw std::runtime_error("Failure to append frame array to list");
}

PyArrayObject* SiffReader::retrieveFramesAsArray(
    const uint64_t frames[],
    const uint64_t framesN,
    PyObject* registrationDict
    ) {
    // Build a PyArray object from the framelist
    // with length equal to the number of frames
    // and dtype uint64_t

    constexpr const int ND = 3;
    // TEMP

    npy_intp retdims[ND];
    retdims[0] = framesN;
    retdims[1] = frameDatas[frames[0]].imageLength;
    retdims[2] = frameDatas[frames[0]].imageWidth;

    PyArrayObject* retArray(
        (PyArrayObject*)PyArray_ZEROS(
            ND,
            retdims,
            NPY_UINT16,
            0
        )
    );

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
            shift_tuple
        );
    }
    return retArray;
};

PyObject* SiffReader::retrieveFrames(
    const uint64_t frames[],
    const uint64_t &framesN,
    PyObject* registrationDict
    ) const {
    // Eliminated some of the flexibility. Always is called with frames[], no more massive FLIM array
    // with this method.
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();
        // create the list into which we shall stuff the numpy arrays
        PyObject* numpyArrayList = PyList_New(0);
        for(uint64_t i = 0; i < framesN; i++){
            PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(frames[i]));
            singleFrameAddToList(params.allIFDs[frames[i]], numpyArrayList, false, shift_tuple);
        }
        return numpyArrayList;
    }
    REPORT_ERR("Error parsing frames: ")
}

/*
PyArrayObject* SiffReader::poolFrames(
    PyObject* listOfLists,
    const bool& flim,
    PyObject* registrationDict
    ) {
    // Pools all frames into one summed element by reading one at a time
    // and then appending them together. Considerably less memory intensive
    // than reading them all into NumPy arrays first and THEN pooling.
    // Permits multiple lists of lists, returning a single numpy array
    // for each sublist.
    try{

        // create a new numpy array of dimensions:
        PyArray_Descr* desc(PyArray_DescrFromType(NPY_UINT64));
        const int ND = 3;

        // ASSUMES consistent dimensionality
        npy_intp retdims[ND];
        retdims[0] = PyList_Size(listOfLists);
        retdims[1] = frameDatas[
            PyLong_AsUnsignedLongLong(
                PyList_GetItem(PyList_GetItem(listOfLists,0),0)
            )
        ].imageLength;
        retdims[2] = frameDatas[
            PyLong_AsUnsignedLongLong(
                PyList_GetItem(PyList_GetItem(listOfLists,0),0)
            )
        ].imageWidth;

        PyArrayObject* retArray((PyArrayObject*)PyArray_ZEROS(
            ND,
            retdims,
            NPY_UINT16,
            0
        ));

        uint16_t *data_ptr = (uint16_t *) PyArray_DATA(retArray);
        const size_t sizeOfFrame = retdims[1] * retdims[2];
        // Now parse the frames into the array

        // iterate through each list of frame indices
        for(Py_ssize_t idx(0); idx < PyList_Size(listOfLists); idx++) {
            // one merged numpy array for all of them. TODO: size checking!!
            // need to ensure they all have compatible dimensions -- for now
            // I just assume it, but as this expands to support mROI...

            // already ensured these were all PyLongs
            PyObject* listOfFrames = PyList_GetItem(listOfLists, idx);
            
            if(PyList_Size(listOfFrames)==0) { // empty list, you silly goose.
                continue;
            }

            // more than one frame in the list
            for(Py_ssize_t frameIdx(0); frameIdx < PyList_Size(listOfFrames); frameIdx++) {
                //PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyList_GetItem(listOfFrames,frameIdx)); 
                
                const FrameData frameData = getTagData(
                    params.allIFDs[ PyLong_AsLongLong(PyList_GetItem(listOfFrames,frameIdx)) ],
                    params,
                    siff
                );

                siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
                if (!(siff.good() || suppress_errors)) throw std::runtime_error("Failure to navigate to data in frame.");
    
                loadArrayWithData(
                    &data_ptr[idx * sizeOfFrame],
                    &PyArray_DIMS(retArray)[1], // y, x only
                    params,
                    frameData,
                    siff,
                    shift_tuple
                );
              
            }
        }
        return retArray;
    }
    REPORT_ERR("Error in pool frames: ")
}
*/

//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////// MASK ////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////


inline uint64_t sumMaskCompressed(
    const FrameData& frameData,
    std::ifstream &siff,
    const bool* mask_data_ptr,
    const npy_intp* mask_dims,
    PyObject* shift_tuple
    ) {
    
    uint64_t photon_count = 0;
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));
    
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;

    if(!(pixelsInImage == uint64_t(mask_dims[0]*mask_dims[1]))) throw std::runtime_error("Mask dimensions don't match frame metadata");
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
    
    uint16_t* frameReads = new uint16_t[pixelsInImage];
    siff.read((char*)frameReads, pixelsInImage * sizeof(uint16_t));
    siff.clear();

    for(uint64_t px = 0; px < pixelsInImage; px++) {
        photon_count += frameReads[
            PIXEL_SHIFT(
                px,
                y_shift,
                x_shift,
                mask_dims[0],
                mask_dims[1]
            )
        ]*mask_data_ptr[px];
    }
    delete[] frameReads;

    return photon_count;
};

inline uint64_t sumMaskRaw(
        const uint64_t& samplesThisFrame,
        const FrameData& frameData,
        std::ifstream &siff,
        const bool* mask_data_ptr,
        const npy_intp* mask_dims,
        PyObject* shift_tuple
    ){

    uint64_t photon_counts = 0;

    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*)frameReads,frameData.stripByteCounts);
    siff.clear();
    
    // if we're just doing intensity, the pointer element to increment should be different
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment if the mask data pointer for that photon is true
        photon_counts += mask_data_ptr[
            READ_TO_PX(
                frameReads[photon],
                y_shift, // we're shifting the mask, not the frame
                x_shift,
                mask_dims[0],
                mask_dims[1]
            )
        ];
    }
    delete[] frameReads;

    return photon_counts;
};

uint64_t sumFrameMask(
    const FrameData& frameData,
    const SiffParams& params,
    const bool* mask_data_ptr,
    const npy_intp* mask_dims,
    PyObject* shift_tuple,
    std::ifstream& siff
    ) {
    // Adds together all photon counts within a frame if those counts are "True" in the mask array
    uint64_t photon_count = 0;

    siff.seekg(frameData.dataStripAddress);
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    
    siff.clear();

    if (params.issiff) {
        photon_count = frameData.siffCompress ?
            sumMaskCompressed(frameData, siff, mask_data_ptr, mask_dims, shift_tuple) : 
            sumMaskRaw(samplesThisFrame, frameData, siff, mask_data_ptr, mask_dims, shift_tuple);
    }
    else {
        if(!(samplesThisFrame == uint64_t(mask_dims[0]*mask_dims[1]))) throw std::runtime_error("Mask dimensions don't match frame metadata");

        uint16_t* frameReads = new uint16_t[samplesThisFrame];
        siff.read((char*)frameReads,frameData.stripByteCounts);
        siff.clear();

        // figure out the rigid deformation for registration
        uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
        uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

        for(uint64_t px = 0; px < samplesThisFrame; px++) {
            photon_count += frameReads[
                PIXEL_SHIFT(
                    px,
                    y_shift,
                    x_shift,
                    mask_dims[0],
                    mask_dims[1]
                )
            ] * mask_data_ptr[px];
        }
        delete[] frameReads;
    }

    return photon_count;
};


PyArrayObject* SiffReader::sumMask(
    const uint64_t frames[],
    const uint64_t framesN,
    PyArrayObject* mask,
    PyObject* registrationDict
    ) const {
    // Sums all pixel elements of the desired frames within the mask and returns a 1d PyArrayObject.
    try{
        if (!siff.is_open()) throw std::runtime_error("No open file.");
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

        // Get the mask formatting right
        if (PyArray_TYPE(mask) != NPY_BOOL) throw std::runtime_error("Mask is not of dtype 'bool'");

        const bool* mask_data_ptr = (bool*) PyArray_DATA(mask);
        const npy_intp* mask_dims = PyArray_DIMS(mask);
        const npy_intp* mask_frame_dims = &mask_dims[PyArray_NDIM(mask) - 2];
        size_t frames_per_mask = 1;
        for (size_t dim_idx = 0; dim_idx < ((size_t) PyArray_NDIM(mask) - 2); dim_idx++) {
            frames_per_mask *= mask_dims[dim_idx];
        }
        const size_t pxPerMask = mask_frame_dims[0] * mask_frame_dims[1];

        for(size_t frame_idx = 0; frame_idx < framesN; frame_idx++){

            const FrameData frameData = getTagData(params.allIFDs[frames[frame_idx]], params, siff);

            PyObject* shift_tuple = PyDict_GetItem(registrationDict, PyLong_FromUnsignedLongLong(frames[frame_idx]));
            
            data_ptr[frame_idx] += sumFrameMask(
                frameData,
                params,
                &mask_data_ptr[(frame_idx % frames_per_mask) * pxPerMask], // point to the relevant part of the mask
                mask_frame_dims, // point to the dimensions of the mask relevant for frames
                shift_tuple,
                siff
            );
        }

        return summedArray;

    }
    REPORT_ERR("Error in sum_roi mask method: ");
}