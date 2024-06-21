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

//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////// MASK ////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

/**
 * @brief Sums all pixels within the mask provided,
 * assuming that `mask` is a 2d numpy array of dtype bool,
 * and returns the sum.
 * 
 * For a single mask!
*/
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
                -y_shift,
                -x_shift,
                mask_dims[0],
                mask_dims[1]
            )
        ]*mask_data_ptr[px];
    }
    delete[] frameReads;

    return photon_count;
};

/**
 * @brief Sums all pixels within each of the
 * masks requested during one read of the `.siff`
 * data. The masks are assumed to be stored with
 * the slowest axis corresponding to the number of
 * masks. But it loads the input array with the masks
 * in FASTEST order (for contiguity), to be transposed
 * by the `SiffIO` function.
*/
inline void sumMasksCompressed(
    uint64_t* data_ptr,
    const FrameData& frameData,
    std::ifstream &siff,
    const bool* mask_data_ptr,
    const npy_intp* mask_frame_dims,
    const size_t n_masks,
    const size_t mask_skip,
    PyObject* shift_tuple
    ) {
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    // Goes to the region where the photon count framedata is stored
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
    
    uint16_t* frameReads = new uint16_t[pixelsInImage];
    siff.read((char*)frameReads, pixelsInImage * sizeof(uint16_t));
    siff.clear();

    uint64_t shifted_px;
    for(uint64_t px = 0; px < pixelsInImage; px++) {
        shifted_px = PIXEL_SHIFT(
            px,
            -y_shift,
            -x_shift,
            mask_frame_dims[0],
            mask_frame_dims[1]
        );
        // Note if crash: come back and check THIS line!
        for(uint64_t mask_idx = 0; mask_idx < n_masks; mask_idx++) {
            data_ptr[mask_idx] += frameReads[shifted_px]
                *mask_data_ptr[mask_idx * mask_skip + px];
        }
    }
    delete[] frameReads;
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
/**
 * @brief Sums all pixels within each of the masks provided,
 * assuming that `mask` is a 2 or 3d numpy array of dtype bool.
 * 
 * Loads the data pointer provided rather than computing a number
 * and returning it.
*/
inline void sumMasksRaw(
    uint64_t* data_ptr,
    const uint64_t& samplesThisFrame,
    const FrameData& frameData,
    std::ifstream &siff,
    const bool* mask_data_ptr,
    const npy_intp* mask_frame_dims,
    const size_t n_masks,
    const size_t mask_skip_px,
    PyObject* shift_tuple
    ) {
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*)frameReads,frameData.stripByteCounts);
    siff.clear();

    uint64_t px;
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment if the mask data pointer for that photon is true
        px = READ_TO_PX(
            frameReads[photon],
            y_shift,
            x_shift,
            mask_frame_dims[0],
            mask_frame_dims[1]
        );
        for (uint64_t mask_idx = 0; mask_idx < n_masks; mask_idx++) {
            data_ptr[mask_idx] += mask_data_ptr[mask_idx * mask_skip_px + px];
        }
    }
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
/**
 * @brief Sums all pixels within each of the masks provided,
 * assuming that `mask` is a 2d numpy array of dtype bool,
 * and loads the array pointed to by `data_ptr` with the sum
 * corresponding to the mask. Slow dimension of `masks` pointer
 * is assumed to be the number of masks.
 * 
 * @param data_ptr Points to the location within the data array
 * where the sum of the masks begins. Values `*data_ptr` to
 * `*(data_ptr + mask_dims[0])` will be filled with the sum of
 * the masks.
 * 
 * @param frameData The metadata for the frame being summed.
 * 
 * @param params The SiffParams struct containing the metadata
 * for the siff file to read the frame correctly.
 * 
 * @param mask_data_ptr Points to the mask data, offset by
 * the z plane count. So the pointer value is always:
 * 0*entire_mask_volume + z_slice * ydim * xdim + 0*ydim + 0*xdim,
 * and to jump to the next mask you need to step by xdim*ydim*zdim
 * 
 * @param mask_frame_dims The y, xdimensions of the mask array. 
 * 
 * @param mask_skip_idx The number of pixels to skip to jump
 * by an entire mask (if the mask is 3d, this is xdim * ydim * zdim,
 * otherwise it's xdim * ydim).
 * 
 * @param shift_tuple A tuple of two integers, the first of which
 * is the y shift and the second of which is the x shift.
 * 
 * @param siff The ifstream object pointing to the siff file.
 * 
 * @return void
 * 
 * @note This function is not yet implemented.
*/
void sumFrameMasks(
    uint64_t* data_ptr,
    const FrameData& frameData,
    const SiffParams& params,
    const bool* mask_data_ptr,
    const npy_intp* mask_frame_dims,
    const size_t n_masks,
    const size_t mask_skip_px,
    PyObject* shift_tuple,
    std::ifstream& siff
    ) {

    siff.seekg(frameData.dataStripAddress);
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;

    siff.clear();

    if (params.issiff) {
        // TODO, needs to be inline
        frameData.siffCompress ? 
            sumMasksCompressed(
                data_ptr,
                frameData,
                siff,
                mask_data_ptr,
                mask_frame_dims,
                n_masks,
                mask_skip_px,
                shift_tuple
            ) :
            sumMasksRaw(
                data_ptr,
                samplesThisFrame,
                frameData,
                siff,
                mask_data_ptr,
                mask_frame_dims,
                n_masks,
                mask_skip_px,
                shift_tuple
            );
    }    
    else {
        if( !(samplesThisFrame == uint64_t(mask_frame_dims[0]*mask_frame_dims[1]))) {
            throw std::runtime_error("Mask dimensions don't match frame metadata");
        }

        uint16_t* frameReads = new uint16_t[samplesThisFrame];
        siff.read((char*)frameReads,frameData.stripByteCounts);
        siff.clear();

        // figure out the rigid deformation for registration
        uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
        uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

        // Iterate over the masks
        for (uint64_t mask_idx = 0; mask_idx < n_masks; mask_idx++) {
            for(uint64_t px = 0; px < samplesThisFrame; px++) {
                data_ptr[mask_idx] += frameReads[
                    PIXEL_SHIFT(
                        px,
                        y_shift,
                        x_shift,
                        mask_frame_dims[0],
                        mask_frame_dims[1]
                    )
                ] * mask_data_ptr[mask_idx * mask_skip_px + px];
            }
        }
    }
};

PyArrayObject* SiffReader::sumMask(
    const uint64_t frames[],
    const uint64_t framesN,
    PyArrayObject* mask,
    PyObject* registrationDict
    ) const {
    // Sums all pixel elements of the desired frames within the mask and returns a 1d PyArrayObject.
    try{
        // More than one mask goes into `sumMasks`
        if (PyArray_NDIM(mask) > 3) return sumMasks(frames, framesN, mask, registrationDict);
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
        if (PyArray_TYPE(mask) != NPY_BOOL) {
            throw std::runtime_error("Mask is not of dtype 'bool'");
        }

        const bool* mask_data_ptr = (bool*) PyArray_DATA(mask);
        const npy_intp* mask_frame_dims = &PyArray_DIMS(mask)[PyArray_NDIM(mask) - 2];        
        size_t frames_per_mask = framesPerMask(mask, false);
        const size_t pxPerMask = mask_frame_dims[0] * mask_frame_dims[1];

        for(size_t frame_idx = 0; frame_idx < framesN; frame_idx++){

            const FrameData frameData = getTagData(
                params.allIFDs[frames[frame_idx]],
                params,
                siff
            );

            PyObject* shift_tuple = PyDict_GetItem(
                registrationDict, 
                PyLong_FromUnsignedLongLong(frames[frame_idx])
            );
            
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

PyArrayObject* SiffReader::sumMasks(
    const uint64_t frames[],
    const uint64_t framesN,
    PyArrayObject* mask,
    PyObject* registrationDict
) const {
   try{
        if (!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();

        const npy_intp n_masks = PyArray_DIMS(mask)[0];
        // 2 dimensional
        npy_intp dims[2];
        // Fast dimension is mask
        dims[1] = n_masks;
        // slow dimension is frames
        dims[0] = framesN;
        PyArrayObject* summedArray = (PyArrayObject*) PyArray_ZEROS(
            2, // 2d array
            dims, // length equal to number of frames x n_masks
            NPY_UINT64, // count data, and compressed enough to not care about saving space
            0 // C order, i.e. last index increases fastest
        ); // Or should I make a sparse array? Maybe make that an option? TODO.

        uint64_t* data_ptr = (uint64_t *) PyArray_DATA(summedArray);

        // Get the mask formatting right
        if (PyArray_TYPE(mask) != NPY_BOOL) {
            throw std::runtime_error("Mask is not of dtype `bool`");
        }

        const bool* mask_data_ptr = (bool*) PyArray_DATA(mask);
        
        // x and y dimensions are the last two
        const npy_intp* mask_frame_dims = &PyArray_DIMS(mask)[PyArray_NDIM(mask) - 2];        
        
        size_t slices_per_mask = framesPerMask(mask, true);
        
        const size_t pxPerMask = mask_frame_dims[0] * mask_frame_dims[1];
        // Number of pixels to skip to get to the next mask
        const size_t mask_skip_px = pxPerMask * slices_per_mask;
        PyObject* shift_tuple;

        for(size_t frame_idx = 0; frame_idx < framesN; frame_idx++){

            const FrameData frameData = getTagData(
                params.allIFDs[frames[frame_idx]],
                params,
                siff
            );

            shift_tuple = PyDict_GetItem(
                registrationDict, 
                PyLong_FromUnsignedLongLong(frames[frame_idx])
            );
            
            sumFrameMasks(
                &data_ptr[frame_idx * n_masks], // point to the number of frames loaded so far * number of masks
                frameData,
                params,
                &mask_data_ptr[(frame_idx % slices_per_mask) * pxPerMask], // point to the relevant part of the mask
                mask_frame_dims, // point to the dimensions of the mask relevant for frames
                n_masks,
                mask_skip_px,
                shift_tuple,
                siff
            );
        }

        return summedArray;
    }

    REPORT_ERR("Error in sum_roi mask method: "); 
}


//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////// GRAVEYARD ///////////////////

/**
inline uint64_t sumMaskRawBuffer(
        std::vector<uint64_t>& readBuffer,
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

    if(readBuffer.size() < samplesThisFrame) {
        // allocate enough
        readBuffer.resize(samplesThisFrame);
    }
    //uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*)readBuffer.data(),frameData.stripByteCounts);
    siff.clear();
    
    // if we're just doing intensity, the pointer element to increment should be different
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment if the mask data pointer for that photon is true
        photon_counts += mask_data_ptr[
            READ_TO_PX(
                readBuffer[photon],
                y_shift, // we're shifting the mask, not the frame
                x_shift,
                mask_dims[0],
                mask_dims[1]
            )
        ];
    }
    //delete[] frameReads;

    return photon_counts;
};

inline uint64_t sumMaskCompressedBuffer(
    std::vector<uint64_t>& readBuffer,
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
    
    if(readBuffer.size() < pixelsInImage) {
        // allocate enough
        readBuffer.resize(pixelsInImage);
    }
    siff.read((char*)readBuffer.data(), pixelsInImage * sizeof(uint16_t));
    siff.clear();

    for(uint64_t px = 0; px < pixelsInImage; px++) {
        photon_count += readBuffer[
            PIXEL_SHIFT(
                px,
                y_shift,
                x_shift,
                mask_dims[0],
                mask_dims[1]
            )
        ]*mask_data_ptr[px];
    }

    return photon_count;
};

PyArrayObject* SiffReader::sumMaskNew(
    const uint64_t frames[],
    const uint64_t framesN,
    PyArrayObject* mask,
    PyObject* registrationDict
    ) const {
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
        if (PyArray_TYPE(mask) != NPY_BOOL) {
            throw std::runtime_error("Mask is not of dtype 'bool'");
        }

        const bool* mask_data_ptr = (bool*) PyArray_DATA(mask);
        const npy_intp* mask_frame_dims = &PyArray_DIMS(mask)[PyArray_NDIM(mask) - 2];        
        size_t frames_per_mask = framesPerMask(mask);
        const size_t pxPerMask = mask_frame_dims[0] * mask_frame_dims[1];

        // To load reads so that you don't have to keep mallocing
        std::vector<uint64_t> readBuffer(0, mask_frame_dims[0]*mask_frame_dims[1]);

        for(size_t frame_idx = 0; frame_idx < framesN; frame_idx++){

            const FrameData frameData = getTagData(
                params.allIFDs[frames[frame_idx]],
                params,
                siff
            );

            PyObject* shift_tuple = PyDict_GetItem(
                registrationDict, 
                PyLong_FromUnsignedLongLong(frames[frame_idx])
            );
            
            data_ptr[frame_idx] += sumFrameMaskNew(
                readBuffer,
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
};

uint64_t sumFrameMaskNew(
    std::vector<uint64_t> readBuffer,
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
            sumMaskCompressedBuffer(readBuffer, frameData, siff, mask_data_ptr, mask_dims, shift_tuple) : 
            sumMaskRawBuffer(readBuffer, samplesThisFrame, frameData, siff, mask_data_ptr, mask_dims, shift_tuple);
    }
    else {
        if(!(samplesThisFrame == uint64_t(mask_dims[0]*mask_dims[1]))) throw std::runtime_error("Mask dimensions don't match frame metadata");
        // tiffs still malloc -- TODO: fix
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
*/