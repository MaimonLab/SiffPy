#include "../../../include/siffreader/siffreader.hpp"
#include "../../../include/framedata/sifdefin.hpp"

void readCompressedForArrivals(
    const uint64_t samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    double_t* lifetime_ptr,
    uint16_t* intensity_ptr,
    const npy_intp* dims,
    const double_t tauo,
    PyObject* shift_tuple
    ){
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));
    
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
    
    // BAD: do not `malloc` on every call!! Very wasteful!
    // TODO: Rewrite to pass a pointer to the array to be filled.

    // first read the unshifted intensity data
    uint16_t* photonCounts = new uint16_t[frameData.imageLength * frameData.imageWidth];
    siff.read((char*)photonCounts, pixelsInImage * sizeof(uint16_t));

    // now read the arrival times
    siff.seekg(frameData.dataStripAddress);
    // uint64_t arrivalsThisFrame[nPhotons];
    uint16_t* arrivalsThisFrame = new uint16_t[frameData.stripByteCounts/sizeof(uint16_t)];

    siff.read((char*) arrivalsThisFrame, frameData.stripByteCounts);

    uint16_t photonsThisPx;
    uint64_t shifted_px;
    
    size_t photonPointer = 0;
    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (uint64_t px = 0; px < pixelsInImage; px++) {
        photonsThisPx = photonCounts[px];
        shifted_px = PIXEL_SHIFT(
            px,
            y_shift,
            x_shift,
            dims[0],
            dims[1]
        );
        intensity_ptr[shifted_px] += photonsThisPx;

        for (uint16_t readNum = 0; readNum < photonsThisPx; readNum++) {
            lifetime_ptr[shifted_px] += arrivalsThisFrame[
                photonPointer + readNum
            ];
        }
        photonPointer += photonsThisPx;
        lifetime_ptr[shifted_px] /= photonsThisPx; // MEAN arrival time
        lifetime_ptr[shifted_px] -= tauo; // subtract the offset
    }
    delete [] photonCounts;
    delete [] arrivalsThisFrame;
}
//

void readRawForArrivals(
    const uint64_t samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    double_t* lifetime_ptr,
    uint16_t* intensity_ptr,
    const npy_intp* dims,
    const double_t tauo,
    PyObject* shift_tuple){
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(0)));
    uint64_t x_shift = PyLong_AsUnsignedLongLong(PyTuple_GetItem(shift_tuple, Py_ssize_t(1)));

    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.seekg(frameData.dataStripAddress);
    siff.read((char*)frameReads,frameData.stripByteCounts);
    siff.clear();
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        
        // increase this pixel's value by ARRIVAL_TIME
        lifetime_ptr[
            READ_TO_PX(
                frameReads[photon],
                y_shift,
                x_shift,
                dims[0],
                dims[1]
            )
        ] += U64TOTAU(frameReads[photon]);

        // increase this pixel's value by 1 for intensity increase
        intensity_ptr[
            READ_TO_PX(
                frameReads[photon],
                y_shift,
                x_shift,
                dims[0],
                dims[1]
            )
        ]++;
    }

    // Divide the lifetime by the intensity for every pixel
    for (size_t px = 0; px < ((size_t)(dims[0]*dims[1])); px++) {
        lifetime_ptr[px] /= intensity_ptr[px];
        lifetime_ptr[px] -= tauo; // subtract the offset
    }

    delete[] frameReads;
}
//

// Stores the empirical lifetime in lifetime_data_ptr --
// be aware when combining!
void loadArrayWithMeanArrivalTimes(
        double_t* lifetime_data_ptr,
        uint16_t* intensity_data_ptr,
        const npy_intp* dims,
        const double_t tauo,
        const SiffParams& params,
        const FrameData& frameData,
        std::ifstream& siff,
        PyObject* shift_tuple = NULL
    ){
    
    if (!shift_tuple) shift_tuple = PyTuple_Pack(Py_ssize_t(2), PyLong_FromLong(0), PyLong_FromLong(0));

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    
    siff.clear();

    if(!params.issiff) {
        throw std::runtime_error(
            "Image files of type .tiff do not contain arrival time data."
        );
    }

    frameData.siffCompress ?
        readCompressedForArrivals(
            samplesThisFrame,
            frameData,
            siff,
            lifetime_data_ptr,
            intensity_data_ptr,
            dims,
            tauo,
            shift_tuple
        ) : 
        readRawForArrivals(
            samplesThisFrame,
            frameData,
            siff,
            lifetime_data_ptr,
            intensity_data_ptr,
            dims,
            tauo,
            shift_tuple
        );
};

PyObject* SiffReader::flimTuple(
    PyObject* FLIMParams,
    const uint64_t frames[],
    const uint64_t framesN,
    const char* conf_measure,
    PyObject* registrationDict
    ) {
    // For the case in which a confidence measure is requested.
    try{
        // check the tauo offset value before we waste time evaluating things
        PyObject* T_O = PyObject_GetAttrString(FLIMParams, "T_O");
        double_t tauo = PyFloat_AS_DOUBLE(T_O);
        Py_DECREF(T_O);
        if ((tauo == -1.0) && PyErr_Occurred()) {
            throw std::runtime_error("Purported FLIMParams object has no attribute 'T_O'.");
        }

        PyObject* retTuple = PyTuple_New(3);

        const int ND(3);
        npy_intp dims[ND];

        dims[0] = framesN;
        dims[1] = frameDatas[frames[0]].imageLength;
        dims[2] = frameDatas[frames[0]].imageWidth;

        PyArrayObject* lifetimeArray((PyArrayObject*) PyArray_ZEROS(
            ND,
            dims,
            NPY_FLOAT64,
            0 // C order, i.e. last index increases fastest
        ));

        PyArrayObject* intensityArray((PyArrayObject*) PyArray_ZEROS(
            ND,
            dims,
            NPY_UINT16,
            0 // C order, i.e. last index increases fastest
        ));

        PyArrayObject* confidenceArray((PyArrayObject*) PyArray_ZEROS(
            ND,
            dims,
            NPY_FLOAT64,
            0 // C order, i.e. last index increases fastest
        ));

        const size_t frameSize = dims[1] * dims[2];
        double_t* lifetimeArrayData = (double_t*) PyArray_DATA(lifetimeArray);
        uint16_t* intensityArrayData = (uint16_t*) PyArray_DATA(intensityArray);
        uint64_t frameIdx;
        FrameData frameData;
        PyObject* shift_tuple;
        
        PyTuple_SET_ITEM(
            retTuple,
            0,
            (PyObject*) lifetimeArray
        );
        PyTuple_SET_ITEM(
            retTuple,
            1,
            (PyObject*) intensityArray
        );
        PyTuple_SET_ITEM(
            retTuple,
            2,
            (PyObject*) confidenceArray
        );

        for(Py_ssize_t idx(0); ((uint64_t)idx) < framesN; idx++) {
            // one merged numpy array for all of them.
            frameIdx = frames[idx];
            frameData = getTagData(params.allIFDs[frameIdx],params,siff);
            shift_tuple = PyDict_GetItem(
                registrationDict,
                PyLong_FromUnsignedLong(frameIdx)
            );

            loadArrayWithMeanArrivalTimes(
                &lifetimeArrayData[idx * frameSize],
                &intensityArrayData[idx * frameSize],
                &dims[1],
                tauo,
                params,
                frameData,
                siff,
                shift_tuple
            );
        }

        return retTuple;
    }
    REPORT_ERR("Error in flimMap");
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
///////////////////// MASK //////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////


// TODO: IEM estimates of lifetime!
double_t sumMaskFLIMCompressedIEM(
    const FrameData& frameData,
    std::ifstream &siff,
    const double_t& offset,
    const bool* mask_data_ptr,
    const npy_intp* mask_dims,
    PyObject* shift_tuple
    ){
    return 0.0;
 }

// Single-ROI
/**
 * @brief Returns the average
 * arrival time of photons within the mask
 * 
*/
inline double_t sumMaskFLIMCompressed(
    const FrameData& frameData, 
    std::ifstream &siff,
    const double_t& offset,
    const bool* mask_data_ptr,
    const npy_intp* mask_dims,
    PyObject* shift_tuple
    ){
    
    double_t summed_arrivals = 0;
    
    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(0))
    );

    uint64_t x_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(1))
    );
    
    
    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;

    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));
   
    // Get the frame intensity data
    uint16_t* frameReads = new uint16_t[pixelsInImage];
    siff.read((char*)frameReads, pixelsInImage * sizeof(uint16_t));
    siff.clear();

    siff.seekg(frameData.dataStripAddress);

    // Tracks photons counted within the mask
    // while you scan the photon stream
    uint64_t counted_photons_in_mask = 0;
    // Tracks how many photons have been read so far
    // regardless of whether they are in the mask
    uint64_t photonsSoFar = 0;

    uint16_t *photonStream = new uint16_t[frameData.stripByteCounts/sizeof(uint16_t)];
    
    uint64_t shiftedPx;
    siff.read((char*)photonStream, frameData.stripByteCounts);
    for(uint64_t px = 0; px < pixelsInImage; px++) {

        uint16_t photonCounts = frameReads[px];
        shiftedPx = PIXEL_SHIFT(
            px,
            y_shift,
            x_shift,
            mask_dims[0],
            mask_dims[1]
        );

        counted_photons_in_mask += photonCounts*mask_data_ptr[shiftedPx];
        for (size_t photon_idx = 0; photon_idx < photonCounts; photon_idx++) {
            summed_arrivals += (double_t) photonStream[
                photonsSoFar + photon_idx
            ]*mask_data_ptr[shiftedPx];
        }
        photonsSoFar += photonCounts;
    }
    delete[] frameReads;
    delete[] photonStream;

    return summed_arrivals/counted_photons_in_mask - offset;
};

// Multi-ROI!
/**
 * @brief Loads the average arrival time of
 * photons into the summedArray for each mask
 * 
 * 
*/
inline void sumMasksFLIMCompressed(
    double_t* summedArray,
    const FrameData& frameData,
    std::ifstream &siff,
    const double_t& offset,
    const bool* mask_data_ptr,
    const npy_intp* mask_frame_dims,
    const npy_intp n_masks,
    const size_t mask_skip_px,
    PyObject* shift_tuple
    ){
    // Feels wasteful?? Too much heap allocation!


    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(0))
    );
    
    uint64_t x_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(1))
    );

    uint64_t pixelsInImage = frameData.imageLength * frameData.imageWidth;
    siff.seekg(frameData.dataStripAddress - pixelsInImage*sizeof(uint16_t));

    uint16_t* frameReads = new uint16_t[pixelsInImage];
    siff.read((char*)frameReads, pixelsInImage * sizeof(uint16_t));
    siff.clear();
    siff.seekg(frameData.dataStripAddress);

    uint16_t *photonStream = new uint16_t[frameData.stripByteCounts/sizeof(uint16_t)];
    siff.read((char*)photonStream, frameData.stripByteCounts);

    uint64_t shifted_px;

    // Tracks the pixelwise number of photons
    uint16_t photonCounts;    
    uint64_t photonsSoFar = 0;

    // Tracks each mask's number of photons counted
    uint64_t* countedPhotons = new uint64_t[n_masks];
    for(uint64_t px = 0; px < pixelsInImage; px++){
        shifted_px = PIXEL_SHIFT(
            px,
            -y_shift,
            -x_shift,
            mask_frame_dims[0],
            mask_frame_dims[1]
        );

        photonCounts = frameReads[px];

        for (size_t mask_idx = 0; mask_idx < n_masks; mask_idx++) {
            // increment if the mask data pointer for that photon is true
            countedPhotons[mask_idx] += photonCounts * mask_data_ptr[
                mask_skip_px*mask_idx + shifted_px
            ];
            for (size_t photon_idx = 0; photon_idx < photonCounts; photon_idx++) {
                summedArray[mask_idx] += (double_t) photonStream[
                    photonsSoFar+photon_idx
                ] * mask_data_ptr[shifted_px + mask_skip_px*mask_idx];
            }
        }
        photonsSoFar += photonCounts;
    }
    for (size_t mask_idx = 0; mask_idx < n_masks; mask_idx++) {
        summedArray[mask_idx] /= countedPhotons[mask_idx];
        summedArray[mask_idx] -= offset;
    }
    delete[] frameReads;
    delete[] photonStream;
    delete[] countedPhotons;
}

// Single-ROI
inline double_t sumMaskFLIMRaw(
        const uint64_t& samplesThisFrame,
        const FrameData& frameData,
        std::ifstream &siff,
        const double_t& offset,
        const bool* mask_data_ptr,
        const npy_intp* mask_dims,
        PyObject* shift_tuple
    ) {

    double_t summed_bins = 0;

    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(0))
    );
    
    uint64_t x_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(1))
    );

    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*)frameReads,frameData.stripByteCounts);
    siff.clear();
    
    uint64_t n_counted = 0;
    // if we're just doing intensity, the pointer element to increment should be different
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment if the mask data pointer for that photon is true
        summed_bins += mask_data_ptr[
            READ_TO_PX(
                frameReads[photon],
                y_shift, // we're shifting the mask, not the frame
                x_shift,
                mask_dims[0],
                mask_dims[1]
            )
        ] * (double_t) U64TOTAU(frameReads[photon]);
        
        n_counted += mask_data_ptr[
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
    return summed_bins/n_counted - offset;
};

// Multi-ROI
/**
 * @brief Sums the arrival times of photons within
 * each mask and loads into into the `summedArray`
 * 
*/
inline void sumMasksFLIMRaw(
    double_t* summedArray,
    const uint64_t& samplesThisFrame,
    const FrameData& frameData,
    std::ifstream &siff,
    const double_t& offset,
    const bool* mask_data_ptr,
    const npy_intp* mask_frame_dims,
    const npy_intp n_masks,
    const size_t mask_skip_px,
    PyObject* shift_tuple
    ) {

    // figure out the rigid deformation for registration
    uint64_t y_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(0))
    );
    
    uint64_t x_shift = PyLong_AsUnsignedLongLong(
        PyTuple_GetItem(shift_tuple, Py_ssize_t(1))
    );

    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*)frameReads,frameData.stripByteCounts);
    siff.clear();
    
    uint64_t n_counted = 0;
    // if we're just doing intensity, the pointer element to increment should be different

    uint64_t px;

    uint64_t* countedPhotons = new uint64_t[n_masks];

    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment if the mask data pointer for that photon is true
        px = READ_TO_PX(
            frameReads[photon],
            y_shift, // we're shifting the mask, not the frame
            x_shift,
            mask_frame_dims[0],
            mask_frame_dims[1]
        );
        for (size_t mask_idx = 0; mask_idx < n_masks; mask_idx++) {
            countedPhotons[mask_idx] += mask_data_ptr[
                mask_skip_px*mask_idx + px
            ];
            summedArray[mask_idx] += mask_data_ptr[
                mask_skip_px*mask_idx + px
            ] * (double_t) U64TOTAU(frameReads[photon]);
        }
    }
    for (size_t mask_idx = 0; mask_idx < n_masks; mask_idx++) {
        summedArray[mask_idx] /= countedPhotons[mask_idx];
        summedArray[mask_idx] -= offset;
    }

    delete[] frameReads;
    delete[] countedPhotons;
    //return summed_bins/n_counted - offset;
};


// single ROI
double_t sumFrameFLIMMask(
        const FrameData& frameData,
        const SiffParams& params,
        const double_t offset,
        const bool* mask_data_ptr,
        const npy_intp* mask_dims,
        PyObject* shift_tuple,
        std::ifstream& siff
    ) {
    // Adds together all arrival times of photons within a frame if those counts are "True" in the mask array
    double_t arrival_time = 0;

    siff.seekg(frameData.dataStripAddress);
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    
    siff.clear();

    arrival_time = frameData.siffCompress ?
        sumMaskFLIMCompressed(frameData, siff, offset, mask_data_ptr, mask_dims, shift_tuple) : 
        sumMaskFLIMRaw(samplesThisFrame, frameData, siff, offset, mask_data_ptr, mask_dims, shift_tuple);

    return arrival_time;
};

// Multi-ROI
void sumFrameFLIMMasks(
    double_t* summedArray,
    const FrameData& frameData,
    const SiffParams& params,
    const double_t offset,
    const bool* mask_data_ptr,
    const npy_intp* mask_frame_dims,
    const npy_intp n_masks,
    const size_t mask_skip_px,
    PyObject* shift_tuple,
    std::ifstream& siff
    ) {
    siff.seekg(frameData.dataStripAddress);
    if (!(siff.good() || params.suppress_warnings)) {
        throw std::runtime_error("Failure to navigate to data in frame.");
    }

    uint16_t bytesPerSample = frameData.bitsPerSample/8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;

    siff.clear();

    frameData.siffCompress ? 
        sumMasksFLIMCompressed(
            summedArray,
            frameData,
            siff,
            offset,
            mask_data_ptr,
            mask_frame_dims,
            n_masks,
            mask_skip_px,
            shift_tuple
        ) :
        sumMasksFLIMRaw(
            summedArray,
            samplesThisFrame,
            frameData,
            siff,
            offset,
            mask_data_ptr,
            mask_frame_dims,
            n_masks,
            mask_skip_px,
            shift_tuple
        );
};

// Single ROI
PyArrayObject* SiffReader::sumFLIMMask(
    const uint64_t frames[],
    const uint64_t framesN,
    PyObject* FLIMParams,
    PyArrayObject* mask,
    PyObject* registrationDict
    ) const {
    // sums empirical lifetime inside the provided mask
    // returned array is 1d, regardless of mask shape.
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

        // Check the formatting of the mask
        if (PyArray_TYPE(mask) != NPY_BOOL) throw std::runtime_error("Mask is not of dtype 'bool'");

        const bool* mask_data_ptr = (bool*) PyArray_DATA(mask);
        npy_intp* mask_dims = PyArray_DIMS(mask);
        npy_intp* mask_frame_dims = &mask_dims[PyArray_NDIM(mask) - 2];
        size_t frames_per_mask = framesPerMask(mask, false);
        const size_t pxPerMask = mask_frame_dims[0] * mask_frame_dims[1];

        for(size_t frame_idx = 0; ((uint64_t)frame_idx) < framesN; frame_idx++){

            const FrameData frameData = getTagData(params.allIFDs[frames[frame_idx]], params, siff);

            PyObject* shift_tuple = PyDict_GetItem(
                registrationDict,
                PyLong_FromUnsignedLongLong(frames[frame_idx])
            );
            
            data_ptr[frame_idx] = sumFrameFLIMMask(
                frameData,
                params,
                tau_o,
                &mask_data_ptr[(frame_idx % frames_per_mask)*pxPerMask],
                mask_frame_dims, // point to the relevant dimension for masking a frame
                shift_tuple,
                siff
            );
        }
        return summedArray;
    }
    REPORT_ERR("Error in sum_roi_flim mask method: ")
};

// Multi-ROIs
PyArrayObject* SiffReader::sumFLIMMasks(
    const uint64_t frames[],
    const uint64_t framesN,
    PyObject* FLIMParams,
    PyArrayObject* masks,
    PyObject* registrationDict
    ) const {
    // sums empirical lifetime inside the provided masks
    // returned array is 2d
    try{
        if (!siff.is_open()) throw std::runtime_error("No open file.");
        siff.clear();
        DEBUG(
            logstream << "SumFLIMMasks " + std::to_string(framesN) + " " +
            std::to_string(debug_clock.now().time_since_epoch().count()) << std::endl;
        )
        // slowest dimension assumed to be mask number
        const npy_intp n_masks = PyArray_DIMS(masks)[0];
        // 2 dimensional
        npy_intp dims[2];
        dims[0] = framesN;
        dims[1] = n_masks;

        PyArrayObject* summedArray = (PyArrayObject*) PyArray_ZEROS(
            2, // 1d array
            dims, // length equal to number of frames
            NPY_DOUBLE, // empirical lifetime (albeit in units of bins), needs sub-bin resolution
            0 // C order, i.e. last index increases fastest
        ); // Or should I make a sparse array? Maybe make that an option? TODO.

        double_t* data_ptr = (double_t *) PyArray_DATA(summedArray);

        // Offset for empirical tau
        double_t tau_o = PyFloat_AsDouble(PyObject_GetAttrString(FLIMParams,"T_O"));

        // Check the formatting of the mask
        if (PyArray_TYPE(masks) != NPY_BOOL) {
            throw std::runtime_error("Mask is not of dtype `bool`");
        }

        const bool* mask_data_ptr = (bool*) PyArray_DATA(masks);
        npy_intp* mask_dims = PyArray_DIMS(masks);
        npy_intp* mask_frame_dims = &mask_dims[PyArray_NDIM(masks) - 2];
        
        size_t slices_per_mask = framesPerMask(masks, true);

        const size_t pxPerMask = mask_frame_dims[0] * mask_frame_dims[1];
        // Number of pixels to skip to get to the next mask
        const size_t mask_skip_px = pxPerMask * slices_per_mask;
        PyObject* shift_tuple;

        double_t *dataArray = (double_t *) PyArray_DATA(summedArray);

        for(size_t frame_idx = 0; ((uint64_t)frame_idx) < framesN; frame_idx++){
            const FrameData frameData = getTagData(
                params.allIFDs[frames[frame_idx]], params, siff
            );

            shift_tuple = PyDict_GetItem(
                registrationDict,
                PyLong_FromUnsignedLongLong(frames[frame_idx])
            );


            DEBUG(
                if (frame_idx % 100 == 0) {
                    logstream << "`sumFLIMMasks frame " 
                    << frame_idx << " " 
                    << debug_clock.now().time_since_epoch().count() << std::endl;
                }
            )
            
            sumFrameFLIMMasks(
                &dataArray[frame_idx*n_masks],
                frameData,
                params,
                tau_o,
                &mask_data_ptr[(frame_idx % slices_per_mask)*pxPerMask],
                mask_frame_dims, // point to the relevant dimension for masking a frame
                n_masks,
                mask_skip_px,
                shift_tuple,
                siff
            );
        }
        return summedArray;
    }
    REPORT_ERR("Error in sum_roi_flim mask method: ");
};