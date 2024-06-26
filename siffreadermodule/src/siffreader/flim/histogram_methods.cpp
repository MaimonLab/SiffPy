#include "../../../include/siffreader/siffreader.hpp"
#include "../../../include/framedata/sifdefin.hpp"

#define TAUMASK (((uint64_t) 1<<32) - 1)
#define U64TOTAU(photon) (uint64_t) (photon & TAUMASK)

/* Hard coded for now. Should be read from the siff at some point */
constexpr uint16_t TAUDIM_DEFAULT = 1024;

void readCompressedArrivals(
        const uint64_t samplesThisFrame,
        const FrameData& frameData,
        std::ifstream& siff,
        uint64_t* data_ptr,
        const uint16_t taudim
    ){
    // Can ignore the preceding array listing photons per pixel.
    // stored before the dataStripADdress
    siff.seekg(frameData.dataStripAddress);

    uint16_t* readsThisFrame = new uint16_t[samplesThisFrame];
    siff.read((char*) readsThisFrame, sizeof(uint16_t)*samplesThisFrame);
    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (size_t idx = 0; idx < samplesThisFrame; idx++) {
        uint16_t tau = readsThisFrame[idx];
        data_ptr[tau] += tau < taudim; // not even needing masking
    }
    delete[] readsThisFrame;
};

void readCompressedArrivalsMasked(
        const uint64_t samplesThisFrame,
        const FrameData& frameData,
        std::ifstream& siff,
        uint64_t* data_ptr,
        const uint16_t taudim,
        const bool *mask_data_ptr,
        const npy_intp *mask_dims
    ){
    // Somewhat more complicated now that we need to know which
    // elements of the read stream correspond to mask elements.
    siff.seekg(frameData.dataStripAddress);

    uint16_t* readsThisFrame = new uint16_t[samplesThisFrame];
    siff.read((char*) readsThisFrame, sizeof(uint16_t)*samplesThisFrame);
    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (size_t idx = 0; idx < samplesThisFrame; idx++) {
        uint16_t tau = readsThisFrame[idx];
        //data_ptr[tau] += tau < taudim && PyArray_GETITEM(mask, readsThisFrame[idx]);
    }
    delete[] readsThisFrame;
};

inline void readRawArrivals(
    const uint64_t samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    uint64_t* data_ptr,
    const uint16_t taudim) // every read is a uint64 of y, x, tau
    {
    
    siff.seekg(frameData.dataStripAddress);
    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*) frameReads,frameData.stripByteCounts);
    uint64_t ptr_idx;
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment the appropriate numpy element
        ptr_idx = U64TOTAU(frameReads[photon]);
        data_ptr[ptr_idx] += ptr_idx < taudim;
    }
    delete[] frameReads;
};

inline void readRawArrivalsMasked(
    const uint64_t samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    uint64_t* data_ptr,
    const uint16_t taudim,
    const bool *mask_data_ptr,
    const npy_intp *mask_dims
    ) // every read is a uint64 of y, x, tau
    {
    
    siff.seekg(frameData.dataStripAddress);
    uint64_t* frameReads = new uint64_t[samplesThisFrame];
    siff.read((char*) frameReads,frameData.stripByteCounts);
    uint64_t ptr_idx;
    const size_t xdim = mask_dims[1];
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        uint64_t read = frameReads[photon];
        // increment the appropriate numpy element
        ptr_idx = U64TOTAU(read);
        data_ptr[ptr_idx] += (ptr_idx < taudim) 
            && mask_data_ptr[U64TOY(read)*xdim + U64TOX(read)];
    }
    delete[] frameReads;
};

PyArrayObject* SiffReader::getHistogram(
    const uint64_t frames[] = NULL,
    const uint64_t framesN = 0
    ) {
    // NOTE! THIS USES UINT64_T BECAUSE YOU'RE LIKELY TO GET >65k PHOTONS PER BIN
    // By default, retrieves ALL frames, returns a single numpy array of the arrival times
    // create the 1-d numpy array.
    uint16_t tau_dim = TAUDIM_DEFAULT; // hardcoded for now. TODO: Implement this measure in SiffWriter
    npy_intp dims[1];
    dims[0] = tau_dim;
    PyArrayObject* numpyArray = (PyArrayObject*) PyArray_ZEROS(
        1,
        dims,
        NPY_UINT64, 
        0 // C order, i.e. last index increases fastest
    );
//
//
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        if(!params.issiff) throw std::runtime_error("Not a .siff -- no arrival time data.");
        siff.clear();
        
        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                singleFrameHistogram(params.allIFDs[frames[i]], numpyArray);
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                singleFrameHistogram(params.allIFDs[i], numpyArray);
            }
        }
        return numpyArray;
    }
    REPORT_ERR("Error collecting histogram: ");
}

void SiffReader::singleFrameHistogram(const uint64_t& thisIFD, PyArrayObject* numpyArray) const {
    // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    const FrameData frameData = getTagData(thisIFD, params, siff);

    uint64_t* data_ptr = (uint64_t *) PyArray_DATA(numpyArray);
    uint16_t taudim = PyArray_DIM(numpyArray, 0);
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint64_t bytesPerSample = frameData.siffCompress ? 2 : 8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    frameData.siffCompress ?
        readCompressedArrivals(samplesThisFrame, frameData, siff, data_ptr, taudim) : 
        readRawArrivals(samplesThisFrame, frameData, siff, data_ptr, taudim);
}

PyArrayObject* SiffReader::getHistogram(
    PyArrayObject *mask,
    const uint64_t frames[] = NULL,
    const uint64_t framesN = 0 
    ) {
    // By default, retrieves ALL frames, returns a single numpy array of the arrival times
    // create the 1-d numpy array.
    uint16_t tau_dim = TAUDIM_DEFAULT; // hardcoded for now. TODO: Implement this measure in SiffWriter
    npy_intp dims[1];
    dims[0] = tau_dim;
    PyArrayObject* numpyArray = (PyArrayObject*) PyArray_ZEROS(
        1,
        dims,
        NPY_UINT64, 
        0 // C order, i.e. last index increases fastest
    );
    try{
        if(!siff.is_open()) throw std::runtime_error("No open file.");
        if(!params.issiff) throw std::runtime_error("Not a .siff -- no arrival time data.");
        siff.clear();
        // mask is frame number mod frames_per_mask
        uint64_t frames_per_mask = framesPerMask(mask, false);
        const npy_intp* mask_dims = PyArray_DIMS(mask);
        const int ndims = PyArray_NDIM(mask);
        const size_t pxPerMask = mask_dims[ndims - 2] * mask_dims[ndims - 1];
        if(frames){
            for(uint64_t i = 0; i < framesN; i++){
                singleFrameHistogram(
                    params.allIFDs[frames[i]],
                    numpyArray,
                    &(((bool*)PyArray_DATA(mask))[
                        (i % frames_per_mask) * pxPerMask
                        ]),
                    mask_dims
                );
            }
        }
        else{
            for(uint64_t i = 0; i<params.numFrames; i++){
                singleFrameHistogram(
                    params.allIFDs[i],
                    numpyArray,
                    &(((bool*)PyArray_DATA(mask))[
                        (i % frames_per_mask) * pxPerMask
                        ]),
                    mask_dims
                );
            }
        }
        return numpyArray;
    }
    REPORT_ERR("Error collecting histogram: ");
}

void SiffReader::singleFrameHistogram(
    const uint64_t& thisIFD,
    PyArrayObject* numpyArray, 
    const bool *mask_data_ptr,
    const npy_intp *mask_dims
    ) const
    {
        // Reads an image's IFD, uses that to guide the output of array data in the siffreader.
    const FrameData frameData = getTagData(thisIFD, params, siff);

    uint64_t* data_ptr = (uint64_t *) PyArray_DATA(numpyArray);
    uint16_t taudim = PyArray_DIM(numpyArray, 0);
    
    siff.clear();
    siff.seekg(frameData.dataStripAddress); //  go to the data (skip the metadata for the frame)
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint64_t bytesPerSample = frameData.siffCompress ? 2 : 8;
    uint64_t samplesThisFrame = frameData.stripByteCounts / bytesPerSample;
    siff.clear();

    frameData.siffCompress ?
        readCompressedArrivalsMasked(
            samplesThisFrame,
            frameData,
            siff,
            data_ptr,
            taudim,
            mask_data_ptr,
            mask_dims
            )
        : 
        readRawArrivalsMasked(
            samplesThisFrame,
            frameData,
            siff,
            data_ptr,
            taudim,
            mask_data_ptr,
            mask_dims
            )
        ;
}