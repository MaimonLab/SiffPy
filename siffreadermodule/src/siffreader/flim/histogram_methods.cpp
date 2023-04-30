#include "../../../include/siffreader/siffreader.hpp"
#include "../../../include/framedata/sifdefin.hpp"

#define TAUMASK (((uint64_t) 1<<32) - 1)
#define U64TOTAU(photon) (uint64_t) (photon & TAUMASK)

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

    uint16_t readsThisFrame[samplesThisFrame];
    siff.read((char*) readsThisFrame, sizeof(uint16_t)*samplesThisFrame);
    // now put the arrival time values that are in succession into the right elements of the numpy array.
    for (auto tau : readsThisFrame) {
        data_ptr[tau] += tau < taudim; // not even needing masking
    }
}

inline void readRawArrivals(
    const uint64_t samplesThisFrame,
    const FrameData& frameData,
    std::ifstream& siff,
    uint64_t* data_ptr,
    const uint16_t taudim) // every read is a uint64 of y, x, tau
    {
    
    siff.seekg(frameData.dataStripAddress);
    uint64_t frameReads[samplesThisFrame];
    siff.read((char*) frameReads,frameData.stripByteCounts);
    uint64_t ptr_idx;
    for(uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        // increment the appropriate numpy element
        ptr_idx = U64TOTAU(frameReads[photon]);
        data_ptr[ptr_idx] += ptr_idx < taudim;
    }
}

PyArrayObject* SiffReader::getHistogram(const uint64_t frames[], const uint64_t framesN) {
    // NOTE! THIS USES UINT64_T BECAUSE YOU'RE LIKELY TO GET >65k PHOTONS PER BIN
    // By default, retrieves ALL frames, returns a single numpy array of the arrival times
    // create the 1-d numpy array.
    uint16_t tau_dim = 1024; // hardcoded for now. TODO: Implement this measure in SiffWriter
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

void SiffReader::singleFrameHistogram(const uint64_t thisIFD, PyArrayObject* numpyArray){
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
