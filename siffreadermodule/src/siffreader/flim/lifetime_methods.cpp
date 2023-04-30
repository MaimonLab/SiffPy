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
    
    // first read the unshifted intensity data
    uint16_t photonCounts[frameData.imageLength * frameData.imageWidth];
    siff.read((char*)&photonCounts, pixelsInImage * sizeof(uint16_t));

    // now read the arrival times
    siff.seekg(frameData.dataStripAddress);
    // uint64_t arrivalsThisFrame[nPhotons];
    uint16_t arrivalsThisFrame[frameData.stripByteCounts/sizeof(uint16_t)];

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
}

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

    uint64_t frameReads[samplesThisFrame];
    siff.seekg(frameData.dataStripAddress);
    siff.read((char*)&frameReads,frameData.stripByteCounts);
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
    for (size_t px = 0; px < dims[0]*dims[1]; px++) {
        lifetime_ptr[px] /= intensity_ptr[px];
        lifetime_ptr[px] -= tauo; // subtract the offset
    }
}

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

        size_t frameSize = dims[1] * dims[2];
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

        for(Py_ssize_t idx(0); idx < framesN; idx++) {
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