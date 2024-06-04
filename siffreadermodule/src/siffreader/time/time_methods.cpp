#include "../../../include/siffreader/siffreader.hpp"
#include "../../../include/framedata/sifdefin.hpp"
#include <string_view>
#include <cstring>
//#include <ctre.hpp>
//#include <regex>

// BAD CODE, REPETITIVE STYLE

// Annoying that these are not constexpr
//const std::regex laserPattern("\nepoch\\s=\\s(\\d+)", std::regex_constants::optimize);
//const std::regex systemPattern("\nmostRecentSystemTimestamp_epoch\\s=\\s(\\d+)", std::regex_constants::optimize);
const char* EXPERIMENT_PATTERN = "\nframeTimestamps_sec = ";
const char* LASER_PATTERN = "\nepoch = ";
const char* SYSTEM_PATTERN = "\nmostRecentSystemTimestamp_epoch = ";
const char* APPENDED_TEXT_PATTERN = "\nAppended text = ";
const char* TEXT_TIMESTAMP_PATTERN = "\nText timestamp = ";

// constexpr auto laserPattern = ctll::fixed_string("\nepoch\\s=\\s(\\d+)");
// constexpr auto systemPattern = ctll::fixed_string("\nmostRecentSystemTimestamp_epoch\\s=\\s(\\d+)");

PyArrayObject* SiffReader::getExperimentTimestamps(
    const uint64_t frames[], const uint64_t framesN
    ) const {
    
    // Get the timestamps of each frame computed using
    // the number of laser pulses since the start of the experiment.
    if (!siff.is_open()) throw std::runtime_error("No file open.");
    siff.clear();

    // 1 dimensional
    npy_intp dims[1] = {(int) framesN};

    // Allocate the array
    PyArrayObject* timestamps = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (timestamps == NULL) throw std::runtime_error("Failed to allocate memory for timestamps.");

    // Get the timestamps
    double_t* timestampsPtr = (double_t*) PyArray_DATA(timestamps);

    const size_t initial_description_length = frameDatas[frames[0]].siffCompress ?
            frameDatas[frames[0]].dataStripAddress 
                - frameDatas[frames[0]].endOfIFD
                - frameDatas[frames[0]].imageLength*frameDatas[frames[0]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[0]].dataStripAddress - frameDatas[frames[0]].endOfIFD
    ;

    char* metaString = new char[initial_description_length];
    size_t descrLength = initial_description_length;
    
    // std::cmatch matches;
    
    for (uint64_t i = 0; i < framesN; i++) {
        // Reallocate if needed
        const size_t new_description_length = frameDatas[frames[i]].siffCompress ?
            frameDatas[frames[i]].dataStripAddress 
                - frameDatas[frames[i]].endOfIFD
                - frameDatas[frames[i]].imageLength*frameDatas[frames[i]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[i]].dataStripAddress - frameDatas[frames[i]].endOfIFD
        ;
        // need more space!
        if (new_description_length > descrLength) {
            delete[] metaString;
            metaString = new char[new_description_length];
            descrLength = new_description_length;
        }
        siff.seekg(frameDatas[frames[i]].endOfIFD, siff.beg);
        siff.read(metaString, descrLength);

        const char* patternStart = std::strstr(metaString, EXPERIMENT_PATTERN);
        if (patternStart != NULL) {
            timestampsPtr[i] = std::stod(patternStart + strlen(EXPERIMENT_PATTERN));
        }
    }

    delete[] metaString;

    return timestamps;
    
};


PyArrayObject* SiffReader::getEpochTimestampsLaser(
    const uint64_t frames[], const uint64_t framesN
    ) const {
    // Get the timestamps of each frame computed using
    // the number of laser pulses since the start of the experiment.
    if (!siff.is_open()) throw std::runtime_error("No file open.");
    siff.clear();

    DEBUG(
        logstream << "Getting epoch laser timestamps for " 
            << framesN << " frames." << std::endl;
        tick = debug_clock.now();
    )

    // 1 dimensional
    npy_intp dims[1] = { (int) framesN};

    // Allocate the array
    PyArrayObject* timestamps = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (timestamps == NULL) throw std::runtime_error("Failed to allocate memory for timestamps.");

    // Get the timestamps
    uint64_t* timestampsPtr = (uint64_t*) PyArray_DATA(timestamps);

    const size_t initial_description_length = frameDatas[frames[0]].siffCompress ?
            frameDatas[frames[0]].dataStripAddress 
                - frameDatas[frames[0]].endOfIFD
                - frameDatas[frames[0]].imageLength*frameDatas[frames[0]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[0]].dataStripAddress - frameDatas[frames[0]].endOfIFD
    ;

    char* metaString = new char[initial_description_length];
    size_t descrLength = initial_description_length;
    
    // std::cmatch matches;
    
    for (uint64_t i = 0; i < framesN; i++) {
        // Reallocate if needed
        const size_t new_description_length = frameDatas[frames[i]].siffCompress ?
            frameDatas[frames[i]].dataStripAddress 
                - frameDatas[frames[i]].endOfIFD
                - frameDatas[frames[i]].imageLength*frameDatas[frames[i]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[i]].dataStripAddress - frameDatas[frames[i]].endOfIFD
        ;
        // need more space!
        if (new_description_length > descrLength) {
            delete[] metaString;
            metaString = new char[new_description_length];
            descrLength = new_description_length;
        }
        siff.seekg(frameDatas[frames[i]].endOfIFD, siff.beg);
        siff.read(metaString, descrLength);

        const char* patternStart = std::strstr(metaString, LASER_PATTERN);
        if (patternStart != NULL) {
            timestampsPtr[i] = std::stoull(patternStart + strlen(LASER_PATTERN));
        }

        // auto match = ctre::match<laserPattern>(input);
        // if (match) {
        //     auto epochView = match.get<1>();
        //     timestampsPtr[i] = std::stoull(epochView.to_string());
        // }

        // if (std::regex_search(metaString, matches, laserPattern)) {
        //     std::string epochString(matches[1].first, matches[1].second);
        //     timestampsPtr[i] = std::stoull(epochString);
        // }
    }

    delete[] metaString;

    DEBUG(
        tock = debug_clock.now();
        logstream << "Getting laser epoch timestamps took " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick).count() 
            << "ms." << std::endl;
    )

    return timestamps;
};

PyArrayObject* SiffReader::getEpochTimestampsSystem(
    const uint64_t frames[], const uint64_t framesN
    ) const {
    // Get the timestamps of each frame computed using
    // the system clock.

    // Get the timestamps of each frame computed using
    // the number of laser pulses since the start of the experiment.
    if (!siff.is_open()) throw std::runtime_error("No file open.");
    siff.clear();

    DEBUG (
        logstream << "Getting epoch system timestamps for " << framesN
            << " frames." << std::endl;
        tick = debug_clock.now();
    )

    // 1 dimensional
    npy_intp dims[1] = { (int) framesN};

    // Allocate the array
    PyArrayObject* timestamps = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (timestamps == NULL) throw std::runtime_error("Failed to allocate memory for timestamps.");

    // Get the timestamps
    uint64_t* timestampsPtr = (uint64_t*) PyArray_DATA(timestamps);

    const size_t initial_description_length = frameDatas[frames[0]].siffCompress ?
            frameDatas[frames[0]].dataStripAddress 
                - frameDatas[frames[0]].endOfIFD
                - frameDatas[frames[0]].imageLength*frameDatas[frames[0]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[0]].dataStripAddress - frameDatas[frames[0]].endOfIFD
    ;

    char* metaString = new char[initial_description_length];
    size_t descrLength = initial_description_length;
    
    // std::cmatch matches;
    
    for (uint64_t i = 0; i < framesN; i++) {
        // Reallocate if needed
        const size_t new_description_length = frameDatas[frames[i]].siffCompress ?
            frameDatas[frames[i]].dataStripAddress 
                - frameDatas[frames[i]].endOfIFD
                - frameDatas[frames[i]].imageLength*frameDatas[frames[i]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[i]].dataStripAddress - frameDatas[frames[i]].endOfIFD
        ;
        // need more space!
        if (new_description_length > descrLength) {
            delete[] metaString;
            metaString = new char[new_description_length];
            descrLength = new_description_length;
        }
        siff.seekg(frameDatas[frames[i]].endOfIFD, siff.beg);
        siff.read(metaString, descrLength);

        const char* patternStart = std::strstr(metaString, SYSTEM_PATTERN);
        if (patternStart != NULL) {
            timestampsPtr[i] = std::stoull(patternStart + strlen(SYSTEM_PATTERN));
        }

        // if (std::regex_search(metaString, matches, systemPattern)) {
        //     std::string epochString(matches[1].first, matches[1].second);
        //     timestampsPtr[i] = std::stoull(epochString);
        // }
    }
    delete[] metaString;

    DEBUG (
        tock = debug_clock.now();
        logstream << "Getting system epoch timestamps took " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick).count() 
            << "ms." << std::endl;
    )

    return timestamps;
};

PyArrayObject* SiffReader::getEpochTimestampsBoth(
    const uint64_t frames[], const uint64_t framesN
    ) const {
    // Get the timestamps of each frame computed using
    // the system clock AND the ones from the laser clock.
    if (!siff.is_open()) throw std::runtime_error("No file open.");
    siff.clear();

    DEBUG(
        logstream << "Getting both epoch timestamps for " << framesN << " frames." << std::endl;
        tick = debug_clock.now();
    )

    // 2 dimensional
    npy_intp dims[2] = {2, (int) framesN};

    // Allocate the array
    PyArrayObject* timestamps = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_UINT64);
    if (timestamps == NULL) throw std::runtime_error("Failed to allocate memory for timestamps.");

    // Get the timestamps
    uint64_t* timestampsPtr = (uint64_t*) PyArray_DATA(timestamps);

    const size_t initial_description_length = frameDatas[frames[0]].siffCompress ?
            frameDatas[frames[0]].dataStripAddress 
                - frameDatas[frames[0]].endOfIFD
                - frameDatas[frames[0]].imageLength*frameDatas[frames[0]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[0]].dataStripAddress - frameDatas[frames[0]].endOfIFD
    ;

    char* metaString = new char[initial_description_length];
    size_t descrLength = initial_description_length;
    
    // std::cmatch matches;
    
    for (uint64_t i = 0; i < framesN; i++) {
        // Reallocate if needed
        const size_t new_description_length = frameDatas[frames[i]].siffCompress ?
            frameDatas[frames[i]].dataStripAddress 
                - frameDatas[frames[i]].endOfIFD
                - frameDatas[frames[i]].imageLength*frameDatas[frames[i]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[i]].dataStripAddress - frameDatas[frames[i]].endOfIFD
        ;
        // need more space!
        if (new_description_length > descrLength) {
            delete[] metaString;
            metaString = new char[new_description_length];
            descrLength = new_description_length;
        }
        siff.seekg(frameDatas[frames[i]].endOfIFD, siff.beg);
        siff.read(metaString, descrLength);

        const char* patternStart = std::strstr(metaString, LASER_PATTERN);
        if (patternStart != NULL) {
            timestampsPtr[i] = std::stoull(patternStart + strlen(LASER_PATTERN));
        }

        patternStart = std::strstr(metaString, SYSTEM_PATTERN);
        if (patternStart != NULL) {
            timestampsPtr[i+framesN] = std::stoull(patternStart + strlen(SYSTEM_PATTERN));
        }

        // if (std::regex_search(metaString, matches, laserPattern)) {
        //     std::string epochString(matches[1].first, matches[1].second);
        //     timestampsPtr[i] = std::stoull(epochString);
        // }
        // if (std::regex_search(metaString, matches, systemPattern)) {
        //     std::string epochString(matches[1].first, matches[1].second);
        //     timestampsPtr[i+framesN] = std::stoull(epochString);
        // }
    }

    delete[] metaString;

    DEBUG(
        tock = debug_clock.now();
        logstream << "Getting both epoch timestamps took " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick).count() 
            << "ms." << std::endl;
    )

    return timestamps;
};


PyObject* SiffReader::getAppendedText(
    const uint64_t frames[], const uint64_t framesN
    ) const {
    // Get the appended text for each frame
    if (!siff.is_open()) throw std::runtime_error("No file open.");
    siff.clear();

    PyObject* retList = PyList_New(0);

    const size_t initial_description_length = frameDatas[frames[0]].siffCompress ?
            frameDatas[frames[0]].dataStripAddress 
                - frameDatas[frames[0]].endOfIFD
                - frameDatas[frames[0]].imageLength*frameDatas[frames[0]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[0]].dataStripAddress - frameDatas[frames[0]].endOfIFD
    ;

    char* metaString = new char[initial_description_length];
    size_t descrLength = initial_description_length;
    
    // std::cmatch matches;
    
    for (uint64_t i = 0; i < framesN; i++) {
        // Reallocate if needed
        const size_t new_description_length = frameDatas[frames[i]].siffCompress ?
            frameDatas[frames[i]].dataStripAddress 
                - frameDatas[frames[i]].endOfIFD
                - frameDatas[frames[i]].imageLength*frameDatas[frames[i]].imageWidth*sizeof(uint16_t)
            :
            frameDatas[frames[i]].dataStripAddress - frameDatas[frames[i]].endOfIFD
        ;
        // need more space!
        if (new_description_length > descrLength) {
            delete[] metaString;
            metaString = new char[new_description_length];
            descrLength = new_description_length;
        }
        siff.seekg(frameDatas[frames[i]].endOfIFD, siff.beg);
        siff.read(metaString, descrLength);

        const char* patternStart = std::strstr(metaString, APPENDED_TEXT_PATTERN);
        if (patternStart != NULL) {

            // Find the end of the substring, or if TEXT_TIMESTAMP_PATTERN is found
            // before that, use that as the end of the substring
            const char* substringStart = patternStart + strlen(APPENDED_TEXT_PATTERN);
            const char* substringEnd = std::strstr(substringStart, TEXT_TIMESTAMP_PATTERN);

            // If TEXT_TIMESTAMP_PATTERN is found, calculate the length of the substring
            size_t substringLength = (substringEnd != nullptr) ?
                (substringEnd - substringStart)
                : std::strlen(substringStart);

            std::string appendedStr(
                patternStart + strlen(APPENDED_TEXT_PATTERN),
                substringLength
            );

            PyList_Append(
                retList,
                PyTuple_Pack(
                    3,
                    PyLong_FromUnsignedLongLong(frames[i]),
                    PyUnicode_FromString(appendedStr.c_str()),
                    substringEnd == nullptr ? (Py_INCREF(Py_None), Py_None) : PyFloat_FromDouble(
                        std::stod(substringEnd + strlen(TEXT_TIMESTAMP_PATTERN))
                    )
                )
            );
        }
    }

    delete[] metaString;

    return retList;
};