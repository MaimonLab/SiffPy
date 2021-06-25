#ifndef SIFFREADER_HPP
#define SIFFREADER_HPP

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string>
#include <fstream>

// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes

#include "../include/siffParams.hpp"
#include <numpy/arrayobject.h>

class SiffReader
{
    private:
        std::ifstream siff; // the .siff file, a very delicately wrapped tiff
        std::ofstream debugger; // file for debug logging.
        std::chrono::high_resolution_clock debug_clock;
        std::string filename; // the name of the file
        SiffParams params; // fixed TIFF parameters

        bool suppress_errors;   // a setting to suppress potentially kernel-killing errors thrown by checks
        bool suppress_warnings; // a setting to ignore errors if there is an issue reading one or more frames.
        bool debug;             // a setting for whether or not to print to the debugger file

        std::string errstring;  // error string to retrieve
        
        void discernFrames();   // quickly runs through the file to identify all frames and their IFDs
        
        PyArrayObject* frameAsNumpy(uint64_t IFD, bool flim, PyObject* shift_tuple = NULL); // returns an ndarray object 
        PyArrayObject* frameAsNumpy(uint64_t IFD, uint64_t terminalBins, bool flim, PyObject* shift_tuple = NULL); // returns an ndarray object 

        void singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, PyObject* shift_tuple = NULL); // internal method called in loops to get each frame and point to the next
        void singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, uint64_t terminalBin, PyObject* shift_tuple = NULL); // internal method called in loops to get each frame and point to the next
        void singleFrameMetaData(uint64_t thisIFD, PyObject* metaDictList); // internal method to get metadata for one frame
        void singleFrameHistogram(uint64_t thisIFD, PyArrayObject* numpyArray); // add this frame's arrival times to the 1-d array numpyArray
        
        PyObject* makeFlimTuple(uint64_t IFD, PyObject* shift_tuple = NULL); // makes a tuple of [lifetime, intensity] from one frame, with lifetime unnormalized
        void fuseIntoFlimTuple(PyObject* FlimTup, uint64_t nextIFD, PyObject* shift_tuple = NULL); // takes existing FlimTuple and merges in a new frame's data
        
        void fuseFrames(PyArrayObject* sourceFrame, uint64_t nextIFD, bool flim, PyObject* shift_tuple = NULL); // takes a source frame and the address of the frame to fuse in
        void fuseFrames(PyArrayObject* sourceFrame, uint64_t nextIFD, bool flim, uint64_t terminalBins, PyObject* shift_tuple = NULL); // takes a source frame and the address of the frame to fuse in
        
        void fuseReadVector(std::vector<uint64_t>& photonReadsTogether, uint64_t nextIFD, PyObject* shift_tuple = NULL); // takes a vector of photon reads and adds the next frame's reads to it.
        uint64_t nextIFD;
        uint64_t getFollowingIFD(uint64_t currIFD);

        void reset(); // when you close one file and open another, wipe the slate clean

    public:
        SiffReader();
        ~SiffReader(){closeFile();};
        
        int openFile(const char* filename);
        
        PyObject* retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim);
        PyObject* retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim, PyObject* registrationDict);
        PyObject* retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim, PyObject* registrationDict, uint64_t terminalBin);
        
        PyObject* poolFrames(PyObject* listOfLists, bool flim = false, PyObject* registrationDict = NULL);
        PyObject* poolFrames(PyObject* listOfLists, uint64_t terminalBins, bool flim = false, PyObject* registrationDict = NULL);

        PyObject* flimMap(PyObject* FLIMParams, PyObject* listOfLists, PyObject* registrationDict = NULL); // returns array of lifetimes, intensity, NO confidence metric
        PyObject* flimMap(PyObject* FLIMParams, PyObject* listOfLists, const char* conf_measure, PyObject* registrationDict = NULL); // returns array of lifetimes, intensity, chi-sq

        PyArrayObject* getHistogram(uint64_t frames[] = NULL, uint64_t framesN = 0); // returns an arrival time vector, independent of pixel location.

        PyObject* readMetaData(uint64_t frames[]=NULL, uint64_t framesN=0); // get metadata enumerated in frames
        PyObject* readFixedData(); // returns the data in the primary ScanImage header
        std::string getNVFD();
        std::string getROIstring();
        
        const char* getErrString();
        
        void closeFile();

        // HEADER FUNCTION DEFS. BAD PRACTICE BUT LESS CLUTTERED.
        void suppressWarnings(bool suppress) {suppress_warnings = suppress; params.suppress_warnings = suppress;};
        uint64_t numFrames(){return siff.is_open() ? params.allIFDs.size() : -1;};
        void setDebug(bool debug_bool) {debug = debug_bool;};

};

#endif
