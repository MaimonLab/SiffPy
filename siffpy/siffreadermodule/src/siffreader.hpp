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
        std::string filename; // the name of the file
        SiffParams params; // fixed TIFF parameters

        bool suppress_errors = false; // a setting to suppress potentially kernel-killing errors thrown by checks
        bool suppress_warnings = false; // a setting to ignore errors if there is an issue reading one or more frames.
        bool debug = true;

        std::string errstring; // error string to retrieve
        void discernFrames(); // quickly runs through the file to identify all frames and their IFDs
        PyArrayObject* frameAsNumpy(uint64_t IFD, bool flim); // returns an ndarray object 
        void singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim); // internal method called in loops to get each frame and point to the next
        void singleFrameMetaData(uint64_t thisIFD, PyObject* metaDictList); // internal method to get metadata for one frame
        void singleFrameHistogram(uint64_t thisIFD, PyArrayObject* numpyArray); // add this frame's arrival times to the 1-d array numpyArray
        void fuseFrames(PyArrayObject* sourceFrame, uint64_t nextIFD, bool flim); // takes a source frame and the address of the frame to fuse in
        void fuseReadVector(std::vector<uint64_t>& photonReadsTogether, uint64_t nextIFD); // takes a vector of photon reads and adds the next frame's reads to it.
        uint64_t nextIFD;
        uint64_t getFollowingIFD(uint64_t currIFD);

        void reset(); // when you close one file and open another, wipe the slate clean

    public:
        SiffReader(){};
        ~SiffReader(){closeFile();};
        int openFile(const char* filename);
        PyObject* retrieveFrames(uint64_t frames[]=NULL, uint64_t framesN=0, bool flim = false);
        PyObject* poolFrames(PyObject* listOfLists, bool flim = false);
        PyObject* readMetaData(uint64_t frames[]=NULL, uint64_t framesN=0); // get metadata enumerated in frames
        PyObject* readFixedData(); // returns the data in the primary ScanImage header
        PyObject* flimMap(PyObject* FLIMParams, PyObject* listOfLists, const char* conf_measure); // returns array of lifetimes, intensity, chi-sq
        PyArrayObject* getHistogram(uint64_t frames[] = NULL, uint64_t framesN = 0); // returns an arrival time vector, independent of pixel location.
        std::string getNVFD();
        std::string getROIstring();
        
        const char* getErrString();
        void closeFile();

        // HEADER FUNCTION DEFS. BAD PRACTICE BUT LESS CLUTTERED.
        void suppressWarnings(bool suppress) {suppress_warnings = suppress; params.suppress_warnings = suppress;};
        uint64_t numFrames(){return siff.is_open() ? params.allIFDs.size() : -1;};

};

#endif
