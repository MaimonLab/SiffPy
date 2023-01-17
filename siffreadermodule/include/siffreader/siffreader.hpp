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

#include "siffParams.hpp"
#include <numpy/arrayobject.h>
#define PY_SSIZE_T_CLEAN

class SiffReader
{
    private:
        // the .siff file, a very delicately wrapped tiff
        std::ifstream siff;
        // file for debug logging.
        std::ofstream debugger;
        // Clock for measuring time to execute functions
        std::chrono::high_resolution_clock debug_clock;
        // fixed TIFF parameters invariant from frame to frame
        SiffParams params;

        // a setting to suppress potentially kernel-killing errors thrown by checks
        bool suppress_errors;
        // a setting to ignore errors if there is an issue reading one or more frames.
        bool suppress_warnings;
        // a setting for whether or not to print to the debugger file
        bool debug;             

        // error string for Python to retrieve
        std::string errstring;
        
        // quickly runs through the file to identify all frames and their IFDs, stores them in the SiffParams
        void discernFrames();
        
        // returns an ndarray object 
        PyArrayObject* frameAsNumpy(uint64_t IFD, bool flim, PyObject* shift_tuple = NULL);
        // returns an ndarray object with the terminal bins dropped
        PyArrayObject* frameAsNumpy(uint64_t IFD, uint64_t terminalBins, bool flim, PyObject* shift_tuple = NULL);

        // internal method called in loops to get each frame and point to the next
        void singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, PyObject* shift_tuple = NULL); 
        // internal method called in loops to get each frame and point to the next, drops photons later than the terminal bin
        void singleFrameRetrieval(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, uint64_t terminalBin, PyObject* shift_tuple = NULL);
        // internal method called in loops to get each frame and point to the next, MASKED BY AN ROI numpy array
        void singleFrameRetrievalMask(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, PyArrayObject* mask, PyObject* shift_tuple = NULL);

        // internal method to get metadata for one frame
        void singleFrameMetaData(uint64_t thisIFD, PyObject* metaDictList);
        // add this frame's arrival times to the 1-d array numpyArray
        void singleFrameHistogram(uint64_t thisIFD, PyArrayObject* numpyArray);
        
        // makes a tuple of [lifetime, intensity] from one frame, with lifetime unnormalized
        PyObject* makeFlimTuple(uint64_t IFD, PyObject* shift_tuple = NULL);
        // takes existing FlimTuple and merges in a new frame's data
        void fuseIntoFlimTuple(PyObject* FlimTup, uint64_t nextIFD, PyObject* shift_tuple = NULL);
        
        // takes a source frame and the pointer to the frame to fuse in
        void fuseFrames(PyArrayObject* sourceFrame, uint64_t nextIFD, bool flim, PyObject* shift_tuple = NULL);
        // takes a source frame and the address of the frame to fuse in, dropping photons after the terminalBin
        void fuseFrames(PyArrayObject* sourceFrame, uint64_t nextIFD, bool flim, uint64_t terminalBins, PyObject* shift_tuple = NULL);
        
        // takes a vector of photon reads and adds the next frame's reads to it.
        void fuseReadVector(std::vector<uint64_t>& photonReadsTogether, uint64_t nextIFD, PyObject* shift_tuple = NULL);
        
        // when you close one file and open another, wipe the slate clean
        void reset();


    public:
        SiffReader();
        //SiffReader operator=(const SiffReader& siffreader);
        ~SiffReader(){closeFile();};
        
        int openFile(const char* filename);
        bool isOpen();
        // the name of the .siff file
        std::string filename;

        // Mask methods
        PyArrayObject* sumMask(uint64_t frames[], uint64_t framesN, PyArrayObject* mask, PyObject* registrationDict); // sums area inside the provided mask
        PyArrayObject* sumFLIMMask(uint64_t frames[], uint64_t framesN, PyObject* FLIMParams, PyArrayObject* mask, PyObject* registrationDict); // sums empirical lifetime inside teh provided mask
        PyArrayObject* roiMask(uint64_t frames[], uint64_t framesN, bool flim, PyArrayObject* mask, PyObject* registrationDict); // Returns a 1d numpy array of only the within-mask values.

        PyObject* retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim);
        PyObject* retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim, PyObject* registrationDict);
        PyObject* retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim, PyObject* registrationDict, uint64_t terminalBin);
        PyObject* retrieveFrames(uint64_t frames[], uint64_t framesN, bool flim, PyObject* registrationDict, uint64_t terminalBin, PyArrayObject* mask);
        
        PyObject* poolFrames(PyObject* listOfList, bool flim); // TODO: IMPLEMENT
        PyObject* poolFrames(PyObject* listOfLists, bool flim = false, PyObject* registrationDict = NULL);
        PyObject* poolFrames(PyObject* listOfLists, uint64_t terminalBins, bool flim = false, PyObject* registrationDict = NULL);
        PyObject* poolFrames(PyObject* listOfLists, uint64_t terminalBins, PyArrayObject* mask, bool flim = false, PyObject* registrationDict = NULL);

        PyObject* flimMap(PyObject* FLIMParams, PyObject* listOfLists); // TODO: IMPLEMENT
        PyObject* flimMap(PyObject* FLIMParams, PyObject* listOfLists, PyObject* registrationDict = NULL); // returns array of lifetimes, intensity, NO confidence metric
        PyObject* flimMap(PyObject* FLIMParams, PyObject* listOfLists, const char* conf_measure, PyObject* registrationDict = NULL); // returns array of lifetimes, intensity, chi-sq
        PyObject* flimMap(PyObject* FLIMParams, PyObject* listOfLists, const char* conf_measure, PyArrayObject* mask, PyObject* registrationDict = NULL); // returns array of lifetimes, intensity, chi-sq, but restricted to a MASKED region

        PyArrayObject* getHistogram(uint64_t frames[] = NULL, uint64_t framesN = 0); // returns an arrival time vector, independent of pixel location.

        PyObject* readMetaData(uint64_t frames[]=NULL, uint64_t framesN=0); // get metadata enumerated in frames
        PyObject* readFixedData(); // returns the data in the primary ScanImage header
        std::string getNVFD();
        std::string getROIstring();
        
        void closeFile();

        const char* getErrString();

        // HEADER FUNCTION DEFS. BAD PRACTICE BUT LESS CLUTTERED.

        // Whether or not to elicit warnings when not-catastrophic events occur
        void suppressWarnings(bool suppress) {suppress_warnings = suppress; params.suppress_warnings = suppress;};
        // Number of frames in the file
        uint64_t numFrames(){return siff.is_open() ? params.allIFDs.size() : -1;};
        // Toggles debug mode on and off
        void setDebug(bool debug_bool) {debug = debug_bool;};

};

#endif
