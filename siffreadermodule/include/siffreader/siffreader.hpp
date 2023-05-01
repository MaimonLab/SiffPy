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

#include "../siffparams/siffparams.hpp"
#include "../framedata/framedatastruct.hpp"
#include "../framedata/pyFrameData.hpp"
#include <numpy/arrayobject.h>
#define PY_SSIZE_T_CLEAN

// Appends the error to errstring and rethrows()
#define REPORT_ERR(x) \
    catch(std::exception &e) {\
        errstring = std::string(x) + e.what();\
        throw e;\
    }

// Appends the error to errstring, executes y, and rethrows
#define REPORT_ERR_EXEC(x,y) \
    catch(std::exception &e) {\
        y; \
        errstring = std::string(x) + e.what();\
        throw e;\
    }

#define YMASK ((uint64_t) 1 << 63) - ((uint64_t) 1 << 48) + ((uint64_t) 1 << 63) // that last bit'll getcha
#define XMASK ((uint64_t) 1 << 48) - ((uint64_t) 1 << 32)
#define TAUMASK (((uint64_t) 1<<32) - 1)

// Get the y value of a siff pixel read (uncompressed)
#define U64TOY(photon) (uint64_t) ((photon & YMASK) >> 48)
// Get the x value of a siff pixel read (uncompressed)
#define U64TOX(photon) (uint64_t) ((photon & XMASK) >> 32)
// Get the arrival time of a siff pixel read (uncompressed)
#define U64TOTAU(photon) (uint64_t) (photon & TAUMASK)

// Converts an uncompressed read to a pixel index (y*x_dim + x)
#define READ_TO_PX(photon, shift_y, shift_x, dim_y, dim_x) \
    (((U64TOY(photon) + y_shift) % dim_y) * dim_x) \
    + (U64TOX(photon) +shift_x) % dim_x

// Shifts the pixel location px by shift_y, shift_x in an
// image of dimensions dim_y, dim_x
#define PIXEL_SHIFT(px, shift_y, shift_x, dim_y, dim_x) \
    (((uint64_t)(px / dim_y) + y_shift) % dim_y) * dim_x \
    + (((px % dim_y) + x_shift) % dim_x)


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
        std::vector<const FrameData> frameDatas;

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
        
        // internal method called in loops to get each frame and point to the next
        void singleFrameAddToList(
            const uint64_t thisIFD,
            PyObject* numpyArrayList,
            const bool flim,
            PyObject* shift_tuple = NULL
        ); 
        // internal method called in loops to get each frame and point to the next, MASKED BY AN ROI numpy array
        void singleFrameRetrievalMask(uint64_t thisIFD, PyObject* numpyArrayList, bool flim, PyArrayObject* mask, PyObject* shift_tuple = NULL);

        // internal method to get metadata for one frame
        void singleFrameMetaData(uint64_t thisIFD, PyObject* metaDictList);
        // add this frame's arrival times to the 1-d array numpyArray
        void singleFrameHistogram(const uint64_t thisIFD, PyArrayObject* numpyArray);
        
        // takes a vector of photon reads and adds the next frame's reads to it.
        void fuseReadVector(
            std::vector<uint64_t>& photonReadsTogether,
            uint64_t nextIFD,
            PyObject* shift_tuple = NULL
        );
        
        // when you close one file and open another, wipe the slate clean
        void reset();


    public:
        SiffReader();
        //SiffReader operator=(const SiffReader& siffreader);
        ~SiffReader(){closeFile();};
        
        int openFile(const char* filename);
        void closeFile();

        bool isOpen();
        // the name of the .siff file
        std::string filename;

        void packFrameDataList(PyObject* frameDataList);

        ////// Mask methods /////////

        // sums area inside the provided mask
        PyArrayObject* sumMask(
            const uint64_t frames[],
            const uint64_t framesN,
            PyArrayObject* mask,
            PyObject* registrationDict
        );
    
        // sums empirical lifetime inside the provided mask
        // IMPORTANT: If mask has more than 2 dimensions, it
        // presumes that the frames list iterates through those
        // dimensions in the same order as the mask!!!
        // NOT smart enough to know better.
        PyArrayObject* sumFLIMMask(
            const uint64_t frames[],
            const uint64_t framesN,
            PyObject* FLIMParams,
            PyArrayObject* mask,
            PyObject* registrationDict
        );
        
        // Returns a 1d numpy array of only the within-mask values.
        PyArrayObject* roiMask(uint64_t frames[], uint64_t framesN, bool flim, PyArrayObject* mask, PyObject* registrationDict);

        /////// Frame methods /////////

        // returns whether the dimensions of all the frames are consistent;
        bool dimensionsConsistent(const uint64_t frames[], const uint64_t framesN);

        // Returns a list of length framesN, with each
        // element an array of dimensions y, x
        PyObject* retrieveFrames(const uint64_t frames[], const uint64_t& framesN, PyObject* registrationDict);

        // Returns an array of dimensions: framesN, y, x
        PyArrayObject* retrieveFramesAsArray(
            const uint64_t frames[],
            const uint64_t framesN,
            PyObject* registrationDict
        );

        // Takes a list of lists, and for each
        // internal list sums the frames and returns
        // them as an array. Returns an array of
        // dimensions : length of outer list, y, x
        PyArrayObject* poolFrames(
            PyObject* listOfLists,
            const bool &flim,
            PyObject* registrationDict = NULL
        );

        // returns a tuple: arrays of lifetimes, intensity, goodness-of-fit
        PyObject* flimTuple(
            PyObject* FLIMParams,
            const uint64_t frames[],
            const uint64_t framesN,
            const char* conf_measure,
            PyObject* registrationDict = NULL
        );

        // returns an arrival time vector, independent of pixel location.
        PyArrayObject* getHistogram(const uint64_t frames[] = NULL, const uint64_t framesN = 0);


        /////// Metadata methods /////////
        // get metadata enumerated in frames
        PyObject* readMetaData(uint64_t frames[]=NULL, uint64_t framesN=0);
        // returns the data in the primary ScanImage header
        PyObject* readFixedData();
        
        std::string getNVFD();
        std::string getROIstring();

        /////// Debug methods /////////
        
        // Returns a string of the last error thrown
        const char* getErrString();

        // Whether or not to elicit warnings when not-catastrophic events occur
        void suppressWarnings(bool suppress);

        // Number of frames in the file
        uint64_t numFrames();

        // Toggles debug mode on and off
        void setDebug(bool debug_bool);
};

#endif
