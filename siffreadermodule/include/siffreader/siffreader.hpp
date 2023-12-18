#ifndef SIFFREADER_HPP
#define SIFFREADER_HPP

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string>
#include <fstream>
//#include <sys/mman.h>
#include <chrono>

// IMPORT_ARRAY() CALLED IN MODULE INIT
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL siff_ARRAY_API

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // yikes

#include "bitmasks.hpp"
#include "../siffparams/siffparams.hpp"
#include "../framedata/framedatastruct.hpp"
#include "../framedata/pyFrameData.hpp"
#include "../debug.hpp"
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

/*
Loads an array with the intensity data from the
provided frame. No pixel shift, intended to load
a C-style array.

@param data_ptr
    A pointer to the array to load with the data.

@param params
    The `SiffParams` object containing the parameters of the
    file being read from.

@param frameData
    The `FrameData` object containing the parameters of the
    frame being read from.

@param siff
    The `std::ifstream` object managing the file being read from.
*/
void loadArrayWithData(
    uint16_t* data_ptr,
    const SiffParams& params,
    const FrameData& frameData,
    std::ifstream& siff
);

/*
Implemented in `siffreader/intensity_methods`.

Loads an array with the intensity data from the
provided frame. Shifts pixels according to the
registration parameters. Intended for loading
a `numpy` array

@param data_ptr
    A pointer to the data portion of array to load with the data.

@param dims
    The dimensions of the array to load with the data.

@param params
    The `SiffParams` object containing the parameters of the
    file being read from.

@param frameData
    The `FrameData` object containing the parameters of the
    frame being read from.

@param siff
    The `std::ifstream` object managing the file being read from.

@param shift_tuple
    A `PyObject*` to a tuple of the form (y_shift, x_shift) to
    shift the pixels by. If NULL, no shifting is performed.
*/
void loadArrayWithData(
    uint16_t* data_ptr,
    const npy_intp* dims,
    const SiffParams& params,
    const FrameData& frameData,
    std::ifstream& siff,
    PyObject* shift_tuple
);


/*
The SiffReader is the primary interface for reading and
transcribing .siff files.
*/
class SiffReader
{
    private:
        size_t _numFrames;
        // the .siff file, a very delicately wrapped tiff
        // Mutable: the file stream is expected to change
        // in almost all cases, so the meaning of const
        // should ignore changes to this object.
        mutable std::ifstream siff;
        
        DEBUG(
            // file for debug logging.
            mutable std::ofstream logstream;
            // Clock for measuring time to execute functions
            std::chrono::high_resolution_clock debug_clock;
        )
        
        // fixed TIFF parameters invariant from frame to frame
        SiffParams params;

        std::vector<FrameData> frameDatas;

        // used when debugging
        DEBUG(
            mutable std::chrono::high_resolution_clock::time_point tick;
            mutable std::chrono::high_resolution_clock::time_point tock;
        )

        // a setting to suppress potentially kernel-killing errors thrown by checks
        bool suppress_errors;
        // a setting to ignore errors if there is an issue reading one or more frames.
        bool suppress_warnings;
        // a setting for whether or not to print to the debugger file
        bool debug;             

        // Error string for Python to retrieve. Mutable because its modification
        // isn't a meaningful change to the obejct
        mutable std::string errstring;
        
        // Quickly runs through the file to identify all frames and their IFDs, stores them in the SiffParams
        void discernFrames();
        
        // internal method called in loops to get each frame and point to the next
        void singleFrameAddToList(
            const uint64_t thisIFD,
            PyObject* numpyArrayList,
            const bool flim,
            PyObject* shift_tuple = NULL
        ) const;
        
        // internal method to get metadata for one frame
        void singleFrameMetaData(const uint64_t& thisIFD, PyObject* metaDictList) const;
        // add this frame's arrival times to the 1-d array numpyArray
        void singleFrameHistogram(const uint64_t& thisIFD, PyArrayObject* numpyArray) const;
        
        // when you close one file and open another, wipe the slate clean
        void reset();

    public:
        /*
        Initializes a SiffReader, which does nothing until it
        opens a file with the function `openFile`.
        */
        SiffReader();
        //SiffReader operator=(const SiffReader& siffreader);
        ~SiffReader(){closeFile();};
        
        /*
        Opens the .siff file at the provided filename.

        @param filename
            The name of the .siff file to open.

        @return 0 if successful, -1 if unsuccessful.
        */
        int openFile(const char* filename);

        // Close the `ifstream` internal reader
        void closeFile();

        /*
        Whether the `ifstream` internal reader is open.

        @return Whether the `ifstream` internal reader is open.
        */
        bool isOpen() const;
        
        // The name of the .siff file
        std::string filename;

        /*
        Populates the input `PyObject*` with the
        `FrameData` objects for each frame in the
        file. NOT ACTUALLY IMPLEMENTED, JUST SETS THE
        PYERR STRING.

        @param frameDataList
            A `PyObject*` to a PyList to populate with the
            `FrameData` objects.
        */
        void packFrameDataList(PyObject* frameDataList) const;

        /*
        Takes the `SiffParams` internal object and uses it
        to construct a header file for a .tiff file or
        .siff file.

        @param outfile
            the `std::ofstream` object managing the
            file to write the header to.
        */
        void writeParamsToHeader(std::ofstream& outfile) const;

        /*
        Copies the frame at the provided frame number to
        the provided `std::ofstream` object as it would
        be formatted in an OME-TIFF frame. Expects the
        `std::ofstream` object to be at the correct
        position to write the frame. Note that this
        frame will NOT contain the standard ScanImage metadata.

        @param outfile
            The `std::ofstream` object managing the
            file to write the frame to. Ensure that
            it is at the correct position to write
            the frame.

        @param frame
            The frame number to rewrite.
        */
       void writeOMEXMLFrame(std::ofstream& outfile, const uint64_t frame) const;

        /*
        Copies the frame at the provided frame number to
        the provided `std::ofstream` object as it would
        be formatted in a .tiff frame. Expects the
        `std::ofstream` object to be at the correct
        position to write the frame.

        @param outfile
            The `std::ofstream` object managing the
            file to write the frame to. Ensure that
            it is at the correct position to write
            the frame.

        @param frame
            The frame number to rewrite.
        */
        void writeFrameAsTiff(std::ofstream& outfile, const uint64_t frame) const;

        ////// Mask methods /////////

        /*
        Sums pixels in the area inside the provided mask

        @param frames 
            the frames to be summed
        
        @param framesN
            the number of frames to be summed

        @param mask
            A `PyArrayObject*` to a `numpy` array of the mask
            of `dtype bool`

        @param registrationDict
            A `PyObject*` to a dictionary of registration
            parameters. The keys are frame numbers and the
            values are tuples of the form (`yshift`, `xshift`)

        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: (framesN,)
        */
        PyArrayObject* sumMask(
            const uint64_t frames[],
            const uint64_t framesN,
            PyArrayObject* mask,
            PyObject* registrationDict
        ) const;
    
        /*
        Sums empirical lifetime inside the provided mask
        IMPORTANT: If mask has more than 2 dimensions, it
        presumes that the frames list iterates through those
        dimensions in the same order as the mask!!!
        NOT smart enough to know better.
        
        @param frames The frames to be summed
        @param framesN The number of frames to be summed
        @param FLIMParams A `PyObject*` to a `FLIMParams` object
        @param mask A `PyArrayObject*` to a `numpy` array of the mask
        of `dtype bool`
        @param registrationDict A `PyObject*` to a dictionary of registration
        parameters. The keys are frame numbers and the
        values are tuples of the form (`yshift`, `xshift`)
        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: framesN, y, x
        */
        PyArrayObject* sumFLIMMask(
            const uint64_t frames[],
            const uint64_t framesN,
            PyObject* FLIMParams,
            PyArrayObject* mask,
            PyObject* registrationDict
        );
        
        /*
        TODO:
        Returns a 1d numpy array of only the within-mask values.    
        */
        //PyArrayObject* roiMask(uint64_t frames[], uint64_t framesN, bool flim, PyArrayObject* mask, PyObject* registrationDict);

        /////// Frame methods /////////

        /*
        Returns whether the dimensions of all the frames are consistent;
        that is, whether they all have the same dimensions.

        @param frames
            The frames to check for consistency.

        @param framesN
            The number of frames to check for consistency.
        
        @return Whether the dimensions of all the frames are consistent.
        */
        bool dimensionsConsistent(const uint64_t frames[], const uint64_t framesN) const;

        /*
        Returns a list of length framesN, with each
        element an array of dimensions y, x

        @param frames
            The frames to retrieve
        
        @param framesN
            The number of frames to retrieve

        @param registrationDict
            A `PyObject*` to a dictionary of registration
            parameters. The keys are the frame number,
            and the values are a tuple of the form
            (`yshift`, `xshift`)

        @return A `PyObject*` to a list of `PyArrayObject*`s
        each of dimensions y,x
        */
        PyObject* retrieveFrames(
            const uint64_t frames[],
            const uint64_t& framesN,
            PyObject* registrationDict
        ) const;

        /*
        Returns an array of dimensions: framesN, y, x

        @param frames
            The frames to retrieve
        
        @param framesN
            The number of frames to retrieve

        @param registrationDict
            A `PyObject*` to a dictionary of registration
            parameters. The keys are the frame number,
            and the values are a tuple of the form
            (`yshift`, `xshift`)

        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: framesN, y, x
        */
        PyArrayObject* retrieveFramesAsArray(
            const uint64_t frames[],
            const uint64_t framesN,
            PyObject* registrationDict
        );

        /*
        Takes a list of lists, and for each
        internal list sums the frames and returns
        them as an array. Returns an array of
        dimensions : length of outer list, y, x

        @param listOfLists
            A `PyObject*` to a list of lists of frames
            to be summed. Each inner list will be summed
            together, and the result will be an array of
            dimensions: len(listOfLists), y, x

        @param flim
            Whether or not to sum the frames as FLIM arrays
            with a tau dimension

        @param registrationDict
            A `PyObject*` to a dictionary of registration
            parameters. The keys are the frame number,
            and the values are a tuple of the form
            (`yshift`, `xshift`)

        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: len(listOfLists), y, x
        */
        PyArrayObject* poolFrames(
            PyObject* listOfLists,
            const bool &flim,
            PyObject* registrationDict = NULL
        );

        /*
        Returns a tuple: arrays of lifetimes, intensity, goodness-of-fit

        @param FLIMParams
            A `PyObject*` to a `FLIMParams` object

        @param frames
            The frames to retrieve

        @param framesN
            The number of frames to retrieve

        @param registrationDict
            A `PyObject*` to a dictionary of registration
            parameters. The keys are the frame number,
            and the values are a tuple of the form
            (`yshift`, `xshift`)

        @return A `PyObject*` to a tuple of 3 `PyArrayObject*`s
        each of dimensions framesN,y,x
        */
        PyObject* flimTuple(
            PyObject* FLIMParams,
            const uint64_t frames[],
            const uint64_t framesN,
            const char* conf_measure,
            PyObject* registrationDict = NULL
        );

        /*
        Returns an arrival time vector, independent of pixel location.
        The vector is of length framesN

        @param frames
            The frames to retrieve

        @param framesN
            The number of frames to retrieve
        
        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: framesN
        */
        PyArrayObject* getHistogram(const uint64_t frames[] = NULL, const uint64_t framesN = 0);


        /////// Metadata methods /////////
        /*
        Read the metadata in a set of frames and return
        it as a list of dictionaries of metadata contents.

        @param frames
            The frames to retrieve metadata from

        @param framesN
            The number of frames to retrieve metadata from

        @return A `PyObject*` to a list of `PyDict`s
        each of which contains the metadata for a frame.
        */
        PyObject* readMetaData(
            const uint64_t frames[],
            const uint64_t framesN
        ) const ;

        /*
        @return Returns the data in the primary ScanImage header
        as a dict
        */
        PyObject* readFixedData() const;

        /*
        Return the timestamps of each frame in the "experiment"
        time coordinates (0 from start of acquisition) measured
        entirely with laser pulses.

        @param frames
            The frames to retrieve timestamps from

        @param framesN
            The number of frames to retrieve timestamps from

        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: (framesN,)
        */
        PyArrayObject* getExperimentTimestamps(
            const uint64_t frames[],
            const uint64_t framesN
        ) const ;

        /*
        Return the timestamps of each frame in the "system"
        time coordinates (in epoch time) measured
        entirely with laser pulses -- no drift correction!

        @param frames
            The frames to retrieve timestamps from

        @param framesN
            The number of frames to retrieve timestamps from

        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: (framesN,)
        */
        PyArrayObject* getEpochTimestampsLaser(
            const uint64_t frames[],
            const uint64_t framesN
        ) const ;

        /*
        Return the timestamps of each frame in the "system"
        time coordinates (in epoch time) measured
        entirely with clock checks -- no correction for the
        latency between reading the MultiHarp buffer and
        when the clock check is actually performed.

        @param frames
            The frames to retrieve timestamps from

        @param framesN
            The number of frames to retrieve timestamps from

        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: (framesN,)
        */
        PyArrayObject* getEpochTimestampsSystem(
            const uint64_t frames[],
            const uint64_t framesN
        ) const ;

        /*
        Return the timestamps of each frame in the "system"
        time coordinates (in epoch time) measured
        both with laser clock and system clock checks, one
        in each row.

        @param frames
            The frames to retrieve timestamps from

        @param framesN
            The number of frames to retrieve timestamps from

        @return A `PyArrayObject*` to a `numpy` array of
        dimensions: (2, framesN)
        */
        PyArrayObject* getEpochTimestampsBoth(
            const uint64_t frames[],
            const uint64_t framesN
        ) const ;
        
        /*
        @return Returns the data in the primary ScanImage header
        (aka NVFD, non-varying frame data)
        */
        std::string getNVFD() const;

        /*
        @return Returns the data in the ROI string containing
        all the mROI data.
        */
        std::string getROIstring() const;

        /////// Debug methods /////////
        
        // Returns a string of the last error thrown
        const char* getErrString() const;

        /*
        Whether or not to elicit warnings when not-catastrophic events occur

        @param suppress
            Whether or not to suppress warnings
        */
        void suppressWarnings(bool suppress);

        // Number of frames in the file
        uint64_t numFrames() const;

        // Toggles debug mode on and off
        void setDebug(bool debug_bool);

        DEBUG(
            // Returns the latest debug tick-tock
            std::string printLastTickTock();
            void toDebugLog(const std::string& message) const;
        )
};

#endif
