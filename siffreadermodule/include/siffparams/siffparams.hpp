#ifndef SIFFPARAMS_HPP
#define SIFFPARAMS_HPP

#define LITTLEENDIAN "II"
#define BIGENDIAN "MM"
constexpr uint16_t TIFFID = 42;
constexpr uint16_t BIGTIFFID = 43;
constexpr uint32_t MAGICNUMBER = 117637889; // identifies this as a scanimage file
constexpr uint32_t SI2019 = 4; // identifies this as 2019 or later

#include <stdlib.h>
#include <string>
#include <vector>

typedef struct SiffParams{
    std::string filename;
    bool little;
    bool bigtiff;
    bool issiff;
    bool flim;
    bool debug;
    bool suppress_warnings;

    uint16_t bytesPerPointer; // bytes per offset value in each image
    uint16_t bytesPerTag; // bigTIFF uses 20 bytes, while TIFF uses 12.
    uint16_t bytesPerNumTags; // bigTIFF uses 8 bytes, TIFF uses 2
    uint64_t firstIFDAddress; // address of the first IFD, containing metadata per frame
    uint32_t NVFD_length; // length of the non-varying frame data
    uint32_t ROI_string_length; // length of the ROI string data
    std::string headerstring; // the header string
    std::string ROI_string; // the ROI string

    uint64_t fileSize;
    uint64_t numROIs; // number of ROIs (not implemented yet)
    uint64_t numFrames; // number of frames
    uint64_t numTimepoints; // number of timepoints (degenerate, = numFrames/(numROIs*numZStacks*numColors))
    uint64_t numZStacks; // number of z stacks
    uint64_t numColors; // number of color channels
    std::vector<uint64_t> allIFDs;
} SiffParams;

#endif