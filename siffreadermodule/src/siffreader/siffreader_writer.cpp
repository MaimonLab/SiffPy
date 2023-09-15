#include "../../include/siffreader/siffreader.hpp"

void SiffReader::writeParamsToHeader(std::ofstream& outfile) const {
    outfile.seekp(0);
    
    // Write the endian
    params.little ? outfile.write(LITTLEENDIAN, sizeof(char)*2) : outfile.write(BIGENDIAN, sizeof(char)*2);
    // Write the bigtiff spec
    params.bigtiff ? outfile.write((char*)&BIGTIFFID, sizeof(uint16_t)) : outfile.write((char*)&TIFFID, sizeof(uint16_t));

    // Write the number of bytes per pointer
    if (params.bigtiff) {
        constexpr uint16_t bytesPerPointer = 8;
        outfile.write((char*)&bytesPerPointer, sizeof(uint16_t));
        // Tiff spec goofiness
        constexpr uint16_t zero = 0;
        outfile.write((char*)&zero, sizeof(uint16_t));
    }
    outfile.write((char*)&params.firstIFDAddress, params.bytesPerPointer);

    // Write the magic numbers for ScanImage
    outfile.write((char*)&MAGICNUMBER, sizeof(uint32_t));
    outfile.write((char*)&SI2019, sizeof(uint32_t));

    // Write the non-varying frame data length
    outfile.write((char*)&params.NVFD_length, sizeof(uint32_t));

    // Write the ROI string length
    outfile.write((char*)&params.ROI_string_length, sizeof(uint32_t));

    // Write the non-varying frame data
    outfile.write(params.headerstring.c_str(), params.NVFD_length);
    outfile.write(params.ROI_string.c_str(), params.ROI_string_length);

    // Make sure the current position is the supposed first IFD location
    if (outfile.tellp() != params.firstIFDAddress){
        throw std::runtime_error("Error writing header: written first IFD address is not where file ended up after writing headers.");
    }
};

void SiffReader::writeFrameAsTiff(std::ofstream& outfile, const uint64_t& frameNum) const {
    // First write the frame data, header etc.
    const FrameData frameData = frameDatas[frameNum];
    writeFrameDataAsTiff(frameData, outfile, params);
};