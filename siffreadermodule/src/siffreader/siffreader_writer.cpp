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

// A constant number of tags are written to the tiff file. These are
// specified by me. 
constexpr uint64_t numTagsWrittenToTiffs = 18;

/*
Local function that writes the tag to the tiff file.
*/
void writeTagTiff(
        std::ofstream& outfile,
        const uint16_t tagID,
        const uint64_t contentVals
    ){
    outfile.write((char*)&tagID, sizeof(uint16_t));
    const uint16_t dtype = tiffDataType(tagID);
    outfile.write((char*)&dtype, sizeof(uint16_t));
    // This is NOT generic, but it's fine for the current version
    // of ScanImage
    constexpr uint32_t numberOfExtries = 1;
    outfile.write((char*)&numberOfExtries, sizeof(uint32_t));

    const uint32_t trueContentVals = contentVals;

    outfile.write((char*)&trueContentVals, sizeof(uint32_t));

};

/*
Local function that writes the tag to the bigtiff file.
*/
void writeTagBigTiff(
        std::ofstream& outfile,
        const uint16_t tagID,
        const uint64_t contentVals
    ) {
    outfile.write((char*)&tagID, sizeof(uint16_t));
    const uint16_t dtype = tiffDataType(tagID);
    outfile.write((char*)&dtype, sizeof(uint16_t));
    // This is NOT generic, but it's fine for the current version
    // of ScanImage
    constexpr uint64_t numberOfEntries = 1;
    outfile.write((char*)&numberOfEntries, sizeof(uint64_t));
    outfile.write((char*)&contentVals, sizeof(uint64_t));
};

void SiffReader::writeFrameAsTiff(std::ofstream& outfile, const uint64_t frame) const {

    const FrameData frameData = frameDatas[frame];
    // Write the frame data in a format consistent with the file type
    outfile.write((char*)&numTagsWrittenToTiffs, params.bytesPerNumTags);

    uint16_t dtype;
    const size_t startOfTags = outfile.tellp();
    const size_t endOfTags = startOfTags + params.bytesPerTag*numTagsWrittenToTiffs
        + params.bytesPerPointer; // for the nextIFD pointer

    // Writes the tag of appropriate size for this format
    void (*tagWriteFunction) (std::ofstream&, const uint16_t, const uint64_t);

    if (params.bigtiff) {
        tagWriteFunction = &writeTagBigTiff;
    } else {
        tagWriteFunction = &writeTagTiff;
    }

    tagWriteFunction(outfile, IMAGEWIDTH, frameData.imageWidth);
    tagWriteFunction(outfile, IMAGELENGTH, frameData.imageLength);
    tagWriteFunction(outfile, BITSPERSAMPLE, 8*sizeof(uint16_t));
    tagWriteFunction(outfile, COMPRESSION, 1);
    tagWriteFunction(outfile, PHOTOMETRIC_INTERPRETATION, 1);
    tagWriteFunction(outfile, IMAGEDESCRIPTION, endOfTags);
    
    const size_t lengthOfDescription = frameData.siffCompress ? 
        frameData.dataStripAddress - frameData.endOfIFD - frameData.imageLength*frameData.imageWidth*sizeof(uint16_t)
        :
        frameData.dataStripAddress - frameData.endOfIFD;

    tagWriteFunction(outfile, STRIPOFFSETS, endOfTags + lengthOfDescription);
    tagWriteFunction(outfile, ORIENTATION, frameData.orientation);
    tagWriteFunction(outfile, SAMPLESPERPIXEL, 1);
    tagWriteFunction(outfile, ROWSPERSTRIP, frameData.rowsPerStrip);
    tagWriteFunction(outfile, STRIPBYTECOUNTS, sizeof(uint16_t)*frameData.imageLength*frameData.imageWidth);
    tagWriteFunction(outfile, XRESOLUTION, frameData.xResolution);
    tagWriteFunction(outfile, YRESOLUTION, frameData.yResolution);
    tagWriteFunction(outfile, PLANARCONFIGURATION, frameData.planarConfig);
    tagWriteFunction(outfile, RESOLUTIONUNIT, frameData.resUnit);
    tagWriteFunction(outfile, SOFTWAREPACKAGE, frameData.NVFD_address);
    tagWriteFunction(outfile, ARTIST, frameData.ROI_address);
    tagWriteFunction(outfile, SAMPLEFORMAT, 1);

    // Write when the end of the frame will be, i.e. next IFD
    const size_t nextIFD = frame == (params.numFrames - 1) ? 0 :
        endOfTags
        + lengthOfDescription
        + (frameData.imageLength*frameData.imageWidth*sizeof(uint16_t))
    ;

    // Points to nextIFD, we'll come back to it and rewrite it if there was a mistake,
    // corrupting this frame but hopefully not the next.
    const size_t nextIFDPointer = outfile.tellp();
    outfile.write((char*)&nextIFD, params.bytesPerPointer);

    // Now write the actual metadata and frame data
    // First read from the siff

    char* descriptionBuffer = new char[lengthOfDescription];
    siff.seekg(frameData.endOfIFD, std::ios::beg);
    siff.read(descriptionBuffer, lengthOfDescription);
    outfile.write(descriptionBuffer, lengthOfDescription);
    delete[] descriptionBuffer;

    // Now write the frame data
    uint16_t* photonCounts = new uint16_t[frameData.imageLength*frameData.imageWidth]{0};
    loadArrayWithData(photonCounts,params,frameData,siff);
    outfile.write(
        (char*)photonCounts,
        frameData.imageLength*frameData.imageWidth*sizeof(uint16_t)
    );
    delete[] photonCounts; 

    // Now check that the next IFD pointer is where it should be
    if ((nextIFD > 0) & (outfile.tellp() != nextIFD)) {
        throw std::runtime_error(
            "Error writing frame: next IFD pointer is not where it should be. "
            "Current position: " + std::to_string(outfile.tellp()) + ", "
            "Expected position: " + std::to_string(nextIFD)
        );
    }
};