#include <stdio.h>
#include <fstream>
#include <stddef.h>
#include <stdlib.h>
#include <string>
#include <map>
#include "../include/sifftotiff.hpp"

// To parse the 64 bit photon reads
#define YMASK ((uint64_t) 1 << 63) - ((uint64_t) 1 << 48) + ((uint64_t) 1 << 63) // that last bit'll getcha
#define XMASK ((uint64_t) 1 << 48) - ((uint64_t) 1 << 32)
#define U64TOY(photon) (uint64_t) ((photon & YMASK) >> 48)
#define U64TOX(photon) (uint64_t) ((photon & XMASK) >> 32)

struct FrameData{
    uint64_t imageWidth;
    uint64_t imageLength;
    uint16_t bitsPerSample;
    uint16_t compression;
    uint16_t photometric;
    uint64_t endOfIFD;
    uint64_t dataStripAddress;
    uint16_t orientation;
    uint16_t samplesPerPixel;
    uint64_t rowsPerStrip;
    uint64_t stripByteCounts;
    uint64_t xResolution;
    uint64_t yResolution;
    uint16_t planarConfig;
    uint16_t resUnit;
    uint64_t NVFD_address;
    uint64_t ROI_address;
    uint16_t sampleFormat;
    std::string frameMetaData; // only populated when retrieving metadata
    bool siffCompress = false; // only present in .siff files and only in newer versions

    uint64_t stringlength;
};

// Writes all the contents of a FrameData and SiffParams object to a frame's IFD in a .tiff
// NOT DONE!!! TODO!!!
inline void writeTagData(uint64_t thisIFD, SiffParams& params, FrameData& thisFrameData, std::ofstream tiff) {
    
    uint64_t numTags = 18; // FIXED, contains all the above params except siffCompress and stringlength
    tiff.write((char*)&numTags, params.bytesPerNumTags);
    for (uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
        char thisTag[params.bytesPerTag];

    }
}

FrameData getTagData(uint64_t IFD, SiffParams& params, std::ifstream& siff){
    // return a FrameData structure that has parsed all the tag information at location:
    // IFD

    siff.clear();
    siff.seekg(IFD); // go there first

    if (!(siff.good())) throw std::runtime_error("Failure to find frame.");

    uint64_t numTags; // number of tags in this directory before the real metadata
    siff.read((char*)&numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.
    siff.clear();
    FrameData frameData;
    // tag parsing TODO: SHOULD BE INLINED SOMEWHERE ELSE
    for(uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
        char thisTag[params.bytesPerTag];
        siff.read(thisTag, params.bytesPerTag);
        
        uint16_t tagID = ((uint8_t) thisTag[(1-params.little)]) + (thisTag[params.little] << 8 );
        // figure out the number of bytes needed to read the data correctly
        uint16_t datatype = (uint8_t) thisTag[(3-params.little)] + (((uint8_t) thisTag[2+params.little]) << 8);
        uint16_t contentChars = datatypeToCharCount(datatype); // defined in sifdefin

        if (tagID == IMAGEDESCRIPTION) contentChars = 8; // UGH this is to correct a mistake I made early on
        // TODO: DO THIS RIGHT. I ALREADY KNOW THEY ALL ONLY USE A SINGLE TAG VALUE HERE BUT I SHOULD
        // MAKE THIS WORK FOR _ALL_ TIFFS
        
        // convert to a single value
        
        uint64_t contentVals = 0;
        // 8 + 4*bigtiff corresponds to the 4 bytes of identifier + the 4 bytes for number of values in tag (or 8 if bigtiff)
        for(int16_t charnum = (contentChars-1); 0<=charnum; charnum--) {
            contentVals <<= 8;
            contentVals += (thisTag[charnum + 8 + 4*params.bigtiff] & 0xFF); // gotta be honest... I don't understand why the  & 0xFF is necessary. Cut me some slack I learned C 2 months ago.
        }
        
        // now correct the typing if it's wrong
        switch(tagID){
            case IMAGEWIDTH:
                frameData.imageWidth = contentVals;
                break;
            case IMAGELENGTH:
                frameData.imageLength = contentVals;
                break;
            case BITSPERSAMPLE:
                frameData.bitsPerSample = (uint16_t) contentVals;
                if (params.issiff) frameData.bitsPerSample = 64; // this is a given... for now.
                break;
            case COMPRESSION:
                frameData.compression = (uint16_t) contentVals;
                break;
            case PHOTOMETRIC_INTERPRETATION:
                frameData.photometric = (uint16_t) contentVals;
                break;
            case IMAGEDESCRIPTION:
                frameData.endOfIFD = contentVals;
                break;
            case STRIPOFFSETS:
                frameData.dataStripAddress = contentVals;
                break;
            case ORIENTATION:
                frameData.orientation = (uint16_t) contentVals;
                break;
            case SAMPLESPERPIXEL:
                frameData.samplesPerPixel = (uint16_t) contentVals;
                break;
            case ROWSPERSTRIP:
                frameData.rowsPerStrip = contentVals;
                break;
            case STRIPBYTECOUNTS:
                frameData.stripByteCounts = contentVals;
                break;
            case XRESOLUTION:
                frameData.xResolution = contentVals;
                break;
            case YRESOLUTION:
                frameData.yResolution = contentVals;
                break;
            case PLANARCONFIGURATION:
                frameData.planarConfig = (uint16_t) contentVals;
                break;
            case RESOLUTIONUNIT:
                frameData.resUnit = (uint16_t) contentVals;
                break;
            case SOFTWAREPACKAGE:
                frameData.NVFD_address = contentVals;
                break;
            case ARTIST:
                frameData.ROI_address = contentVals;
                break;
            case SAMPLEFORMAT:
                frameData.sampleFormat = (uint16_t) contentVals;
                break;
            case SIFFTAG:
                frameData.siffCompress = (bool) contentVals;
                break;
            default:
                if (params.suppress_warnings) break;
                throw std::runtime_error("Invalid .siff tag detected!");
        }
        siff.clear();        
    }

    uint64_t description_length = frameData.siffCompress ?
        frameData.dataStripAddress - frameData.endOfIFD - frameData.imageLength*frameData.imageWidth*sizeof(uint16_t)
        :
        frameData.dataStripAddress - frameData.endOfIFD;

    frameData.stringlength = description_length;

    if (frameData.dataStripAddress<frameData.endOfIFD) throw std::runtime_error("Invalid data strip address read.");    
    siff.clear();
    return frameData;
}

// Takes info about the current (SiffCompressed) frame and the pointer to an array and populates it
void readCompressed(uint64_t samplesThisFrame, FrameData& thisFrameData, std::ifstream& siff, uint16_t* pxWiseData, uint64_t num_pixels) {
    // The intensity data is stored in the start of the frame, in the metadata! Easy!
    siff.seekg(thisFrameData.dataStripAddress - sizeof(uint16_t) * num_pixels);
    siff.read((char*) pxWiseData, num_pixels * sizeof(uint16_t));
    siff.clear();
}

void readRaw(uint64_t& samplesThisFrame, FrameData& thisFrameData, std::ifstream& siff, uint16_t* pxWiseData, uint64_t num_pixels) {
    // Need to extract the intensity data by going through each photon and taking just the x and y coordinates
    uint64_t frameReads[samplesThisFrame];
    siff.read((char*)&frameReads, thisFrameData.stripByteCounts);
    siff.clear();

    for (uint64_t photon = 0; photon < samplesThisFrame; photon++) {
        pxWiseData[
            U64TOY(frameReads[photon]) * thisFrameData.imageWidth +
            U64TOX(frameReads[photon])
        ]++;
    }
}

// Takes a pointer to a uint16_t array of length num_pixels and populates it with the data pointed to in the frame
void parseFrame(uint16_t* pxWiseData, uint64_t num_pixels, FrameData& thisFrameData, SiffParams& params, std::ifstream& siff){
    siff.clear();
    siff.seekg(thisFrameData.dataStripAddress);
    if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

    uint16_t bytesPerSample = thisFrameData.bitsPerSample/8;
    uint64_t samplesThisFrame = thisFrameData.stripByteCounts / bytesPerSample;
    siff.clear();

    thisFrameData.siffCompress ? 
        readCompressed(samplesThisFrame, thisFrameData, siff, pxWiseData, num_pixels) :
        readRaw(samplesThisFrame, thisFrameData, siff, pxWiseData, num_pixels);
}

// Writes a frame from a siff file into a frame in a tiff file, return the next IFD
uint64_t siff_to_tiff_frame(std::ifstream& siff, std::ofstream& tiff, SiffParams& params, uint64_t siffIFDAddress) {
    // Go to IFD
    siff.clear();

    // Parse the siff frame header data
    uint64_t numTags; // Tags in the siff frame
    siff.read((char*) &numTags, params.bytesPerNumTags);
    tiff.write((char*) &numTags, params.bytesPerNumTags);

    char tagData[params.bytesPerTag * numTags]; // read them all at once and copy them over
    siff.read(tagData, params.bytesPerTag * numTags);
    tiff.write(tagData, params.bytesPerTag * numTags);

    // where the next one will be in the read file.
    uint64_t nextIFD_SIFF;
    siff.read((char*)&nextIFD_SIFF, params.bytesPerPointer); 

    FrameData thisFrameData = getTagData(siffIFDAddress, params, siff);

    // num pixels
    uint64_t num_pixels = thisFrameData.imageLength * thisFrameData.imageWidth;

    uint64_t description_length = thisFrameData.siffCompress ?
        thisFrameData.dataStripAddress - thisFrameData.endOfIFD - num_pixels*sizeof(uint16_t)
        :
        thisFrameData.dataStripAddress - thisFrameData.endOfIFD;

    // Write when the next IFD will be in the TIFF
    uint64_t nextIFD_TIFF = thisFrameData.endOfIFD +
        description_length +
        sizeof(uint16_t)*num_pixels;
    
    if (nextIFD_SIFF == 0) nextIFD_TIFF = 0;
    
    tiff.write((char*) &nextIFD_TIFF, sizeof(uint64_t));

    
    char descriptionString[description_length];

    siff.seekg(thisFrameData.endOfIFD, siff.beg);
    siff.read(descriptionString, description_length);
    tiff.write(descriptionString, description_length);
    

    // The rest of the data is parsed differently if it's a siffcompress
    // or not.

    uint16_t pxWiseData[num_pixels];
    for(uint64_t px = 0; px < num_pixels; px++) pxWiseData[px] = 0;

    parseFrame(pxWiseData, num_pixels, thisFrameData, params, siff);
    
    tiff.write((char*) &pxWiseData, sizeof(uint16_t)*num_pixels);
   

    return nextIFD_SIFF;
}

// Converts a siff file to a tiff file once the streams are settled
void siff_to_tiff_method(std::ifstream& siff, std::ofstream& tiff, SiffParams params) {

    // Haven't made this endian proof yet. 
    // So check that your endian matches the file's
    uint16_t test = 1;
    char* bytewise = (char*) &test; // first address is 0 in big endian, 1 in little endian
    if (! (( (bool) bytewise) == params.little) ) throw std::runtime_error("Incompatible endians. Let me know that this is an issue, and I'll fix it -- SCT");

    // First, copy over the main data. Everything before the first IFD
    // is the same!
    siff.clear();
    siff.seekg(0, siff.beg);
    tiff.clear();
    tiff.seekp(0,tiff.beg);

    char init[params.firstIFDAddress];
    siff.read(init, params.firstIFDAddress);
    tiff.write(init, params.firstIFDAddress);

    // Now go through the frames and convert each to a tiff.
    uint64_t nextIFD_SIFF = params.firstIFDAddress;
    while (!(nextIFD_SIFF == 0)) {
        nextIFD_SIFF = siff_to_tiff_frame(siff, tiff, params, nextIFD_SIFF);
        if (nextIFD_SIFF >= params.fileSize) break;
        if (!siff.good()) throw std::runtime_error( std::string("Error writing siff file at byte ") + std::to_string(nextIFD_SIFF));
    }
}

// Uses the same methodology as the SiffReader to parse a siff file's parameters
SiffParams params_from_siff(std::ifstream& siff) {
    try{
    
        if (!(siff.is_open())) throw std::runtime_error("Could not open putative .siff file. Check that path exists.\n");
        
        SiffParams params;

        // Get the file size:
        siff.seekg(0, siff.end);
        params.fileSize = siff.tellg();
        siff.seekg(0, siff.beg);
        
        // Now check that it's a siff file.
        
        // Gotta know the endianness
        char * endian = new char[2];
        siff.read(endian, sizeof(char)*2);   

        // strcmp == 0 if they match.
        if ((strcmp(endian,BIGENDIAN) !=0) && (strcmp(endian,LITTLEENDIAN) != 0)) throw std::runtime_error("Could not deduce endian. May not be .siff/.tiff file. First two bytes (should be II or MM): "+std::string(endian));
        params.little = (strcmp(endian,LITTLEENDIAN) == 0); // true if little, false if big.

        delete[] endian;

        // temporary solution: if endian-ness doesn't match, give up.
        uint16_t i = 1; // the uint16_t 1 is 0x01 in big endian, 0x10 in little endian
        char* c = (char*)&i;
        // dereferencing c will be 1 if the least significant byte is first. 0 if not.
        if(! (((bool)*c) == params.little) ) throw std::runtime_error("ENDIANS DON'T MATCH AND I HAVEN'T FIXED THAT YET.");
        
        // Check the magic numbers
        uint16_t tiffid;
        siff.read((char*)&tiffid, sizeof(uint16_t));

        if(!((tiffid == BIGTIFFID) || (tiffid == TIFFID))) throw std::runtime_error("Could not verify that file is a true .tiff or .siff based on magic numbers.");
        params.bigtiff = (tiffid == BIGTIFFID);

        if (params.bigtiff) {
            //  here the headers diverge a bit
            uint16_t offset_size;
            siff.read((char*)&offset_size, sizeof(uint16_t));
            params.bytesPerPointer = offset_size; // sure to be 8 byte.
            params.bytesPerNumTags = 8;
            siff.read((char*)&offset_size, sizeof(uint16_t)); // these are always 0.
            if(offset_size) throw std::runtime_error("File is not a valid BIGTIFF or .SIFF.");
            
            uint64_t firstIFD;
            siff.read((char*)&firstIFD,params.bytesPerPointer);
            params.firstIFDAddress = firstIFD;
            params.bytesPerTag = 20;
        }
        else {
            // regular ol' tiff
            uint32_t firstIFD;
            siff.read((char*)&firstIFD, sizeof(uint32_t));
            params.firstIFDAddress = (uint64_t) firstIFD;
            params.bytesPerTag = 12;
            params.bytesPerNumTags = 2;
        }
    
        // Now do the ScanImage-specific checks!
        uint32_t magic;
        siff.read((char*)&magic, sizeof(uint32_t));
        
        uint32_t si;
        siff.read((char*)&si, sizeof(uint32_t));
        
        if( !( (magic == MAGICNUMBER) && (si == SI2019) ) ) throw std::runtime_error("File is a .tiff, but was not produced by ScanImage");

        // ScanImage data stuff
        uint32_t NVFD;
        siff.read((char*)&NVFD, sizeof(uint32_t));
        params.NVFD_length = NVFD; // in bytes

        uint32_t ROI;
        siff.read((char*)&ROI, sizeof(uint32_t));
        params.ROI_string_length = ROI; // in bytes

        
        char headerstring[params.NVFD_length];
        siff.read(headerstring, params.NVFD_length);
        params.headerstring = std::string(headerstring);

        char roistring[params.ROI_string_length];
        siff.read(roistring, params.ROI_string_length);
        params.ROI_string = std::string(roistring);
              
        return params;
    }
    catch(std::exception& e){
        if (siff.is_open()) siff.close();
        throw std::runtime_error(std::string("Could not parse parameters from file:\n") + e.what()); 
    }

}

// Converts a .siff file to a .tiff file
void siff_to_tiff(std::string sourcepath, std::string savepath) { 
    
    std::ifstream siff(sourcepath, std::ios::in | std::ios::binary);
    std::ofstream tiff(savepath, std::ios::out | std::ios::binary);
    SiffParams params = params_from_siff(siff);

    try{
        siff_to_tiff_method(siff, tiff, params); // the meat of things.
        siff.close();
        tiff.close();
    }

    catch(std::exception &e) { 
        siff.close();
        tiff.close();
        throw std::runtime_error(
            (
                std::string("Error encountered in siff_to_tiff: ") +
                e.what()
            ).c_str()
        );
    }

}

// Uses the sourcepath to identify the save path
void siff_to_tiff(std::string sourcepath) {
    std::string savepath = sourcepath.substr(0, sourcepath.find_last_of(".")) + ".tiff";
    siff_to_tiff(sourcepath, savepath);
}

