#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#include "../include/framedata/framedatastruct.hpp"
#include "../include/framedata/sifdefin.hpp"
#include "../include/siffparams/siffparams.hpp"

const FrameData getTagData(uint64_t IFD, SiffParams& params, std::ifstream& siff){
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
                PyErr_WarnEx(PyExc_RuntimeWarning,
                    (std::string("INVALID TIFF TAG DETECTED: ") + std::to_string(tagID) +  "\n" +
                    std::string("To suppress this warning in the future, call siffreader.suppress_warnings()")).c_str(),
                    Py_ssize_t(1)
                );
        }
        siff.clear();        
    }

    if (frameData.dataStripAddress<frameData.endOfIFD) throw std::runtime_error("Invalid data strip address read.");    
    siff.clear();
    return frameData;
}