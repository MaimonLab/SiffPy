#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#include "../include/framedata/framedatastruct.hpp"
#include "../include/framedata/sifdefin.hpp"
#include "../include/siffparams/siffparams.hpp"
#include "../include/ome.hpp"

PyObject* frameDataToDict(FrameData& frameData){
    PyObject* dataDict = PyDict_New();

    PyDict_SetItemString(dataDict, "Width", Py_BuildValue("K", frameData.imageWidth));
    PyDict_SetItemString(dataDict, "Length", Py_BuildValue("K", frameData.imageLength));
    PyDict_SetItemString(dataDict, "endOfIFD", Py_BuildValue("K", frameData.endOfIFD));
    PyDict_SetItemString(dataDict, "dataStripAddress",Py_BuildValue("K", frameData.dataStripAddress));
    PyDict_SetItemString(dataDict, "stringLength", Py_BuildValue("K", frameData.stringlength));
    PyDict_SetItemString(dataDict, "X Resolution", Py_BuildValue("K", frameData.xResolution));
    PyDict_SetItemString(dataDict, "YResolution", Py_BuildValue("K", frameData.yResolution));
    PyDict_SetItemString(dataDict, "Bytecount", Py_BuildValue("K", frameData.stripByteCounts));
    PyDict_SetItemString(dataDict, "Frame metadata", Py_BuildValue("s#", frameData.frameMetaData.c_str(), frameData.stringlength));
    //PyDict_SetItemString(dataDict, "Tag bytes", Py_BuildValue("O",frameData.tagList));
    PyDict_SetItemString(dataDict, "Siff compression", Py_BuildValue("O",frameData.siffCompress ? Py_True : Py_False));

    return dataDict;
};

// const FrameData getTagData(const uint64_t IFD, const SiffParams& params, std::ifstream& siff){
//     // return a FrameData structure that has parsed all the tag information at location:
//     // IFD

//     siff.clear();
//     siff.seekg(IFD); // go there first

//     if (!(siff.good())) throw std::runtime_error("Failure to find frame.");

//     uint64_t numTags; // number of tags in this directory before the real metadata
//     siff.read((char*)&numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.
//     siff.clear();
//     FrameData frameData;

//     char* thisTag = new char[params.bytesPerTag];

//     uint16_t tagID; // the tag identifier
//     uint16_t datatype; // the datatype of the tag (coded)
//     uint16_t contentChars; // the number of characters in the tag
//     uint64_t contentVals; // the value of the tag

//     // tag parsing TODO: SHOULD BE INLINED SOMEWHERE ELSE
//     for(uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
//         siff.read(thisTag, params.bytesPerTag);
        
//         tagID = ((uint8_t) thisTag[(1-params.little)]) + (thisTag[params.little] << 8 );
//         // figure out the number of bytes needed to read the data correctly
//         datatype = (uint8_t) thisTag[(3-params.little)] + (((uint8_t) thisTag[2+params.little]) << 8);
//         contentChars = datatypeToCharCount(datatype); // defined in sifdefin

//         if (tagID == IMAGEDESCRIPTION) contentChars = 8; // UGH this is to correct a mistake I made early on
        
//         // convert to a single value
        
//         contentVals = 0;
//         // 8 + 4*bigtiff corresponds to the 4 bytes of identifier + the 4 bytes for number of values in tag (or 8 if bigtiff)
//         for(int16_t charnum = (contentChars-1); 0<=charnum; charnum--) {
//             contentVals <<= 8;
//             contentVals += (thisTag[charnum + 8 + 4*params.bigtiff] & 0xFF); // gotta be honest... I don't understand why the  & 0xFF is necessary. Cut me some slack I learned C 2 months ago.
//         }
        
//         // now correct the typing if it's wrong
//         switch(tagID){
//             case IMAGEWIDTH:
//                 frameData.imageWidth = contentVals;
//                 break;
//             case IMAGELENGTH:
//                 frameData.imageLength = contentVals;
//                 break;
//             case BITSPERSAMPLE:
//                 frameData.bitsPerSample = (uint16_t) contentVals;
//                 if (params.issiff) frameData.bitsPerSample = 64; // this is a given... for now.
//                 break;
//             case COMPRESSION:
//                 frameData.compression = (uint16_t) contentVals;
//                 break;
//             case PHOTOMETRIC_INTERPRETATION:
//                 frameData.photometric = (uint16_t) contentVals;
//                 break;
//             case IMAGEDESCRIPTION:
//                 frameData.endOfIFD = contentVals;
//                 break;
//             case STRIPOFFSETS:
//                 frameData.dataStripAddress = contentVals;
//                 break;
//             case ORIENTATION:
//                 frameData.orientation = (uint16_t) contentVals;
//                 break;
//             case SAMPLESPERPIXEL:
//                 frameData.samplesPerPixel = (uint16_t) contentVals;
//                 break;
//             case ROWSPERSTRIP:
//                 frameData.rowsPerStrip = contentVals;
//                 break;
//             case STRIPBYTECOUNTS:
//                 frameData.stripByteCounts = contentVals;
//                 break;
//             case XRESOLUTION:
//                 frameData.xResolution = contentVals;
//                 break;
//             case YRESOLUTION:
//                 frameData.yResolution = contentVals;
//                 break;
//             case PLANARCONFIGURATION:
//                 frameData.planarConfig = (uint16_t) contentVals;
//                 break;
//             case RESOLUTIONUNIT:
//                 frameData.resUnit = (uint16_t) contentVals;
//                 break;
//             case SOFTWAREPACKAGE:
//                 frameData.NVFD_address = contentVals;
//                 break;
//             case ARTIST:
//                 frameData.ROI_address = contentVals;
//                 break;
//             case SAMPLEFORMAT:
//                 frameData.sampleFormat = (uint16_t) contentVals;
//                 break;
//             case SIFFTAG:
//                 frameData.siffCompress = (bool) contentVals;
//                 break;
//             default:
//                 if (params.suppress_warnings) break;
//                 PyErr_WarnEx(PyExc_RuntimeWarning,
//                     (std::string("INVALID TIFF TAG DETECTED: ") + std::to_string(tagID) +  "\n" +
//                     std::string("To suppress this warning in the future, call siffreader.suppress_warnings()")).c_str(),
//                     Py_ssize_t(1)
//                 );
//         }
//         siff.clear();        
//     }
//     delete[] thisTag;
    
//     if (frameData.dataStripAddress<frameData.endOfIFD) throw std::runtime_error("Invalid data strip address read.");    
//     siff.clear();
//     return frameData;
// }

const FrameData getTagData(const uint64_t IFD, const SiffParams& params, std::ifstream& siff){
    // return a FrameData structure that has parsed all the tag information at location:
    // IFD

    siff.clear();
    siff.seekg(IFD); // go there first

    if (!(siff.good())) throw std::runtime_error("Failure to find frame.");

    uint64_t numTags; // number of tags in this directory before the real metadata
    siff.read((char*)&numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.
    siff.clear();
    FrameData frameData;

    char* thisTag = new char[params.bytesPerTag];

    uint16_t tagID; // the tag identifier
    uint16_t datatype; // the datatype of the tag (coded)
    uint16_t contentChars; // the number of characters in the tag
    uint64_t contentVals; // the value of the tag

    // tag parsing TODO: SHOULD BE INLINED SOMEWHERE ELSE
    for(uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
        siff.read(thisTag, params.bytesPerTag);
        
        tagID = ((uint8_t) thisTag[(1-params.little)]) + (thisTag[params.little] << 8 );
        // figure out the number of bytes needed to read the data correctly
        datatype = (uint8_t) thisTag[(3-params.little)] + (((uint8_t) thisTag[2+params.little]) << 8);
        contentChars = datatypeToCharCount(datatype); // defined in sifdefin

        if (tagID == IMAGEDESCRIPTION) contentChars = 8; // UGH this is to correct a mistake I made early on
        
        // convert to a single value
        
        contentVals = 0;
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
    delete[] thisTag;
    
    if (frameData.dataStripAddress<frameData.endOfIFD) throw std::runtime_error("Invalid data strip address read.");    
    siff.clear();
    return frameData;
}

const FrameData getTagData(const uint64_t IFD, const SiffParams& params, std::ifstream& siff, std::ofstream& logstream){
    // return a FrameData structure that has parsed all the tag information at location:
    // IFD

    siff.clear();
    siff.seekg(IFD); // go there first

    DEBUG(logstream << "Reading tag data from IFD " << IFD << std::endl;)

    if (!(siff.good())) throw std::runtime_error("Failure to find frame.");

    uint64_t numTags; // number of tags in this directory before the real metadata
    siff.read((char*)&numTags, params.bytesPerNumTags); // this style should avoid hairiness of bigtiff vs tiff spec.
    siff.clear();
    FrameData frameData;

    char* thisTag = new char[params.bytesPerTag];

    uint16_t tagID; // the tag identifier
    uint16_t datatype; // the datatype of the tag (coded)
    uint16_t contentChars; // the number of characters in the tag
    uint64_t contentVals; // the value of the tag

    DEBUG(logstream << "Reading " << numTags << " tags." << std::endl;)

    // tag parsing TODO: SHOULD BE INLINED SOMEWHERE ELSE
    for(uint64_t tagNum = 0; tagNum < numTags; tagNum++) {
        siff.read(thisTag, params.bytesPerTag);

        DEBUG(logstream << "Reading tag " << tagNum << std::endl;)
        
        tagID = ((uint8_t) thisTag[(1-params.little)]) + (thisTag[params.little] << 8 );
        // figure out the number of bytes needed to read the data correctly
        datatype = (uint8_t) thisTag[(3-params.little)] + (((uint8_t) thisTag[2+params.little]) << 8);
        contentChars = datatypeToCharCount(datatype); // defined in sifdefin

        if (tagID == IMAGEDESCRIPTION) contentChars = 8; // UGH this is to correct a mistake I made early on
        
        // convert to a single value

        DEBUG(logstream << "Tag ID: " << tagID << std::endl;)
        
        contentVals = 0;
        // 8 + 4*bigtiff corresponds to the 4 bytes of identifier + the 4 bytes for number of values in tag (or 8 if bigtiff)
        for(int16_t charnum = (contentChars-1); 0<=charnum; charnum--) {
            contentVals <<= 8;
            contentVals += (thisTag[charnum + 8 + 4*params.bigtiff] & 0xFF); // gotta be honest... I don't understand why the  & 0xFF is necessary. Cut me some slack I learned C 2 months ago.
        }
        
        DEBUG(logstream << "Content value: " << contentVals << std::endl;)

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
    delete[] thisTag;

    DEBUG(logstream << "Done reading tags." << std::endl;)
    
    if (frameData.dataStripAddress<frameData.endOfIFD) throw std::runtime_error("Invalid data strip address read.");    
    siff.clear();

    DEBUG(logstream << "Returning frame data." << std::endl;)
    return frameData;
}

const std::string toOMEXML(const FrameData& frameData, const SiffParams& params){
    std::string retString(OME_WARNING_BLOCK);

    return retString;
}