#ifndef SIFDEFIN_HPP
#define SIFDEFIN_HPP

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

constexpr uint16_t IMAGEWIDTH = 256;
constexpr uint16_t IMAGELENGTH = 257;
constexpr uint16_t BITSPERSAMPLE = 258;
constexpr uint16_t COMPRESSION = 259;
constexpr uint16_t PHOTOMETRIC_INTERPRETATION = 262;
constexpr uint16_t IMAGEDESCRIPTION = 270;
constexpr uint16_t STRIPOFFSETS = 273;
constexpr uint16_t ORIENTATION = 274;
constexpr uint16_t SAMPLESPERPIXEL = 277;
constexpr uint16_t ROWSPERSTRIP = 278;
constexpr uint16_t STRIPBYTECOUNTS = 279;
constexpr uint16_t XRESOLUTION = 282;
constexpr uint16_t YRESOLUTION = 283;
constexpr uint16_t PLANARCONFIGURATION = 284;
constexpr uint16_t RESOLUTIONUNIT = 296;
constexpr uint16_t SOFTWAREPACKAGE = 305;
constexpr uint16_t ARTIST = 315;
constexpr uint16_t SAMPLEFORMAT = 339;
constexpr uint16_t SIFFTAG = 907;

constexpr uint16_t TIFF_BYTE = 1;
constexpr uint16_t TIFF_ASCII = 2;
constexpr uint16_t TIFF_SHORT = 3;
constexpr uint16_t TIFF_LONG = 4;
constexpr uint16_t TIFF_RATIONAL = 5;
constexpr uint16_t TIFF_SBYTE = 6;
constexpr uint16_t TIFF_UNDEFINE = 7;
constexpr uint16_t TIFF_SSHORT = 8;
constexpr uint16_t TIFF_SLONG = 9;
constexpr uint16_t TIFF_SRATION = 10;
constexpr uint16_t TIFF_FLOAT = 11;
constexpr uint16_t TIFF_DOUBLE = 12;

// BigTIFF only
constexpr uint16_t TIFF_LONG8 = 16;
constexpr uint16_t TIFF_SLONG8 = 17;
constexpr uint16_t TIFF_IFD8 = 18;

inline const uint16_t tiffDataType(const uint16_t tiffTag){

    switch(tiffTag) {
        case IMAGEWIDTH:
            return TIFF_LONG;
        case IMAGELENGTH:
            return TIFF_LONG;
        case BITSPERSAMPLE:
            return TIFF_SHORT;
        case COMPRESSION:
            return TIFF_SHORT;
        case PHOTOMETRIC_INTERPRETATION:
            return TIFF_SHORT;
        case IMAGEDESCRIPTION:
            return TIFF_LONG8;
        case STRIPOFFSETS:
            return TIFF_LONG8;
        case ORIENTATION:
            return TIFF_SHORT;
        case SAMPLESPERPIXEL:
            return TIFF_SHORT;
        case ROWSPERSTRIP:
            return TIFF_LONG;
        case STRIPBYTECOUNTS:
            return TIFF_LONG8;
        case XRESOLUTION:
            return TIFF_RATIONAL;
        case YRESOLUTION:
            return TIFF_RATIONAL;
        case PLANARCONFIGURATION:
            return TIFF_SHORT;
        case RESOLUTIONUNIT:
            return TIFF_SHORT;
        case SOFTWAREPACKAGE:
            return TIFF_ASCII;
        case ARTIST:
            return TIFF_ASCII;
        case SAMPLEFORMAT:
            return TIFF_SHORT;
        case SIFFTAG:
            return TIFF_BYTE;
        default:
            return 0;
    }
};


inline const uint16_t datatypeToCharCount(const uint16_t typeTag) {
    switch(typeTag) {
        
        case TIFF_BYTE:
            return 1;
        case TIFF_ASCII:
            return 1;
        case TIFF_SHORT: 
            return 2;
        case TIFF_LONG:
            return 4;
        case TIFF_RATIONAL:
            return 8; // not standard TIFF form actually, usually two longs
        case TIFF_SBYTE: 
            return 1;
        case TIFF_UNDEFINE:
            return 1;
        case TIFF_SSHORT:
            return 2;
        case TIFF_SLONG:
            return 2;
        case TIFF_SRATION:
            return 2;
        case TIFF_FLOAT:
            return 4;
        case TIFF_DOUBLE:
            return 8;
        case TIFF_LONG8:
            return 8;
        case TIFF_SLONG8:
            return 8;
        case TIFF_IFD8:
            return 8;
        default: // UNKNOWN TYPE TREAT AS UINT64
            return 8;
    }
};


#endif