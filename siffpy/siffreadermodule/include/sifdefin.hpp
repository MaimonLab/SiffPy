#ifndef SIFDEFIN_HPP
#define SIFDEFIN_HPP

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define IMAGEWIDTH 256
#define IMAGELENGTH 257
#define BITSPERSAMPLE 258
#define COMPRESSION 259
#define PHOTOMETRIC_INTERPRETATION 262
#define IMAGEDESCRIPTION 270
#define STRIPOFFSETS 273
#define ORIENTATION 274
#define SAMPLESPERPIXEL 277
#define ROWSPERSTRIP 278
#define STRIPBYTECOUNTS 279
#define XRESOLUTION 282
#define YRESOLUTION 283
#define PLANARCONFIGURATION 284
#define RESOLUTIONUNIT 296
#define SOFTWAREPACKAGE 305
#define ARTIST 315
#define SAMPLEFORMAT 339
#define SIFFTAG 907

#define TIFF_BYTE 1
#define TIFF_ASCII 2
#define TIFF_SHORT 3
#define TIFF_LONG 4
#define TIFF_RATIONAL 5
#define TIFF_SBYTE 6
#define TIFF_UNDEFINE 7
#define TIFF_SSHORT 8
#define TIFF_SLONG 9
#define TIFF_SRATION 10
#define TIFF_FLOAT 11
#define TIFF_DOUBLE 12

// BigTIFF only
#define TIFF_LONG8 16
#define TIFF_SLONG8 17
#define TIFF_IFD8 18

inline uint16_t tiffDataType(uint16_t tiffTag){

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


inline uint16_t datatypeToCharCount(uint16_t typeTag) {
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