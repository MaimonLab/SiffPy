#ifndef SIFFTOTIFF_HPP
#define SIFFTOTIFF_HPP
#include <stdio.h>
#include <string>
#include <map>

#ifndef PY_SSIZE_T_CLEAN
    #define PY_SSIZE_T_CLEAN
#endif

#include "siffparams/siffparams.hpp"
#include "framedata/sifdefin.hpp"
#include "siffreader/siffreader.hpp"

/**
 * The TiffMode enum is used to select the tiff mode to be used
 * (OME compliant not yet implemented...)
*/
typedef enum TiffMode {
    SCANIMAGE = 0,
    OME = 1
} TiffMode;

/**
 * Converts a siff file to a tiff file by reading each frame and
 * re-writing the intensity data to the tiff format. Writes to
 * the same location as the source file. Uses the ScanImage tiff mode

 * @param sourcepath the path to the siff file to be converted
*/
void siff_to_tiff(std::string sourcepath);

/**
* Converts a siff file to a tiff file by reading each frame and
* re-writing the intensity data to the tiff format. Allows selection
* of the tiff mode.

* @param sourcepath the path to the siff file to be converted
* @param mode the tiff mode to be used
*/
void siff_to_tiff(std::string sourcepath, TiffMode mode);

/**
 * Converts a siff file to a tiff file by reading each frame and
 * re-writing the intensity data to the tiff format. Uses the ScanImage
 * tiff mode.

 * @param sourcepath the path to the siff file to be converted
 * @param savepath the path to the tiff file to be written
*/
void siff_to_tiff(std::string sourcepath, std::string savepath);

/**
 * Converts a siff file to a tiff file by reading each frame and
 * re-writing the intensity data to the tiff format. Allows selection
 * of the tiff mode.

 * @param sourcepath the path to the siff file to be converted
 * @param savepath the path to the tiff file to be written
 * @param mode the tiff mode to be used
*/
void siff_to_tiff(std::string sourcepath, std::string savepath, TiffMode mode);

#endif