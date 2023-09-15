#ifndef SIFFTOTIFF_HPP
#define SIFFTOTIFF_HPP
#include <stdio.h>
#include <string>
#include <map>

#include "siffparams/siffparams.hpp"
#include "framedata/sifdefin.hpp"
#include "siffreader/siffreader.hpp"

/*
Converts a siff file to a tiff file by reading each frame and
re-writing the intensity data to the tiff format. Writes to
the same location as the source file.

@param sourcepath the path to the siff file to be converted
*/
void siff_to_tiff(std::string sourcepath);

/*
Converts a siff file to a tiff file by reading each frame and
re-writing the intensity data to the tiff format

@param sourcepath the path to the siff file to be converted
@param savepath the path to the tiff file to be written
*/
void siff_to_tiff(std::string sourcepath, std::string savepath);

#endif