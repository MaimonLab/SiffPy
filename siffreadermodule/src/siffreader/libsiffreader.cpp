/**
 * @file libsiffreader.cpp
 * 
 * @brief This file contains the public-facing
 * SiffReader API implementation that's usable
 * by other C++ code, instead of being
 * forced to use the Python API.
*/

#include "../../lib/libsiffreader.h"
//#include "../../include/siffreader/siffreader.hpp"
#include "../../include/sifftotiff.hpp"
#include <string.h>

API_EXPORT void read_siff_file(
    const char* file_path
){
    //SiffReader reader;
    //return reader.read_siff_file(file_path);
}

API_EXPORT void siff_to_tiff(
    const char* sourcepath
){
    return siff_to_tiff(
        std::string(sourcepath)
    );
    //SiffReader reader;
    //reader.siff_to_tiff(sourcepath);
}

