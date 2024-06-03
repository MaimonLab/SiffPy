/**
 * @file libsiffreader.cpp
 * 
 * @brief This file contains the public-facing
 * SiffReader API implementation that's usable
 * by other C++ code, instead of being
 * forced to use the Python API. TODO: Make this
 * not rely on `numpy` so that this can be exported
 * (and compared against `corrosiff`!)
*/

#include "../../lib/libsiffreader.h"
//#include "../../include/siffreader/siffreader.hpp"
#include "../../include/sifftotiff.hpp"
#include <string.h>

#ifdef __cplusplus
extern "C"{
#endif // __cplusplus

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

API_EXPORT void load_intensity(
    const uint64_t* intensityArray,
    const uint64_t* frameNumbers
){
}

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
