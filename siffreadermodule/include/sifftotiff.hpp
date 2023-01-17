#ifndef SIFFTOTIFF_HPP
#define SIFFTOTIFF_HPP
#include <stdio.h>
#include <string>
#include <map>

#include "siffreader/siffParams.hpp"
#include "siffreader/sifdefin.hpp"

void siff_to_tiff(std::string sourcepath);
void siff_to_tiff(std::string sourcepath, std::string savepath);

#endif