#include "../include/sifftotiff.hpp"

void siff_to_tiff(std::string sourcepath) {
    std::string savepath = sourcepath.substr(0, sourcepath.find_last_of(".")) + ".tiff";
    siff_to_tiff(sourcepath, savepath);
};

void siff_to_tiff(std::string sourcepath, std::string savepath){
    SiffReader *siffreader = new SiffReader();

    if (siffreader->openFile(sourcepath.c_str()) != 0) {
        delete(siffreader);
        throw std::runtime_error("Could not open putative .siff file. Check that path exists.\n");
    }

    // Could open the file, let's make a save file
    std::ofstream outfile(savepath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!outfile.is_open()) {
        delete(siffreader);
        throw std::runtime_error("Could not open save file for writing.\nPossible invalid path?");
    }

    // Transcribe the header file, most of which will be the same
    siffreader->writeParamsToHeader(outfile);

    const size_t nFrames = siffreader->numFrames();
    for (size_t i = 0; i < nFrames; i++) {
        siffreader->writeFrameAsTiff(outfile, i);
    }

    outfile.close();
    delete(siffreader);
};