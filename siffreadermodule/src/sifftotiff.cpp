#include "../include/sifftotiff.hpp"

void siff_to_tiff(std::string sourcepath) {
    std::string savepath = sourcepath.substr(0, sourcepath.find_last_of(".")) + ".tiff";
    siff_to_tiff(sourcepath, savepath, SCANIMAGE);
};

void siff_to_tiff(std::string sourcepath, TiffMode mode) {
    std::string extension;
    switch(mode) {
        case SCANIMAGE:
            extension = ".tiff";
            break;
        case OME:
            extension = ".ome.tiff";
            break;
        default:
            throw std::runtime_error("Invalid tiff mode specified.");
    }
    std::string savepath = sourcepath.substr(0, sourcepath.find_last_of(".")) + extension;
    siff_to_tiff(sourcepath, savepath, mode);
};

void siff_to_tiff(std::string sourcepath, std::string savepath) {
    siff_to_tiff(sourcepath, savepath, SCANIMAGE);
};

void siff_to_tiff(std::string sourcepath, std::string savepath, TiffMode mode) {

    std::string extension;
    switch(mode) {
        case SCANIMAGE:
            extension = ".tiff";
            break;
        case OME:
            extension = ".ome.tiff";
            break;
        default:
            throw std::runtime_error("Invalid tiff mode specified.");
    }
    std::ofstream outfile;
    SiffReader *siffreader = new SiffReader();
    try{

        if (siffreader->openFile(sourcepath.c_str()) != 0) {
            delete(siffreader);
            throw std::runtime_error("Could not open putative .siff file. Check that path exists.\n");
        }

        // Could open the file, let's make a save file
        std::ofstream outfile = std::ofstream(savepath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!outfile.is_open()) {
            delete(siffreader);
            throw std::runtime_error("Could not open save file for writing.\nPossible invalid path?");
        }

        // Transcribe the header file, most of which will be the same
        siffreader->writeParamsToHeader(outfile);

        switch (mode) {
            case SCANIMAGE:
                siffreader->writeFrameAsTiff(outfile, 0);
                break;
            case OME:
                siffreader->writeOMEXMLFrame(outfile, 0);
                break;
            default:
                delete(siffreader);
                throw std::runtime_error("Invalid tiff mode specified.");
        }

        const size_t nFrames = siffreader->numFrames();
        for (size_t i = 1; i < nFrames; i++) {
            siffreader->writeFrameAsTiff(outfile, i);
        }

        outfile.close();
        delete(siffreader);
    }
    catch(std::exception& e) {
        outfile.close();
        delete(siffreader);
        if(!remove(savepath.c_str())) {
            throw std::runtime_error(
                std::string("Error writing .tiff file with additional failure to delete")
                + " partially written file. Check permissions? Exception: \n" + e.what()
            );
        }
        throw e;
    }
};
//