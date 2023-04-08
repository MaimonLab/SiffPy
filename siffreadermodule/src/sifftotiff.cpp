#include <stdio.h>
#include <fstream>
#include <stddef.h>
#include <stdlib.h>
#include <string>
#include <map>
#include "../include/sifftotiff.hpp"
#include "../include/siffparams/siffparams.hpp"
#include "../include/framedata/framedatastruct.hpp"



// // Takes a pointer to a uint16_t array of length num_pixels and populates it with the data pointed to in the frame
// void parseFrame(uint16_t* pxWiseData, uint64_t num_pixels, FrameData& thisFrameData, SiffParams& params, std::ifstream& siff){
//     siff.clear();
//     siff.seekg(thisFrameData.dataStripAddress);
//     if (!(siff.good() || params.suppress_warnings)) throw std::runtime_error("Failure to navigate to data in frame.");

//     uint16_t bytesPerSample = thisFrameData.bitsPerSample/8;
//     uint64_t samplesThisFrame = thisFrameData.stripByteCounts / bytesPerSample;
//     siff.clear();

//     thisFrameData.siffCompress ? 
//         readCompressed(samplesThisFrame, thisFrameData, siff, pxWiseData, num_pixels) :
//         readRaw(samplesThisFrame, thisFrameData, siff, pxWiseData, num_pixels);
// }

// // Writes a frame from a siff file into a frame in a tiff file, return the next IFD
// uint64_t siff_to_tiff_frame(std::ifstream& siff, std::ofstream& tiff, SiffParams& params, uint64_t siffIFDAddress) {
//     // Go to IFD
//     siff.clear();

//     // Parse the siff frame header data
//     uint64_t numTags; // Tags in the siff frame
//     siff.read((char*) &numTags, params.bytesPerNumTags);
//     tiff.write((char*) &numTags, params.bytesPerNumTags);

//     char tagData[params.bytesPerTag * numTags]; // read them all at once and copy them over
//     siff.read(tagData, params.bytesPerTag * numTags);
//     tiff.write(tagData, params.bytesPerTag * numTags);

//     // where the next one will be in the read file.
//     uint64_t nextIFD_SIFF;
//     siff.read((char*)&nextIFD_SIFF, params.bytesPerPointer); 

//     FrameData thisFrameData = getTagData(siffIFDAddress, params, siff);

//     // num pixels
//     uint64_t num_pixels = thisFrameData.imageLength * thisFrameData.imageWidth;

//     uint64_t description_length = thisFrameData.siffCompress ?
//         thisFrameData.dataStripAddress - thisFrameData.endOfIFD - num_pixels*sizeof(uint16_t)
//         :
//         thisFrameData.dataStripAddress - thisFrameData.endOfIFD;

//     // Write when the next IFD will be in the TIFF
//     uint64_t nextIFD_TIFF = thisFrameData.endOfIFD +
//         description_length +
//         sizeof(uint16_t)*num_pixels;
    
//     if (nextIFD_SIFF == 0) nextIFD_TIFF = 0;
    
//     tiff.write((char*) &nextIFD_TIFF, sizeof(uint64_t));

    
//     char descriptionString[description_length];

//     siff.seekg(thisFrameData.endOfIFD, siff.beg);
//     siff.read(descriptionString, description_length);
//     tiff.write(descriptionString, description_length);
    

//     // The rest of the data is parsed differently if it's a siffcompress
//     // or not.

//     uint16_t pxWiseData[num_pixels];
//     for(uint64_t px = 0; px < num_pixels; px++) pxWiseData[px] = 0;

//     parseFrame(pxWiseData, num_pixels, thisFrameData, params, siff);
    
//     tiff.write((char*) &pxWiseData, sizeof(uint16_t)*num_pixels);
   

//     return nextIFD_SIFF;
// }

// // Converts a siff file to a tiff file once the streams are settled
// void siff_to_tiff_method(std::ifstream& siff, std::ofstream& tiff, SiffParams params) {

//     // Haven't made this endian proof yet. 
//     // So check that your endian matches the file's
//     uint16_t test = 1;
//     char* bytewise = (char*) &test; // first address is 0 in big endian, 1 in little endian
//     if (! (( (bool) bytewise) == params.little) ) throw std::runtime_error("Incompatible endians. Let me know that this is an issue, and I'll fix it -- SCT");

//     // First, copy over the main data. Everything before the first IFD
//     // is the same!
//     siff.clear();
//     siff.seekg(0, siff.beg);
//     tiff.clear();
//     tiff.seekp(0,tiff.beg);

//     char init[params.firstIFDAddress];
//     siff.read(init, params.firstIFDAddress);
//     tiff.write(init, params.firstIFDAddress);

//     // Now go through the frames and convert each to a tiff.
//     uint64_t nextIFD_SIFF = params.firstIFDAddress;
//     while (!(nextIFD_SIFF == 0)) {
//         nextIFD_SIFF = siff_to_tiff_frame(siff, tiff, params, nextIFD_SIFF);
//         if (nextIFD_SIFF >= params.fileSize) break;
//         if (!siff.good()) throw std::runtime_error( std::string("Error writing siff file at byte ") + std::to_string(nextIFD_SIFF));
//     }
// }

// // Uses the same methodology as the SiffReader to parse a siff file's parameters
// SiffParams params_from_siff(std::ifstream& siff) {
//     try{
    
//         if (!(siff.is_open())) throw std::runtime_error("Could not open putative .siff file. Check that path exists.\n");
        
//         SiffParams params;

//         // Get the file size:
//         siff.seekg(0, siff.end);
//         params.fileSize = siff.tellg();
//         siff.seekg(0, siff.beg);
        
//         // Now check that it's a siff file.
        
//         // Gotta know the endianness
//         char * endian = new char[2];
//         siff.read(endian, sizeof(char)*2);   

//         // strcmp == 0 if they match.
//         if ((strcmp(endian,BIGENDIAN) !=0) && (strcmp(endian,LITTLEENDIAN) != 0)) throw std::runtime_error("Could not deduce endian. May not be .siff/.tiff file. First two bytes (should be II or MM): "+std::string(endian));
//         params.little = (strcmp(endian,LITTLEENDIAN) == 0); // true if little, false if big.

//         delete[] endian;

//         // temporary solution: if endian-ness doesn't match, give up.
//         uint16_t i = 1; // the uint16_t 1 is 0x01 in big endian, 0x10 in little endian
//         char* c = (char*)&i;
//         // dereferencing c will be 1 if the least significant byte is first. 0 if not.
//         if(! (((bool)*c) == params.little) ) throw std::runtime_error("ENDIANS DON'T MATCH AND I HAVEN'T FIXED THAT YET.");
        
//         // Check the magic numbers
//         uint16_t tiffid;
//         siff.read((char*)&tiffid, sizeof(uint16_t));

//         if(!((tiffid == BIGTIFFID) || (tiffid == TIFFID))) throw std::runtime_error("Could not verify that file is a true .tiff or .siff based on magic numbers.");
//         params.bigtiff = (tiffid == BIGTIFFID);

//         if (params.bigtiff) {
//             //  here the headers diverge a bit
//             uint16_t offset_size;
//             siff.read((char*)&offset_size, sizeof(uint16_t));
//             params.bytesPerPointer = offset_size; // sure to be 8 byte.
//             params.bytesPerNumTags = 8;
//             siff.read((char*)&offset_size, sizeof(uint16_t)); // these are always 0.
//             if(offset_size) throw std::runtime_error("File is not a valid BIGTIFF or .SIFF.");
            
//             uint64_t firstIFD;
//             siff.read((char*)&firstIFD,params.bytesPerPointer);
//             params.firstIFDAddress = firstIFD;
//             params.bytesPerTag = 20;
//         }
//         else {
//             // regular ol' tiff
//             uint32_t firstIFD;
//             siff.read((char*)&firstIFD, sizeof(uint32_t));
//             params.firstIFDAddress = (uint64_t) firstIFD;
//             params.bytesPerTag = 12;
//             params.bytesPerNumTags = 2;
//         }
    
//         // Now do the ScanImage-specific checks!
//         uint32_t magic;
//         siff.read((char*)&magic, sizeof(uint32_t));
        
//         uint32_t si;
//         siff.read((char*)&si, sizeof(uint32_t));
        
//         if( !( (magic == MAGICNUMBER) && (si == SI2019) ) ) throw std::runtime_error("File is a .tiff, but was not produced by ScanImage");

//         // ScanImage data stuff
//         uint32_t NVFD;
//         siff.read((char*)&NVFD, sizeof(uint32_t));
//         params.NVFD_length = NVFD; // in bytes

//         uint32_t ROI;
//         siff.read((char*)&ROI, sizeof(uint32_t));
//         params.ROI_string_length = ROI; // in bytes

        
//         char headerstring[params.NVFD_length];
//         siff.read(headerstring, params.NVFD_length);
//         params.headerstring = std::string(headerstring);

//         char roistring[params.ROI_string_length];
//         siff.read(roistring, params.ROI_string_length);
//         params.ROI_string = std::string(roistring);
              
//         return params;
//     }
//     catch(std::exception& e){
//         if (siff.is_open()) siff.close();
//         throw std::runtime_error(std::string("Could not parse parameters from file:\n") + e.what()); 
//     }

// }

// // Converts a .siff file to a .tiff file
// void siff_to_tiff(std::string sourcepath, std::string savepath) { 
    
//     std::ifstream siff(sourcepath, std::ios::in | std::ios::binary);
//     std::ofstream tiff(savepath, std::ios::out | std::ios::binary);
//     SiffParams params = params_from_siff(siff);

//     try{
//         siff_to_tiff_method(siff, tiff, params); // the meat of things.
//         siff.close();
//         tiff.close();
//     }

//     catch(std::exception &e) { 
//         siff.close();
//         tiff.close();
//         throw std::runtime_error(
//             (
//                 std::string("Error encountered in siff_to_tiff: ") +
//                 e.what()
//             ).c_str()
//         );
//     }

// }

// // Uses the sourcepath to identify the save path
// void siff_to_tiff(std::string sourcepath) {
//     std::string savepath = sourcepath.substr(0, sourcepath.find_last_of(".")) + ".tiff";
//     siff_to_tiff(sourcepath, savepath);
// }

