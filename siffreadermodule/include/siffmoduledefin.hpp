#ifndef SIFFMODULEDEFIN_HPP
#define SIFFMODULEDEFIN_HPP


// KEYWORD ARGS BY FUNCTION
// NOTE: REGULAR ARGS HAVE TO BE HERE TOO!
#define SIFF_TO_TIFF_KEYWORDS (const char*[]){"sourcepath", "savepath", NULL}


// DOCSTRING DEFS

#define MODULE_DOC \
    "siffreader C extension module\n"\
    "Reads and interprets .siff and ScanImage .tiffs\n"\
    "Can be used in one of two ways: either directly calling"\
    "functions from siffreader (siffreader.get_frames(*args, **kwargs)"\
    "or instantiating a siffreader.SiffIO object (preferred).\n"\
    "\n"\
    "CLASSES:\n"\
    "SiffIO:\n"\
    "siffreadermodule.SiffIO(filename : str = None)\n"\
    "FUNCTIONS:\n"\
    "suppress_warnings():\n"\
        "\tSuppresses module-specific warnings.\n"\
    "report_warnings():\n"\
        "\tAllows module-specific warnings.\n"\
    "debug(debug : bool):\n"\
        "\tEnable or disable siffreadermodule debugging log."\
    "sifftotiff(sourcepath : str, savepath : str = None):\n"\
        "\tConverts a .siff file (in sourcepath) to a .tiff file (saved in savepath), discarding arrival time information, if relevant."

#define SUPPRESS_WARNINGS_DOCSTRING \
    "suppress_warnings()->None\n"\
    "Suppresses output warnings for siffreader functions."

#define REPORT_WARNINGS_DOCSTRING \
    "report_warnings()->None\n"\
    "Forces reporting of warnings for siffreader functions."

#define DEBUG_DOCSTRING \
    "debug(debug : bool) -> None\n"\
    "Creates a debug log if debug is True. If False, stops debugging.\n"

#define SIFF_TO_TIFF_DOCSTRING \
    "siff_to_tiff(sourcepath : str, savepath : str = None) -> None\n"\
    "Converts a siff file located at sourcepath to a tiff file and saves it in location savepath"


#endif