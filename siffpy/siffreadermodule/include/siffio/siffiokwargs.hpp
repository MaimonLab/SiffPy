#ifndef SIFFIO_KWARGS_HPP
#define SIFFIO_KWARGS_HPP
#include <stddef.h>

static char* GET_FRAMES_KWARGS[] = {"frames", "type", "flim", "registration", "discard_bins", NULL};

static char* GET_FRAMES_METADATA_KEYWORDS[] = {"frames", NULL};

static char* POOL_FRAMES_KEYWORDS[] = {"pool_lists", "type", "flim", "registration", "discard_bins", NULL};

static char* FLIM_MAP_KEYWORDS[] = {"params","frames", "confidence_metric", "registration","sizeSafe", "discard_bins", NULL};

static char* SUM_ROIS_KEYWORDS[] = {"mask", "frames", "registration", NULL};

static char* SUM_ROI_FLIM_KEYWORDS[] = {"mask", "params", "frames", "registration", NULL};

static char* GET_HISTOGRAM_KEYWORDS[] = {"frames", NULL};

#endif