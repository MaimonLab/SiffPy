#ifndef BITMASKS_HPP
#define BITMASKS_HPP

#include <cstdint>
#include <stdint.h>

// Extract the y value of a photon read (uncompressed)
constexpr uint64_t YMASK = (((uint64_t) 1 << 63) - ((uint64_t) 1 << 48) + ((uint64_t) 1 << 63)); // that last bit'll getcha
// Extract the x value of a photon read (uncompressed)
constexpr uint64_t XMASK = (((uint64_t) 1 << 48) - ((uint64_t) 1 << 32));
// Extract the arrival time of a photon read (uncompressed)
constexpr uint64_t TAUMASK = (((uint64_t) 1<<32) - 1);

// Get the y value of a siff pixel read (uncompressed)
#define U64TOY(photon) (uint64_t) ((photon & YMASK) >> 48)
// Get the x value of a siff pixel read (uncompressed)
#define U64TOX(photon) (uint64_t) ((photon & XMASK) >> 32)
// Get the arrival time of a siff pixel read (uncompressed)
#define U64TOTAU(photon) (uint64_t) (photon & TAUMASK)

#define READ_TO_PX_NOSHIFT(photon, dim_y, dim_x) \
    (((U64TOY(photon)) * dim_x) + (U64TOX(photon)))

// Converts an uncompressed read to a pixel index (y*x_dim + x)
#define READ_TO_PX(photon, y_shift, x_shift, dim_y, dim_x) \
    ((((U64TOY(photon) + y_shift) % dim_y) * dim_x) \
    + (U64TOX(photon) + x_shift) % dim_x)

// Shifts the pixel location px by shift_y, shift_x in an
// image of dimensions dim_y, dim_x
#define PIXEL_SHIFT(px, y_shift, x_shift, dim_y, dim_x) \
    ( \
        ( \
            (((uint64_t)((px) / (dim_x)) + (y_shift)) % (dim_y)) * (dim_x)) \
    + (((px) % (dim_x) + (x_shift)) % (dim_x)) \
    )

#endif