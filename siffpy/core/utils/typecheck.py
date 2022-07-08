# I know it's not Pythonic. Sue me. I want to type check stuff because I can be forgetful.
import logging

def x_across_time_TYPECHECK(
        num_slices, z_list,
        flim,
        num_colors, color_list
    ) -> tuple[list[int], bool, list[int]]:
    
    # make them iterables if they're not, make sure they're reasonable numbers
    if isinstance(z_list, int):
        if z_list > num_slices - 1:
            logging.warn("Provided a z plane number larger than number of slices. Producing list of all planes",
                stacklevel=2
            )
            z_list = list(range(num_slices))
        z_list = [z_list]
    if isinstance(z_list, list):
        for z_plane in z_list:
            if z_plane > num_slices:
                z_list.remove(z_plane)
        if not len(z_list):
            raise Exception("No valid z plane numbers provided!")
    
    if not isinstance(flim, bool):
        flim = False

    if isinstance(color_list, int):
        if color_list >= num_colors:
            logging.warn("Provided a color index greater than that available. Using all channels.",
                stacklevel=2
            )
            color_list = list(range(num_colors))
        else:
            color_list = [color_list]
    if isinstance(color_list, list):
        for color in color_list:
            if color >= num_colors:
                color_list.remove(color)
        if not len(color_list):
            raise Exception("No valid color channels provided!")

    # make them the full volume if they're None
    if z_list is None:
        z_list = list(range(num_slices))
    if color_list is None:
        color_list = list(range(num_colors))

    if len(z_list) > num_slices:
        logging.warn("Length of z_list is greater than the number of planes in a stack.\n" +
            "Defaulting to full volume (%s slices)" %num_slices,
            stacklevel=2
        )
        z_list = list(range(num_slices))

    return (z_list, flim, color_list)