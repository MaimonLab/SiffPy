from typing import Any, Tuple, List, Dict, Optional

import numpy as np

def open_file(filename : str)->'SiffIO':...

class SiffIO():
    """
    This class is a wrapper for the Rust `corrosiff` library.

    Controls file reading and formats data streams into
    `numpy` arrays, `Dict`s, etc.
    """
    @property
    def filename(self)->str:
        """The name of the file being read"""
        ...

    def get_file_header(self)->Dict:
        ...


    def get_num_frames(self)->int:
        """ Number of frames (including flyback) """
        ...

    def get_frame_metadata(self, frames : Optional[List[int]] = [])->List[Dict]:
        """
        Retrieves metadata for the requested frames as
        a list of dictionaries. If no frames are requested,
        retrieves metadata for all frames. This is probably
        the slowest method of retrieving frame-specific
        data, because the list of dictionaries means that
        it's constrained by the GIL to parse one frame
        at a time, rather than multithreading. Probably
        can be bypassed with better code -- I almost never
        use this method so it's not a priority for me!

        ## Arguments

        * `frames` : List[int] (optional)
            A list of frames for which to retrieve metadata.
            If `None`, retrieves metadata for all frames.

        ## Returns

        * `List[Dict]`
            A list of dictionaries containing metadata for
            each frame.
        """
        ...


    def get_frames(
        self,
        frames : Optional[List[int]] = None,
        registration : Optional[Dict] = {},
    )->'np.ndarray[Any, np.dtype[np.uint16]]':
        """
        Retrieves frames from the file without
        respect to dimension or flyback.

        ## Arguments

        * `frames` : List[int]
            A list of frames to retrieve. If `None`, all frames
            will be retrieved.

        * `registration` : Dict            
            A dictionary containing registration information
            (the keys correspond to the frame number, the values
            are tuples of (y,x) offsets).

        ## Returns

        * `np.ndarray[Any, np.dtype[np.uint16]]`
            A numpy array containing the frames in the
            order they were requested.

        ## Example
            
            ```python
            import numpy as np
            import corrosiff_python

            # Load the file
            filename = '/path/to/file.siff'
            siffio = corrosiff_python.open_file(filename)

            # Get the data as an array
            frame_data = siffio.get_frames(list(range(1000)))
            print(frame_data.shape, frame_data.dtype)

            >>> ((1000, 512, 512), np.uint16)
            ```

        """
        ...

    # def pool_frames(
    #     self,
    #     frames : List[int],
    #     flim : bool = False,
    #     registration : Optional[Dict] = None,
    # )->'np.ndarray[Any, np.dtype[np.uint16]]':
    #     """ NOT IMPLEMENTED """

    # def flim_map(
    #     self,
    #     params : 'FLIMParams',
    #     frames : List[int],
    #     confidence_metric : str = 'chi_sq',
    #     registration : Optional[Dict] = None,
    # )->Tuple['np.ndarray[Any, np.dtype[np.float64]]', 'np.ndarray[Any, np.dtype[np.uint16]]', 'np.ndarray[Any, np.dtype[np.float64]]']:
    #     """
    #     Returns a tuple of (flim_map, intensity_map, confidence_map)
    #     where flim_map is the empirical lifetime with the offset of
    #     params subtracted.
    #     """
    #     ...

    # def sum_roi(
    #     self,
    #     mask : 'np.ndarray[Any, np.dtype[np.bool_]]',
    #     *,
    #     frames : Optional[List[int]] = None,
    #     registration : Optional[Dict] = None,
    # )->'np.ndarray[Any, np.dtype[np.uint16]]':
    #     """
    #     Mask may have more than 2 dimensions, but
    #     if so then be aware that the frames will be
    #     iterated through sequentially, rather than
    #     aware of the correspondence between frame
    #     number and mask dimension. Returns a 1D
    #     array of the same length as the frames
    #     provided, regardless of mask shape.
    #     """
    #     ...

    # def sum_rois(
    #     self,
    #     masks : 'np.ndarray[Any, np.dtype[np.bool_]]',
    #     *,
    #     frames : Optional[List[int]] = None,
    #     registration : Optional[Dict] = None,
    # )->'np.ndarray[Any, np.dtype[np.uint16]]':
    #     """
    #     Masks may have more than 2 dimensions, but
    #     if so then be aware that the frames will be
    #     iterated through sequentially, rather than
    #     aware of the correspondence between frame
    #     number and mask dimension. Returns a 2D
    #     array of dimensions `(len(masks), len(frames))`.
    #     """
    #     ...

    # def sum_roi_flim(
    #     self,
    #     mask : 'np.ndarray[Any, np.dtype[np.bool_]]',
    #     params : 'FLIMParams',
    #     *,
    #     frames : Optional[List[int]] = None,
    #     registration : Optional[Dict] = None,
    # )->'np.ndarray[Any, np.dtype[np.float64]]':
    #     """
    #     Mask may have more than 2 dimensions, but
    #     if so then be aware that the frames will be
    #     iterated through sequentially, rather than
    #     aware of the correspondence between frame
    #     number and mask dimension. Returns a 1D
    #     arrary of the same length as the frames
    #     provided, regardless of mask shape.
    #     """
    #     ...

    # def sum_rois_flim(
    #     self,
    #     masks : Union['np.ndarray[Any, np.dtype[np.bool_]]', List['np.ndarray[Any, np.dtype[np.bool_]]']],
    #     params : 'FLIMParams',
    #     *,
    #     frames : Optional[List[int]] = None,
    #     registration : Optional[Dict] = None,
    # ) -> 'np.ndarray[Any, np.dtype[np.float64]]':
    #     """
    #     If `masks` is an array, the slowest dimension
    #     is assumed to be the mask dimension.

    #     Masks may have more than 2 dimensions, but
    #     if so then be aware that the frames will be
    #     iterated through sequentially, rather than
    #     aware of the correspondence between frame
    #     number and mask dimension. Returns a 2D
    #     array of dimensions `(len(masks), len(frames))`.
    #     """
    #     ...

    def get_histogram(
        self,
        frames : Optional[List[int]] = None,
    )-> 'np.ndarray[Any, np.dtype[np.uint64]]':
        """
        Retrieves the arrival time histogram from the
        file. Width of the histogram corresponds to the
        number of BINS in the histogram. For the time units
        of the histogram, use the metadata.

        ## Arguments

        * `frames` : List[int]
            A list of frames to retrieve. If `None`, all frames
            will be retrieved.

        ## Returns

        * `np.ndarray[Any, np.dtype[np.uint64]]`
            A numpy array containing the histogram of dimensions
            (`frames.len()`, `num_bins`)

        ## Example

            ```python
            import numpy as np
            import corrosiff_python

            # Load the file
            filename = '/path/to/file.siff'
            siffio = corrosiff_python.open_file(filename)

            hist = siffio.get_histogram(frames = list(range(1000)))
            print(hist.shape, hist.dtype)

            # 629 time bins with a 20 picosecond resolution
            # = 12.58 nanoseconds, ~ 80 MHz
            >>> ((1000, 629), np.uint64)

            ```
        """
        ...

    def get_experiment_timestamps(
        self,
        frames : Optional[List[int]] = None,
    )->'np.ndarray[Any, np.dtype[np.float64]]':
        """
        Returns an array of timestamps of each frame based on
        counting laser pulses since the start of the experiment.

        Units are seconds.

        Extremely low jitter, small amounts of drift (maybe 50 milliseconds an hour).

        ## Arguments

        * `frames` : List[int]
            A list of frames to retrieve. If `None`, all frames
            will be retrieved.

        ## Returns

        * `np.ndarray[Any, np.dtype[np.float64]]`
            Seconds since the beginning of the microscope acquisition.

        ## Example

            ```python
            import numpy as np
            import corrosiff_python

            # Load the file
            filename = '/path/to/file.siff'
            siffio = corrosiff_python.open_file(filename)

            time_exp = siffio.get_experiment_timestamps(frames = list(range(1000)))
            print(time_exp.shape, time_exp.dtype)

            >>> ((1000,), np.float64)
            ```
        
        ## See also
        - `get_epoch_timestamps_laser`
        - `get_epoch_timestamps_system`
        - `get_epoch_both`
        """
        ...

    def get_epoch_timestamps_laser(
            self,
            frames : Optional[List[int]] = None,
        )->'np.ndarray[Any, np.dtype[np.uint64]]':
        """
        Returns an array of timestamps of each frame based on
        counting laser pulses since the start of the experiment.
        Extremely low jitter, small amounts of drift (maybe 50 milliseconds an hour).

        Can be corrected using get_epoch_timestamps_system.

        ## Arguments

        * `frames` : List[int]
            A list of frames to retrieve. If `None`, all frames
            will be retrieved.

        ## Returns

        * `np.ndarray[Any, np.dtype[np.uint64]]`
            Nanoseconds since epoch, counted by tallying
            laser sync pulses and using an estimated sync rate.

        ## Example

            ```python
            import numpy as np
            import corrosiff_python

            # Load the file
            filename = '/path/to/file.siff'
            siffio = corrosiff_python.open_file(filename)

            time_laser = siffio.get_epoch_timestamps_laser(frames = list(range(1000)))
            print(time_laser.shape, time_laser.dtype)

            >>> ((1000,), np.uint64)

        ## See also
        - `get_epoch_timestamps_system`
        - `get_epoch_both`
        """
        ...

    def get_epoch_timestamps_system(
            self,
            frames : Optional[List[int]] = None,
        )->'np.ndarray[Any, np.dtype[np.uint64]]':
        """
        Returns an array of timestamps of each frame based on
        the system clock. High jitter, but no drift.

        WARNING if system timestamps do not exist, the function
        will throw an error. Unlike `siffreadermodule`/the `C++`
        module, this function will not crash the Python interpreter.

        ## Arguments

        * `frames` : List[int]
            A list of frames to retrieve. If `None`, all frames
            will be retrieved.

        ## Returns

        * `np.ndarray[Any, np.dtype[np.uint64]]`
            Nanoseconds since epoch, measured using
            the system clock of the acquiring computer.
            Only called about ~1 time per second, so
            this will be the same number for most successive
            frames.

        ## Example

            ```python
            import numpy as np
            import corrosiff_python

            # Load the file
            filename = '/path/to/file.siff'
            siffio = corrosiff_python.open_file(filename)

            time_system = siffio.get_epoch_timestamps_system(frames = list(range(1000)))
            print(time_system.shape, time_system.dtype)

            >>> ((1000,), np.uint64)
            ```

        ## See also

        - `get_epoch_timestamps_laser`
        - `get_epoch_both`
        """
        ...

    def get_epoch_both(
            self, 
            frames : Optional[List[int]] = None,
        )->'np.ndarray[Any, np.dtype[np.uint64]]':
        """
        Returns an array containing both the laser timestamps
        and the system timestamps.

        The first row is laser timestamps, the second
        row is system timestamps. These can be used to correct one
        another

        WARNING if system timestamps do not exist, the function
        will throw an error. Unlike `siffreadermodule`/the `C++`
        module, this function will not crash the Python interpreter!

        ## Arguments

        * `frames` : List[int]
            A list of frames to retrieve. If `None`, all frames
            will be retrieved.

        ## Returns

        * `np.ndarray[Any, np.dtype[np.uint64]]`
            Nanoseconds since epoch, measured using
            the laser pulses in the first row and the system
            clock calls in the second row.

        ## Example

            ```python
            import numpy as np
            import corrosiff_python

            # Load the file
            filename = '/path/to/file.siff'
            siffio = corrosiff_python.open_file(filename)

            time_both = siffio.get_epoch_both(frames = list(range(1000)))
            print(time_both.shape, time_both.dtype)

            >>> ((2, 1000), np.uint64)
            ```

        ## See also

        - `get_epoch_timestamps_laser`
        - `get_epoch_timestamps_system`
        """
        ...

    def get_appended_text(
            self,
        )->List[Tuple[int, str, Optional[float]]]:
        """
        Returns a list of strings containing the text appended
        to each frame. Only returns elements where there was appended text.
        If no frames are provided, searches all frames.

        Returns a list of tuples of (frame number, text, timestamp).

        For some versions of `ScanImageFLIM`, there is no timestamp entry,
        so the tuple for those files will be (frame number, text, None).

        ## Arguments

        * `frames` : List[int]
            A list of frames to retrieve. If `None`, all frames
            will be retrieved.

        ## Returns

        * `List[Tuple[int, str, Optional[float]]]`
            A list of tuples containing the frame number, the text
            appended to the frame, and the timestamp of the text
            (if it exists).

        ## Example

            ```python
            import numpy as np
            import corrosiff_python

            # Load the file
            filename = '/path/to/file.siff'
            siffio = corrosiff_python.open_file(filename)

            text = siffio.get_appended_text(frames = list(range(1000)))
            ```
        """
        ...
    
def siff_to_tiff(
        sourcepath : str,
        /,
        savepath : Optional[str] = None,
        mode : Optional[str] = None,
    )->None:
    """
    Converts a .siff file to a .tiff file containing only intensity information.

    TODO: Contain OME-TIFF metadata for more convenient ImageJ/Fiji viewing of
    output.

    Arguments
    --------

    sourcepath : str

        Path to a .siff file (will also work for a .tiff file, though I don't know
        why you'd try that)

    savepath : str

        Path to where the .tiff should be saved. If None, will be saved
        in same directory as the .siff file.

    mode : str

        Either 'scanimage' or 'ome'. If 'scanimage', will save the tiff
        in the same format as ScanImage, with the same metadata. For
        more info about the ScanImage tiff specification, see
        https://docs.scanimage.org/Appendix/ScanImage+BigTiff+Specification.html
        
        If 'ome',
        will save the tiff as an OME-TIFF, which is more convenient for
        viewing in Fiji/ImageJ, but contaminates the metadata of the first frame.
    """
    ...