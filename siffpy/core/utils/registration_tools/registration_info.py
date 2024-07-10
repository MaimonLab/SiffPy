from enum import Enum
from typing import Optional, Callable, Tuple, Dict, TYPE_CHECKING
from pathlib import Path
from abc import ABC, abstractmethod

from h5py import File as h5File
import numpy as np

#from siffreadermodule import SiffIO
#from corrosiffpy import SiffIO
from siffpy.core.utils import ImParams
from siffpy.core.utils.types import PathLike

if TYPE_CHECKING:
    from corrosiffpy import SiffIO

def populate_dict_across_colors(
    im_params : ImParams,
    alignment_color_channel : int,
    yx_shifts : Dict[int, Tuple[int,int]],
):
    """
    Populates the yx_shifts dict with the same values for all color channels
    except the alignment color channel, which are assumed to already be populated
    """
    for color_matlab_int in im_params.color_list:
        color_channel = color_matlab_int - 1
        offset = color_channel - alignment_color_channel
        for alignment_frame, alignment_val in list(yx_shifts.items()): # can't change during iter
            yx_shifts[alignment_frame + offset] = alignment_val
    

class RegistrationType(Enum):
    Caiman = 'caiman'
    Suite2p = 'suite2p'
    Siffpy = 'siffpy'
    Average = 'average'
    Other = 'other'

class RegistrationInfo(ABC):
    """ Base class for all Registration implementations """

    REGISTRATION_INFO_SUFFIX = ".h5"
    backend : RegistrationType = RegistrationType.Siffpy
    multithreading_compatible : bool = True # Whether this registration method can be run in parallel

    def __init__(
            self,
            siffio : 'SiffIO',
            im_params : ImParams,
        ):
        self.filename = siffio.filename if siffio is not None else None
        self.yx_shifts : Dict[int, Tuple[int,int]]= {}
        self.reference_frames : np.ndarray = None
        self.im_params = im_params
        self.registration_color_channel = None

    def __get_item__(self, idx : int):
        return self.yx_shifts[idx]
    
    @property
    def registration_type(self)->'RegistrationType':
        return self.__class__.backend

    @abstractmethod
    def register(
        self,
        siffio : 'SiffIO',
        *args,
        alignment_color_channel : int = 0,
        **kwargs
        ):
        raise NotImplementedError()

    def align_to_reference(
        self,
        images : np.ndarray,
        z_plane : int
        )->Tuple[int,int]:
        raise NotImplementedError()

    def save(self, save_path : Optional[PathLike] = None):
        if save_path is None:
            save_path = Path(self.filename).with_suffix("")
        save_path = Path(save_path)
        save_path = save_path / f"{Path(self.filename).stem}_registration_info{self.REGISTRATION_INFO_SUFFIX}"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5File(save_path, 'w') as f:
            f.attrs['filename'] = Path(self.filename).stem
            f.attrs['registration_type'] = self.registration_type.value
            f.attrs['registration_color'] = self.registration_color_channel
            
            # Store shifts
            yx_shifts = f.create_group(
                'yx_shifts',
            )

            yx_shifts.create_dataset(
                'frame_index',
                data = np.array(list(self.yx_shifts.keys())),
                dtype = np.int64,
            )

            yx_shifts.create_dataset(
                'shift_values',
                data = np.array(list(self.yx_shifts.values())),
                dtype = np.int64,
            )

            if hasattr(self, 'saved_attrs'):
                for attr_name in self.saved_attrs:
                    try:
                        f.attrs[attr_name] = getattr(self, attr_name)
                    except Exception as e:
                        print(f"Failed to save attribute {attr_name} with error: {e}")

            # Store reference frames
            f.create_dataset(
                "reference_frames",
                data = self.reference_frames,
                dtype = np.float32,   
            )
    
    def assign_siffio(self, siffio : 'SiffIO'):
        """
        Required to call if you load a `RegistrationInfo`
        from a file and want to use new frames.
        """
        self.siffio = siffio
        self.filename = siffio.filename

    def from_dict(self, dict : Dict):
        self.filename = dict['filename']
        self.registration_color_channel = dict['registration_color']
        self.yx_shifts = dict['yx_shifts']
        self.reference_frames = dict['reference_frames']

    @classmethod
    def load_as_dict(
        cls,
        path : PathLike,   
    )->dict:
        """
        Returns a dict that can be used to instantiate a `RegistrationInfo` subclass
        """
        path = Path(path)
        if not path.suffix == cls.REGISTRATION_INFO_SUFFIX:
            raise ValueError(
                f"File {path} does not have the correct extension for a RegistrationInfo file."
            )
        
        with h5File(path, 'r') as f:
            filename = f.attrs['filename']
            registration_type = f.attrs['registration_type']
            registration_type = RegistrationType(registration_type)
            registration_color_channel = f.attrs['registration_color']
            yx_shifts = f['yx_shifts']
            frame_index = yx_shifts['frame_index'][:]
            shift_values = yx_shifts['shift_values'][:]
            yx_shifts = {idx : tuple(shift) for idx, shift in zip(frame_index, shift_values.tolist())}
            reference_frames = f['reference_frames'][:]

        return {
            'filename' : filename,
            'registration_type' : registration_type,
            'registration_color' : registration_color_channel,
            'yx_shifts' : yx_shifts,
            'reference_frames' : reference_frames,
        }
        
    def __repr__(self):
        return f"{self.registration_type} RegistrationInfo for {self.filename}"
        
class CustomRegistrationInfo(RegistrationInfo):
    """
    A class to allow definition of custom registration functions.
    """

    def __init__(
            self,
            siffio : 'SiffIO',
            register_func : Callable,
            alignment_func : Callable,
            name : str = 'Unspecified'
        ):
        super().__init__(siffio, RegistrationType.Other)
        self.register_func = register_func
        self.alignment_func = alignment_func
        self.name = name

    def register(self, *args, **kwargs):
        return self.register_func(*args, **kwargs)

    def align_to_reference(
            self,
            image : np.ndarray,
            z_plane : int
        )->Tuple[int,int]:
        return self.alignment_func(image, z_plane)