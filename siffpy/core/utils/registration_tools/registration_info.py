from enum import Enum
from typing import Union, Callable
from pathlib import Path
import pickle
from abc import ABC, abstractmethod

import numpy as np

from siffreadermodule import SiffIO
from siffpy.core.utils.im_params import ImParams

class RegistrationType(Enum):
    Caiman = 'caiman'
    Suite2p = 'suite2p'
    Siffpy = 'siffpy'
    Average = 'average'
    Other = 'other'

class RegistrationInfo(ABC):
    """ Base class for all Registration implementations """

    def __init__(
            self,
            siffio : SiffIO,
            im_params : ImParams,
            backend : Union[RegistrationType,str]
        ):
        if isinstance(backend, str):
            backend = RegistrationType(backend)
        self.registration_type = backend
        self.filename = siffio.filename
        self.yx_shifts : list[tuple[int,int]]= []
        self.reference_frames : np.ndarray = None
        self.im_params = im_params

    def __get_item__(self, idx):
        return self.yx_shifts[idx]
    
    @abstractmethod
    def register(
        self,
        siffio : SiffIO,
        *args,
        **kwargs
        ):
        raise NotImplementedError()

    @abstractmethod
    def align_to_reference(
        self,
        images : np.ndarray,
        z_plane : int
        )->tuple[int,int]:
        raise NotImplementedError()

    def save(self, save_path : Union[str, Path]):
        save_path = Path(save_path)
        save_path = save_path / f"{self.filename}.rdct"
        with save_path.open('wb') as f:
            pickle.dump(self, f)
    
    def assign_siffio(self, siffio : SiffIO):
        """
        Required to call if you load a `RegistrationInfo`
        from a file and want to use new frames.
        """
        self.siffio = siffio

    @classmethod
    def load(cls, path : Union[str, Path]):
        if not path.suffix == '.rdct':
            raise ValueError(
                f"File {path} does not have the correct extension for a RegistrationInfo file."
            )
        with path.open('rb') as f:
            return pickle.load(f)
        
    def __repr__(self):
        return f"{self.registration_type} RegistrationInfo for {self.filename}"
        
class CustomRegistrationInfo(RegistrationInfo):
    """
    A class to allow definition of custom registration functions.
    """

    def __init__(
            self,
            siffio : SiffIO,
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
        )->tuple[int,int]:
        return self.alignment_func(image, z_plane)