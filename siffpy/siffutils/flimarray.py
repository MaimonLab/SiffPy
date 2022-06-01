import logging
from multiprocessing.sharedctypes import Value

import numpy as np

from . import FLIMParams

class FlimArray():
    """
    Class written to behave like a numpy array, but stores TWO arrays: the
    empirical lifetime (as a float) and the photon count (as a uint16).

    This class is designed to facilitate arithmetic operations involving
    FLIM data, where the lifetimes interact nonlinear when you pool frames
    or datasets.
    """

    def __init__(self, intensity : np.ndarray, lifetime : np.ndarray, confidence : np.ndarray = None, FLIMParams : list[FLIMParams] = None):
        """
        Initialize a FlimArray using two numpy arrays: the intensity (in photon counts) and the empirical
        lifetime (an array of floats). An optional keyword argument is a third array, the confidence metric,
        for example a pixelwise chi-square statistic. At present I haven't implemented operations that
        take the confidence metric into account. Another optional argument is a list of FLIMParam objects,
        one for each color channel.
        """
        if not ( isinstance(intensity, np.ndarray) and isinstance(lifetime,np.ndarray) ):
            raise ValueError("Must provide two numpy arrays to construct a FlimArray")
        if not (intensity.shape == lifetime.shape):
            raise ValueError(f"Shapes of intensity array ({intensity.shape}) and lifetime array ({lifetime.shape}) are not compatible.")
        if (not (confidence is None)) and not(confidence.shape == lifetime.shape):
            raise ValueError(f"Shape of confidence array is not compatible with the image array dimensions")
        
        self.intensity = intensity
        self.lifetime = lifetime
        if not confidence is None:
            self.confidence = confidence
        if not FLIMParams is None:
            self.FLIMParams = FLIMParams

#    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
#        if out is None:
#            pass
#            #out = FlimArray()
#        return out
#
#    def reshape(newshape, order='C'):
#        if not order == 'C':
#            raise ValueError("Non C style ordering not implemented.")
#        raise NotImplementedError()
    
    def __add__(self, other : 'FlimArray') -> 'FlimArray':
        """
        Combines two FlimArray objects together additively
        """
        if not isinstance(other, FlimArray):
            return NotImplemented
        
        new_intensity = self.intensity + other.intensity
        new_lifetime = np.nansum([self.lintensity, other.lintensity],axis=0)
        np.divide(new_lifetime, new_intensity, out=new_lifetime)

        kwargs = {}
        if hasattr(self, 'confidence') and hasattr(other, 'confidence'):
            raise NotImplementedError("Confidence addition has not yet been implemented.")

        if hasattr(self, 'FLIMParams') and hasattr(other, 'FLIMParams'):
            if not (self.FLIMParams == other.FLIMParams):
                raise NotImplementedError("")
        if hasattr(self, 'FLIMParams'):
            kwargs['FLIMParams'] = self.FLIMParams
        else:
            if hasattr(other,'FLIMParams'):
                kwargs['FLIMParams'] = other.FLIMParams
        return FlimArray(new_intensity, new_lifetime, **kwargs)

    def __iadd__(self, other : 'FlimArray'):
        if not isinstance(other, FlimArray):
            return NotImplemented
        np.nansum([self.lintensity, other.lintensity],axis=0,out=self.lifetime)
        self.lifetime /= self.intensity + other.intensity
        self.intensity += other.intensity
        if hasattr(self,'confidence') and hasattr(other,'confidence'):
            raise NotImplementedError("Confidence addition has not yet been implemented")
        return self
        
    def __mul__(self, other) -> 'FlimArray':
        kwargs = {}
        if hasattr(self,'confidence'):
            kwargs['confidence'] = self.confidence
        if hasattr(self, 'FLIMParams'):
            kwargs['FLIMParams'] = self.FLIMParams
        
        if isinstance(other, float):
            return FlimArray(self.intensity*other, self.lifetime, **kwargs)
        if isinstance(other, np.ndarray):
            return FlimArray(self.intensity*other, self.lifetime, **kwargs)
        if isinstance(other, FlimArray):
            raise ValueError("FlimArrays cannot be multiplied by other FlimArrays -- lifetime operation undefined.")
            #return FlimArray(self.intensity*other.intensity, self.lifetime, **kwargs)
        
        return NotImplemented

    def __imul__(self, other):
        if isinstance(other, float):
            self.intensity *= other
            return self
        if isinstance(other, np.ndarray):
            self.intensity *= other
            return self
        return NotImplemented

    def __truediv__(self, other):
        kwargs = {}
        if hasattr(self, 'confidence') and hasattr(other, 'confidence'):
            raise NotImplementedError("Confidence addition has not yet been implemented.")

        if hasattr(self, 'FLIMParams') and hasattr(other, 'FLIMParams'):
            if not (self.FLIMParams == other.FLIMParams):
                raise NotImplementedError("")
        if hasattr(self, 'FLIMParams'):
            kwargs['FLIMParams'] = self.FLIMParams
        else:
            if hasattr(other,'FLIMParams'):
                kwargs['FLIMParams'] = other.FLIMParams
        if isinstance(other, float):
            return FlimArray(self.intensity/other, self.lifetime, **kwargs)
        if isinstance(other, np.ndarray):
            return FlimArray(self.intensity/other, self.lifetime, **kwargs)
        return NotImplemented
        
    def __itruediv__(self, other):
        if isinstance(other, float):
            self.intensity /= other
            return self
        if isinstance(other, np.ndarray):
            self.intensity = other
            return self
        raise NotImplementedError()
            
    def __getattr__(self, key : str):
        if key == 'T':
            return self.transpose()
        if key == 'shape':
            return self.intensity.shape
        if key == 'lintensity':
            return self.intensity * self.lifetime
        if key == 'tau':
            return np.nansum(self.lintensity/np.sum(self.intensity))
        return super().__getattr__(key)

    def __repr__(self)->str:
        return "Intensity: "+ (self.intensity.__repr__()) + "\nLifetime: " + (self.lifetime.__repr__())

    # Numpylike functions
    def transpose(self)->'FlimArray':
        """
        Transposes the internal arrays and returns a new
        FlimArray object
        """

        kwargs = {}
        if hasattr(self, 'confidence'):
            kwargs['confidence'] = self.confidence.T
        if hasattr(self, 'FLIMParams'):
            kwargs['FLIMParaks'] = self.FLIMParams
        
        return FlimArray(self.intensity.T, self.lifetime.T, **kwargs)

