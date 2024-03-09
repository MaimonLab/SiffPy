from typing import Union, TYPE_CHECKING, List, Tuple, Any
import numpy as np

if TYPE_CHECKING:
    from siffpy.siffmath.flim import FlimTrace
    from siffpy.siffmath.fluorescence import FluorescenceTrace
    from siffpy.siffmath.phase.traces import PhaseTrace

ImageArray = np.ndarray[Any, np.dtype[np.int_]]

FluorescenceVectorLike = Union[
    List['FluorescenceTrace'],
    Tuple['FluorescenceTrace'],
]
FlimVectorLike = Union[
    List['FlimTrace'],
    Tuple['FlimTrace'],
]

FluorescenceArrayLike = Union[np.ndarray, 'FluorescenceTrace', FluorescenceVectorLike]
FlimArrayLike = Union[np.ndarray, 'FlimTrace', FlimVectorLike]
PhaseTraceLike = Union[np.ndarray, 'PhaseTrace']
