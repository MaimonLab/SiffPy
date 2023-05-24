from typing import Union, TYPE_CHECKING, List, Tuple
import numpy as np

if TYPE_CHECKING:
    from siffpy.siffmath.flim import FlimTrace
    from siffpy.siffmath.fluorescence import FluorescenceTrace
    from siffpy.core.flim.flimunits import FlimUnits
    from siffpy.siffmath.phase.traces import PhaseTrace

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
