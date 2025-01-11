import pytest
import numpy as np

from siffpy.siffmath.phase.phase_estimates import INTERP_FUNC

@pytest.fixture(scope = 'function')
def test_interp_func():
    """ Tests the relative magnitude lookup function exists """
    INTERP_FUNC(np.arange(0, 2*np.pi, 0.01))