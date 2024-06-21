from typing import Tuple
import pytest

import corrosiffpy
import siffreadermodule

#filename = '/Users/stephen/Desktop/Data/imaging/2024-04/2024-04-17/21Dhh_GCaFLITS/Fly1/Flashes_1.siff'
filename = '/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/R60D05_TqCaFLITS/Fly1/EBAgain_1.siff'
#filename = '/Users/stephen/Desktop/Data/imaging/2024-05/2024-05-27/SS02255_greenCamui_alpha/Fly1/PB_1.siff'

@pytest.fixture(scope='session')
def siffreaders() -> Tuple:
    return (corrosiffpy.open_file(filename), siffreadermodule.SiffIO(filename))
