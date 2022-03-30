""" 
Plots the "Maimon lab classic": 

Heading
Forward movement (wrapped)
Walking speed (mm/s)
"""

from typing import Type
import holoviews as hv
import numpy as np
import logging
from scipy.stats import circmean

from .tracplotter import *
from ..utils.fcns import *

class FullPlotter(TracPlotter):
    """
    Plotter class dedicated to plotting and representing behavioral data from FicTrac data broadly.

    Creates a plotter of each type but for ONLY ONE LOG, then its plot method compiles them all
    into a single consistent Layout.


    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

        


