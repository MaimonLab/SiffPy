from typing import Iterable, Union

import holoviews as hv
import numpy as np

from ...integratedplotter import IntegratedPlotter
from ..siff_plotters import PhasePlotter
from ..trac_plotters import HeadingPlotter
from ...utils.enums import CorrStyle

from ....core import SiffReader
from ....sifftrac import FictracLog
from ....siffmath.fluorescence.traces import FluorescenceVector
from ....siffmath.phase import phase_analyses


class HeadingPhasePlotter(IntegratedPlotter):

    DEFAULT_SIFFPLOTTER_OPTS = {

    }

    DEFAULT_TRACPLOTTER_OPTS = {

    }

    def __init__(self, siffreader : SiffReader, fictraclog : FictracLog, fluorescence_vector : FluorescenceVector = None):
        
        # Type hinting for linters
        self.siffplotter : PhasePlotter
        self.tracplotter : HeadingPlotter

        pp = PhasePlotter(siffreader, fluorescence_vector = fluorescence_vector)
        hp = HeadingPlotter(fictraclog)

        super().__init__(pp, hp)

    def aligned_phase_plot(self, fluorescence_vector : FluorescenceVector = None, phase_method : str = None)->hv.Overlay:
        """ Returns the overlaid phase of the fluorescence data and direction of the heading data """
        if fluorescence_vector is None:
            if self.siffplotter.data is None:
                raise ValueError("If no fluorescence data was provided on initialization, it must be provided in call to phase_plot")
            fluorescence_vector = self.siffplotter.data

        phase = self.siffplotter.estimate_phase(
            vector_timeseries = fluorescence_vector,
            phase_method = phase_method
        )

        phase_plot = self.siffplotter.plot_phase(
                self.siffplotter.siffreader.t_axis(), 
                phase
            ).opts({'Path':{'line_color':"#698DB7"}})
    
        offset = phase_analyses.fit_offset(
            self.tracplotter.logs[0][0]._dataframe['integrated_heading_lab'].values,
            phase,
            fictrac_time = self.tracplotter.logs[0][0]._dataframe['image_time'].values,
            phase_time = self.siffplotter.siffreader.t_axis()
        )
        
        heading_plot = self.tracplotter.plot(offset=-offset)

        merged_plot = (
            phase_plot *
            heading_plot
        ).opts(
            {
                'Path' : {
                    'yaxis' : True,
                    'yticks' : [
                        ((-offset)%(2*np.pi), 'Rear'),
                        ((np.pi - offset)%(2*np.pi), 'Front'),
                    ],
                    'ylabel' : 'Bar\nposition',
                }
            }
        )

        return merged_plot

    def corr_plot(self,
            windows : Iterable[float],
            style : Union[CorrStyle,str] = CorrStyle.HEATMAP,
        ) -> hv.Element:
        """ 
        Returns a plot of the correlations between heading and the phase of the sampled data
        over the time windows provided.

        Arguments
        --------

        windows : Iterable[float]

            Timescales over which to compute the sliding circular correlation. Units are SECONDS.

        style : CorrStyle or str

            What type of correlation to use. If a string is provided, must be able to be cast into
            a `siffpy.siffplot.utils.enums.CorrStyle` Enum

        Returns
        -------

        plot : hv.Element

            A plot of the correlations

        """

        if isinstance(style, str):
            style = CorrStyle(style)
        
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def visualize(self, *args, **kwargs) -> Union[hv.Layout, hv.Element]:
        raise NotImplementedError()