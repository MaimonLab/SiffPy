import holoviews as hv
import bokeh
from holoviews import opts
hv.extension('bokeh')

from .siffplot import SiffPlot

from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats

def tileplot(ims, num_slices, **kwargs):
    """
    Accepts matplotlib subplots kwargs
    """
    nrows = int(np.ceil(np.sqrt(num_slices)))
    ncols = int(num_slices / nrows)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    
    for idx in range(num_slices):
        r = int(idx / ncols)
        c = int(idx%ncols)
        if (r>1):
            if (c>1):
                ax[r][c].imshow(ims[idx],alpha=alphas[idx],vmin=vmin,vmax=vmax,cmap=cmap)
                ax[r][c].set_facecolor("white")
            else:
                ax[r].imshow(ims[idx],alpha=alphas[idx],vmin=vmin,vmax=vmax,cmap=cmap)
                ax[r].set_facecolor("white")
        else:
            ax.imshow(ims[idx],alpha=alphas[idx],vmin=vmin,vmax=vmax, cmap=cmap)
    #return (fig, ax)

def chisq_to_alpha(chisq : np.ndarray, nbins : int = 620, **kwargs) -> np.ndarray:
    """
    Transform an array of chi-squared statistics into the corresponding
    chi-sq distribution p-value, with a transform determined by the keyword arguments.

    Defaults to just returning the p-value array

    Keyword arguments:
        transform (function)

        log (bool)

        power (float)
    """

    alpha = np.zeros(chisq.shape)
    pvals = 1.0-stats.chi2.cdf(chisq,nbins)
    if 'transform' in kwargs:
        alpha = kwargs[transform](pvals)
        alpha[np.isnan(alpha)] = 0
        return alpha
    elif 'log' in kwargs:
        if isinstance(kwargs['log'], bool) and kwargs['log']:
            alpha = 1.0 - np.log(pvals)/np.log(np.min(pvals.flatten()))
            alpha[np.isnan(alpha)] = 0
            return alpha
    elif 'power' in kwargs:
        if isinstance(kwargs['power'],float):
            alpha = np.power(1.0-pvals, power)
            alpha[np.isnan(alpha)] = 0
            return alpha
    else:
        return pvals