from .exp_math import *
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

class FLIMParams(object):
    """
    A class for storing, fitting,
    and interacting with parameters
    fitting FLIM arrival time data
    to an exponential.

    TODO: OFFER MORE OPTIONS FOR FITTING
    E.G. WITH MYSTIC, WITH DIFFERENT
    CONSTRAINED NONLINEAR FITTING APPROACHES,
    ETC.

    Attributes
    ----------

    Ncomponents (int) :
        Number of individual exponentials fit

    Exp_params (list) : 
        List of size Ncomponents corresponding
        to the fit parameters of each in format DICT:
            KEY -- VALUE
            FRAC -- (double) fraction in this state
            TAU -- (double) time constant of this state (in units of histogram bins)

    IRF (dict)  :
        'DIST' -- (string) name of distribution used for IRF.
            Possible types:
                Gaussian
                TODO: add more!
        'PARAMS' -- (dict) dictionary corresponding to the parameters of the IRF

    T_O (double) :
        The temporal offset, in histogram bin units, between the laser pulse and
        the arrival of photons.

    Methods
    -------

    fit_data(self, data, num_components, x0, metric):
        Takes in the ndarray data and fits the parameters of the model defined by this
        FLIMParams object. metric reflects the objective function.

    param_tuple(self):
        Returns a tuple listing the parameter values in a standardized order.

    param_dict(self):
        Returrns a dict listing parameter values in an easy-to-understand key value logic

    SCT March 28 2021, rainy day in Maywood NJ
    """

    def __init__(self, param_dict = None):
        if param_dict is None:
            self.Ncomponents = None
            self.Exp_params = []
            self.IRF = {}
            self.T_O = 0
        else:
            self.Ncomponents = param_dict['NCOMPONENTS']
            self.Exp_params = param_dict['EXPPARAMS']
            self.IRF = param_dict['IRF']
            self.T_O = param_dict['T_O']

    @classmethod
    def from_tuple(cls, param_tuple):
        """ Instantiate a FLIMParams from the parameter tuple """
        num_components = len(param_tuple) - 2
        exp_params = [
            {'FRAC' : param_tuple[comp*num_components], 
            'TAU' : param_tuple[comp*num_components + 1]} 
            for comp in range(num_components)
        ]

        params_dict = {
            'NCOMPONENTS' : num_components,
            'EXPPARAMS' : exp_params,
            'CHISQ' : 0.0,
            'T_O' : param_tuple[-2],
            'IRF' : param_tuple[-1]
        }
        return cls(param_dict=params_dict)


    def __str__(self):
        return self.param_dict().__str__()

    def chi_sq(self, data, params=None):
        """
        Computes the chi-squared statistic of the
        input data "data" using the fit parameters
        in this class.
        TODO: USE THE WRAPAROUND
        """
        if params is None:
            params = self.param_tuple()

        arrival_p = np.zeros(data.shape) # incrementally updated by each component
        for component in range(self.Ncomponents):
            arrival_p += params[2*component] * \
                monoexponential_prob(
                    np.arange(arrival_p.shape[0])-params[-2], # arrival time shifted by t_o
                    params[2*component + 1], # this component's tau value
                    params[-1] # the IRF width
                )
        
        total_photons = np.sum(data)
        chi_sq = ((data - total_photons*arrival_p)**2)/arrival_p

        chi_sq[np.isinf(chi_sq)]=0 # ignore the pre-pulse data where arrival_p = 0
        return np.nansum(chi_sq)

    def fit_data(self, data, num_components=2, x0=None ,metric=None):
        """
        Takes in the data and adjusts the internal
        parameters of this FLIMParams object to
        minimize the metric input. Default is CHI-SQUARED.

        Inputs
        ------
        data (1d-ndarray):

            A numpy array of the arrival time histogram. Data[n] = number
            of photons arriving in bin n

        num_components (int):

            Number of differently distributed monoexponential states.

        x0 (tuple-like):

            Guess for initial params in FLIMParams.param_tuple() format.
        
        metric (function of two variables):

            Defines the cost function for curve fitting. Defaults to chi-squared

            Variable 1: DATA (1d-ndarray) as above

            Variable 2: PARAMS (tuple)

            

        """

        self.Ncomponents = num_components

        if metric is None:
            objective = lambda x: self.chi_sq(data, params=x)

        else:
            objective = lambda x: metric(data, x)

        if x0 is None:
            x0 = (0.5,120,0.5,40,20,4) # seems to generally behave well

        fit_obj = minimize(objective, x0, method='trust-constr',
               constraints=generate_linear_constraints_trust(self.param_dict()),
               bounds=generate_bounds(self.param_dict()))

        fit_tuple = fit_obj.x

        self.Exp_params = [{'FRAC': fit_tuple[2*comp], 'TAU':fit_tuple[2*comp+1]} for comp in range(self.Ncomponents)]

        self.T_O = fit_tuple[-2]
        self.IRF = {'DIST':'GAUSSIAN', 'PARAMS': {'SIGMA':fit_tuple[-1]}}

        return fit_obj

    def param_tuple(self):
        """
        Return the FLIM parameters as a tuple. Order:

        frac_1 (double) the fraction in state 1
        tau_1 (double) the time constant of state 1 (in units of histogram bins)
        ...
        ...
        frac_n
        tau_n
        tau_o (double) the offset of the laser pulse relative to the first bin
        tau_g (double) the width of the estimated instrument response Gaussian
        """

        param_tuple = []
        for comp in range(self.Ncomponents):
            param_tuple.extend([self.Exp_params[comp]['FRAC'],self.Exp_params[comp]['TAU']])
        param_tuple.append(self.T_O)
        param_tuple.append(self.IRF['PARAMS']['SIGMA'])
        return param_tuple

    def param_dict(self):
        """
        Return the FLIM parameters as a dict as follows:

        
        params(dict). Format is KEY (string) -- (type of value) value interpretation
                NCOMPONENTS -- (int) number of exponentials being fit
                SUM_TOTAL -- (int) coefficient of the overall exponential fits
                EXPPARAMS -- (list of length NCOMPONENTS, param dict for each exponential) Format:
                    FRAC -- (double) proportion of molecules in state (1.0 if monoexponential fit)
                    TAU -- (double) time constant of emission in state (in units of HISTOGRAM BINS)
                CHISQ -- (double) chi-squared statistic to assess goodness of fit
                T_O (that's an oh, not a zero) -- (double) time offset of the "zero" point of laser encountering the sample. (in units of HISTOGRAM BINS)
                IRF -- (dict) parameters relating to the (usually Gaussian) instrument response function fit:
                    DIST -- (string) name of the distribution used for the IRF. Options:
                        GAUSSIAN
                            Params:
                                SIGMA -- (double) inferred width of the Gaussian spread (in units of HISTOGRAM BINS)
                        TODO: Add more
                    PARAMS -- (dict) dictionary of IRF parameter fits
        """

        params_dict = {
            'NCOMPONENTS' : self.Ncomponents,
            'EXPPARAMS' : self.Exp_params,
            'CHISQ' : 0.0,
            'T_O' : self.T_O,
            'IRF' : self.IRF
        }
        return params_dict



### LOCAL FUNCTIONS

def generate_bounds(params):
    """
    All params > 0
    
    fracs < 1
    
    Param order:
    frac_1
    tau_1
    ...
    frac_n
    tau_n
    t_o
    sigma
    """
    lower_bounds_frac = [0 for x in range(params['NCOMPONENTS'])]
    lower_bounds_tau = [0 for x in range(params['NCOMPONENTS'])]
    
    lb = [val for pair in zip(lower_bounds_frac, lower_bounds_tau) for val in pair]
    lb.append(0.0)
    lb.append(0.0)
    
    upper_bounds_frac = [1 for x in range(params['NCOMPONENTS'])]
    upper_bounds_tau = [np.inf for x in range(params['NCOMPONENTS'])]
    
    ub = [val for pair in zip(upper_bounds_frac, upper_bounds_tau) for val in pair]
    ub.append(np.inf)
    ub.append(np.inf)
    
    return Bounds(lb, ub)

def generate_linear_constraints_trust(params):
    """ Only one linear constraint, sum of fracs == 1"""
    lin_op = np.zeros((2*params['NCOMPONENTS']+2))
    
    for comp in range(params['NCOMPONENTS']):
        lin_op[params['NCOMPONENTS']*comp] = 1
    
    return LinearConstraint(lin_op,1.0,1.0)

def generate_linear_constraints_slsqp(params):
    
    def sum_frac_is_one(params):
        return 1.0-np.sum([params[x*params['NCOMPONENTS']] for x in range(params['NCOMPONENTS'])])
    
    return [{'type' : 'eq', 'fun': sum_frac_is_one}]
                



