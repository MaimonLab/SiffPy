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

    fit_data(self, data, num_components, x0, metric) -> None:
        Takes in the ndarray data and fits the parameters of the model defined by this
        FLIMParams object. metric reflects the objective function.

    param_tuple(self) -> tuple:
        Returns a tuple listing the parameter values in a standardized order.

    param_dict(self) -> dict:
        Returns a dict listing parameter values in an easy-to-understand key value logic

    p_dist(self, x_range) -> np.ndarray:
        Return a numpy array reflecting the probability distribution of the fit variables along the input x range

    SCT March 28 2021, rainy day in Maywood NJ
    """

    def __init__(self, param_dict : dict = None, use_noise : bool = False):
        if param_dict is None:
            self.Ncomponents = None
            self.Exp_params = []
            self.IRF = {}
            self.T_O = 0
            self.CHI_SQD = np.nan
            if use_noise:
                self.noise = 0.0
        else:
            self.Ncomponents = param_dict['NCOMPONENTS']
            self.Exp_params = param_dict['EXPPARAMS']
            self.IRF = param_dict['IRF']
            self.T_O = param_dict['T_O']
            self.CHI_SQD = param_dict['CHISQ']
            if 'NOISE' in param_dict:
                self.noise = param_dict['NOISE']
            else:
                if use_noise:
                    self.noise = 0.0

    @classmethod
    def from_tuple(cls, param_tuple : tuple):
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
            'CHISQ' : np.nan,
            'T_O' : param_tuple[-2],
            'IRF' : param_tuple[-1]
        }
        return cls(param_dict=params_dict)

    def allow_noise(self):
        self.noise = 0.0 # creates the attr if it does not exist already

    def __str__(self):
        return self.param_dict().__str__()

    def __repr__(self):
        retstr = "FLIM parameters: \n"
        for key,val in self.param_dict().items():
            retstr += "\t" + str(key) + " : " + str(val) + "\n"
        return retstr

    def chi_sq(self, data : np.ndarray, params : tuple = None, use_noise : bool = False) -> float:
        """
        Computes the chi-squared statistic of the
        input data "data" using the fit parameters
        in this class.
        TODO: USE THE WRAPAROUND
        """
        if params is None:
            params = self.param_tuple()

        arrival_p = np.zeros(data.shape) # incrementally updated by each component

        if use_noise:
            for component in range(self.Ncomponents):
                arrival_p += params[2*component] * \
                    monoexponential_prob(
                        np.arange(arrival_p.shape[0])-params[-3], # arrival time shifted by t_o
                        params[2*component + 1], # this component's tau value
                        params[-2] # the IRF width
                    )
        else:
            for component in range(self.Ncomponents):
                arrival_p += params[2*component] * \
                    monoexponential_prob(
                        np.arange(arrival_p.shape[0])-params[-2], # arrival time shifted by t_o
                        params[2*component + 1], # this component's tau value
                        params[-1] # the IRF width
                    )
        
        if use_noise:
            arrival_p *= (1.0-params[-1])
            arrival_p += params[-1]/arrival_p.shape[0]
            

        total_photons = np.sum(data)
        chi_sq = ((data - total_photons*arrival_p)**2)/(total_photons*arrival_p)

        chi_sq[np.isinf(chi_sq)]=0 # ignore the pre-pulse data where arrival_p = 0
        
        true_chisq = np.nansum(chi_sq)

        return true_chisq

    def fit_data(self, data : np.ndarray, num_components : int = 2, x0 : tuple = None , metric = None) -> None:
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

        if hasattr(self, 'noise'):
            use_noise = True
        else:
            use_noise = False

        if metric is None:
            objective = lambda x: self.chi_sq(data, params=x, use_noise = use_noise)

        else:
            objective = lambda x: metric(data, x)

        if x0 is None:
            if use_noise:
                x0 = (0.5,120,0.5,40,20,4,0.1)
            else:
                x0 = (0.5,120,0.5,40,20,4) # seems to generally behave well

        fit_obj = minimize(objective, x0, method='trust-constr',
               constraints=generate_linear_constraints_trust(self.param_dict()),
               bounds=generate_bounds(self.param_dict()))

        fit_tuple = fit_obj.x

        self.Exp_params = [{'FRAC': fit_tuple[2*comp], 'TAU':fit_tuple[2*comp+1]} for comp in range(self.Ncomponents)]

        if use_noise:
            self.T_O = fit_tuple[-3]
            self.IRF = {'DIST':'GAUSSIAN', 'PARAMS': {'SIGMA':fit_tuple[-2]}}
            self.noise = fit_tuple[-1]
        else:
            self.T_O = fit_tuple[-2]
            self.IRF = {'DIST':'GAUSSIAN', 'PARAMS': {'SIGMA':fit_tuple[-1]}}

        self.CHI_SQD = self.chi_sq(data, use_noise=use_noise)

        return fit_obj

    def param_tuple(self) -> tuple:
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
        noise (double) the ratio of noise photons to total photons
        """

        param_tuple = []
        for comp in range(self.Ncomponents):
            param_tuple.extend([self.Exp_params[comp]['FRAC'],self.Exp_params[comp]['TAU']])
        param_tuple.append(self.T_O)
        param_tuple.append(self.IRF['PARAMS']['SIGMA'])
        if hasattr(self, 'noise'):
            param_tuple.append(self.noise)
        return tuple(param_tuple)

    def param_dict(self) -> dict:
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
                NOISE -- (float, optional) ratio of signal photons to noise photons
        """

        params_dict = {
            'NCOMPONENTS' : self.Ncomponents,
            'EXPPARAMS' : self.Exp_params,
            'CHISQ' : self.CHI_SQD,
            'T_O' : self.T_O,
            'IRF' : self.IRF,
        }

        if hasattr(self, 'noise'):
            params_dict['NOISE'] = self.noise

        return params_dict

    def p_dist(self, x_range : np.ndarray, **kwargs) -> np.ndarray:
        """
        Return the fit value's probability distribution. To plot against a
        data set, rescale this by the total number of photons in the data set.
        Assumes x_range is in units of BIN SIZE, not true time.

        INPUTS
        ------
        x_range : np.ndarray (1-dimensional)

            The x values you want the output probabilities of. Usually this will be something like
            np.arange(MAX_BIN_VALUE), e.g. np.arange(1024)

        RETURN VALUES
        ------------
        p_out : np.ndarray(1-dimensional)
            
            The probability of observing a photon in each corresponding bin of x_range.
        """
        
        if len(x_range.shape) > 1:
            raise Exception("x_range must be one dimensional")

        params = self.param_dict()

        arrival_p = np.zeros(x_range.shape) # incrementally updated by each component
        for component in range(params['NCOMPONENTS']):
            exp_param = params['EXPPARAMS'][component]
            arrival_p += exp_param['FRAC'] * \
                monoexponential_prob(
                    x_range-params['T_O'], # arrival time shifted by t_o
                    exp_param['TAU'], # this component's tau value
                    params['IRF']['PARAMS']['SIGMA'], # the IRF width
                    **kwargs
                )
        if 'NOISE' in params:
            arrival_p *= 1.0 - params['NOISE']
            arrival_p += params['NOISE']/arrival_p.shape[0]

        return arrival_p



### LOCAL FUNCTIONS

def generate_bounds(params : dict):
    """
    All params > 0
    
    fracs < 1

    noise < 1
    
    Param order:
    frac_1
    tau_1
    ...
    frac_n
    tau_n
    t_o
    sigma  
    noise (if present)
    """
    lower_bounds_frac = [0 for x in range(params['NCOMPONENTS'])]
    lower_bounds_tau = [0 for x in range(params['NCOMPONENTS'])]
    
    lb = [val for pair in zip(lower_bounds_frac, lower_bounds_tau) for val in pair]
    lb.append(0.0)
    lb.append(0.0)
    
    upper_bounds_frac = [1 for x in range(params['NCOMPONENTS'])]
    upper_bounds_tau = [np.inf for x in range(params['NCOMPONENTS'])]
    
    ub = [val for pair in zip(upper_bounds_frac, upper_bounds_tau) for val in pair]
    ub.append(np.inf) # tau_o
    ub.append(np.inf) # sigma

    if 'NOISE' in params:
        lb.append(0.0)
        ub.append(1.0)
    
    return Bounds(lb, ub)

def generate_linear_constraints_trust(params : dict):
    """ Only one linear constraint, sum of fracs == 1"""
    if 'NOISE' in params:
        lin_op = np.zeros((2*params['NCOMPONENTS']+3))
    else:
        lin_op = np.zeros((2*params['NCOMPONENTS']+2))
    
    for comp in range(params['NCOMPONENTS']):
        lin_op[params['NCOMPONENTS']*comp] = 1
    
    return LinearConstraint(lin_op,1.0,1.0)

def generate_linear_constraints_slsqp(params):
    
    def sum_frac_is_one(params):
        return 1.0-np.sum([params[x*params['NCOMPONENTS']] for x in range(params['NCOMPONENTS'])])
    
    return [{'type' : 'eq', 'fun': sum_frac_is_one}]
                



