import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, OptimizeResult

from .exponentials import chi_sq_exp
from .flimunits import FlimUnits

class FLIMParams():
    """
    A class for storing parameters related
    to fitting distributions of fluorescence
    lifetime or photon arrival data.

    Currently only implements combinations of exponentials.
    """
    
    def __init__(self, *args, color_channel : int = None, units = FlimUnits.ARRIVAL_BINS, allow_noise : bool = False, **params):
        
        self.exps = [arg for arg in args if isinstance(arg, Exp)]
        self.irf = next((x for x in args if isinstance(x, Irf)), None)
        self.color_channel = color_channel
        self.units = units
        self.allow_noise = allow_noise

    @property
    def tau_g(self)->float:
        if hasattr(self, 'irf'):
            return self.irf.tau_g

    @property
    def tau_offset(self)->float:
        if hasattr(self, 'irf'):
            return self.irf.tau_offset

    @property
    def param_tuple(self)->tuple:
        """
        Returns a tuple for all of the parameters together so they can be
        passed into numerical solvers.
        """
        retlist = []
        for exp in self.exps:
            retlist += exp.param_list
        retlist += self.irf.param_list
        return tuple(retlist)    

    def convert_units(self, to_units : FlimUnits, flim_info = None):
        """
        Converts the units of all the parameters of this FLIMParams to
        the requested unit type of to_units.

        Arguments
        ---------

        to_units : FlimUnits

            A FlimUnits object specifying the units
            into which the FLIMParams will be transformed.

        flim_info : FlimInfo

            A FlimInfo object that is necessary to determine how
            to interchange between arrival_bins and real time units
            like picoseconds and nanoseconds. If converting between
            real time units, this parameter can be ignored

        Returns
        -------
        None
        """
        raise NotImplementedError("Haven't implemented unit conversions in FLIMParams yet.")

    @property
    def ncomponents(self)->int:
        """ Number of exponentials in this FLIMParams """
        if hasattr(self, 'exps'):
            return len(self.exps)
        return 0

    def fit_to_data(self, data : np.ndarray, num_exps : int = 2, initial_guess : tuple = None, metric : callable = None)->OptimizeResult:
        """
        Takes in the data and adjusts the internal
        parameters of this FLIMParams object to
        minimize the metric input. Default is CHI-SQUARED.

        Stores new parameter values IN PLACE, but will return
        the scipy OptimizeResult object

        Inputs
        ------
        data : np.ndarray

            A numpy array of the arrival time histogram. Data[n] = number
            of photons arriving in bin n

        num_exps : int

            Number of differently distributed monoexponential states.

        initial_guess : tuple

            Guess for initial params in FLIMParams.param_tuple format.
        
        metric : callable

            Defines the cost function for curve fitting. Defaults to chi-squared

            Argument 1: DATA (1d-ndarray) as above

            Argument 2: PARAMS (tuple)

        Returns
        -------

        fit : scipy.optimize.OptimizeResult

            The OptimizeResult object for diagnostics on the fit.
        """

        if metric is None:
            objective = lambda x: chi_sq_exp(data, x)

        else:
            objective = lambda x: metric(data, x)

        # Just some numbers. Assumes lifetimes of 1200 picoseconds, 2400 picoseconds etc. ascending.
        if initial_guess is None:
            x0 = []
            for exp in range(num_exps):
                x0 += [1.0/num_exps, 6*(exp+1)]
            x0 += [4, 0.2] # tau_offset, tau_g
            initial_guess = tuple(x0) 

        fit_obj = minimize(objective, initial_guess, method='trust-constr',
               constraints=generate_linear_constraints_trust(initial_guess),
               bounds=generate_bounds(initial_guess)
        )

        fit_tuple = fit_obj.x

        self.exps = [Exp(tau=fit_tuple[2*exp_idx], frac = fit_tuple[2*exp_idx + 1]) for exp_idx in range(num_exps)]
        self.irf = Irf()
        
        self.T_O = fit_tuple[-2]
        self.IRF = Irf(tau_offset = fit_tuple[-2], tau_g = fit_tuple[-1])

        return fit_obj

    def chi_sq(self, data : np.ndarray, cut_negatives : bool = True)->float:
        """ Presumes all units are in ARRIVAL_BIN units """
        return chi_sq_exp(data, self.param_tuple, cut_negatives=cut_negatives)

    @classmethod
    def from_tuple(cls, param_tuple : tuple):
        """ Instantiate a FLIMParams from the parameter tuple """
        num_components = len(param_tuple) - 2

        args = []
        args += [
            Exp(
                tau=param_tuple[comp*num_components + 1],
                frac =param_tuple[comp*num_components]
            )
            for comp in range(num_components)
        ]

        args += [
            Irf(
                tau_offset = param_tuple[-2],
                tau_g = param_tuple[-1],
            )
        ]
        return cls(*args)

    def probability_dist(self, data :np.ndarray):
        
        raise NotImplementedError()

    def __repr__(self):
        retstr = "FLIMParams object: \n\n"
        retstr += "\tParameters:\n"
        for exp in self.exps:
            retstr += "\t\t"+exp.__repr__() + "\n"
        retstr += "\t\t"+self.irf.__repr__() + "\n"
        return retstr

    def __getattr__(self, attr : str):
        """ Back-compatibility """
        if attr == 'T_O':
            return self.tau_offset
        else:
            super().__getattr__(attr)

class FLIMParameter():
    """
    Base class for the various types of parameters.

    Doesn't do anything special, just a useful organizer
    for shared behavior.
    """
    class_params = []

    def __init__(self, **params):
        # map and filter is cuter but this is more readable.
        for key, val in params.items():
            if key in self.__class__.class_params:
                setattr(self, key, val)
        for param in self.__class__.class_params:
            if not hasattr(self, param):
                setattr(self, param, None)
    
    @property
    def param_list(self)->list:
        return [getattr(self, attr) for attr in self.__class__.class_params]

    @property
    def param_tuple(self)->tuple:
        return tuple(self.param_list)

    def __repr__(self):
        retstr = self.__class__.__name__ + "\n"
        for param in self.__class__.class_params:
            retstr += "\t" + str(param) + " : " + str(getattr(self,param)) + "\n"
        return retstr

class Exp(FLIMParameter):
    """ Monoexponential parameter fits """
    class_params = ['tau', 'frac']
    def __init__(self, **params):
        """ Exp(tau : float, frac : float) """
        super().__init__(**params)


class Irf(FLIMParameter):
    """ Instrument response function """
    class_params = ['tau_offset', 'tau_g']
    def __init__(self, **params):
        """ Irf(tau_offset : float, tau_g : float)"""
        super().__init__(**params)


### LOCAL FUNCTIONS


def generate_bounds(param_tuple : tuple)->Bounds:
    """
    All params > 0
    
    fracs < 1

    noise < 1
    
    Param order:
    tau_1
    frac_1
    ...
    tau_n
    frac_n
    t_offset
    tau_g
    """
    n_exps = (len(param_tuple)-2)//2
    lower_bounds_frac = [0 for x in range(n_exps)]
    lower_bounds_tau = [0 for x in range(n_exps)]
    
    lb = [val for pair in zip(lower_bounds_tau, lower_bounds_frac) for val in pair]
    lb.append(0.0) # tau_o
    lb.append(0.0) # tau_g
    
    upper_bounds_frac = [1 for x in range(n_exps)]
    upper_bounds_tau = [np.inf for x in range(n_exps)]
    
    ub = [val for pair in zip(upper_bounds_tau, upper_bounds_frac) for val in pair]
    ub.append(np.inf) # tau_o
    ub.append(np.inf) # tau_g

    return Bounds(lb, ub)

def generate_linear_constraints_trust(param_tuple : tuple)->LinearConstraint:
    """ Only one linear constraint, sum of fracs == 1"""

    lin_op = np.zeros_like(param_tuple)
    n_exps = (len(param_tuple)-2)//2
    # Tuple looks like:
    # (tau, frac, tau, frac, ... , tau_o, tau_g)
    for exp_idx in range(n_exps):
        lin_op[(exp_idx * 2) + 1] = 1
    
    return LinearConstraint(lin_op,1.0,1.0)