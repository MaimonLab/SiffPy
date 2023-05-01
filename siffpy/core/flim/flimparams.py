import json
from typing import Any
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, OptimizeResult

from siffpy.core.utils.types import PathLike
from siffpy.core.flim.exponentials import chi_sq_exp, monoexponential_prob
from siffpy.core.flim.flimunits import FlimUnits, convert_flimunits

class FLIMParams():
    """
    A class for storing parameters related
    to fitting distributions of fluorescence
    lifetime or photon arrival data.

    Currently only implements combinations of exponentials.
    """
    
    def __init__(self,
        *args,
        color_channel : int = None,
        noise : float = 0.0,
        name : str = None,
        ):
        
        self.exps = [arg for arg in args if isinstance(arg, Exp)]
        self.irf = next((x for x in args if isinstance(x, Irf)), None)
        self.color_channel = color_channel
        self.allow_noise = noise>0
        self.noise = noise
        self.name = name

    @property
    def tau_g(self)->float:
        if hasattr(self, 'irf'):
            return self.irf.tau_g

    @property
    def tau_offset(self)->float:
        if hasattr(self, 'irf'):
            return self.irf.tau_offset

    @property
    def params(self)->list['FLIMParameter']:
        """ Returns a list of all FLIMParameter objects contained by this FLIMParams """
        retlist = [x for x in self.exps]
        retlist += [self.irf]
        return retlist

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

    @property
    def units(self)->FlimUnits:
        if not all((x.units == self.params[0].units for x in self.params)):
            return FlimUnits.UNKNOWN
        else:
            return self.params[0].units


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
            real time units, this parameter can be ignored. NOT
            YET IMPLEMENTED ACTUALLY. TODO!!

        Returns
        -------
        None
        """
        for param in self.params:
            param.convert_units(to_units)

    def nondimensionalize(self):
        """ Converts to a non-dimensionalized unit """
        for param in self.params:
            param.nondimensionalize()
            

    def redimensionalize(self, to_units : FlimUnits):
        """ Takes a non-dimensionalized FLIMParams and converts it to the specified units """
        for param in self.params:
            param.redimensionalize(to_units)

    @property
    def ncomponents(self)->int:
        """ Number of exponentials in this FLIMParams """
        if hasattr(self, 'exps'):
            return len(self.exps)
        return 0

    def fit_to_data(
            self,
            data            : np.ndarray,
            num_exps        : int       = 2,
            initial_guess   : tuple     = None,
            metric          : callable  = None,
            solver          : callable  = None,
            **kwargs
        )->OptimizeResult:
        """
        Takes in the data and adjusts the internal
        parameters of this FLIMParams object to
        minimize the metric input. Default is CHI-SQUARED.

        Stores new parameter values IN PLACE, but will return
        the scipy OptimizeResult object.

        TODO: KEEP THIS UNITFUL

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

            All other arguments must be KWARGS.

        solver : callable

            A function that takes the metric and an initial guess and returns
            some object that has an attribute called 'x' that is a tuple with
            the same format as the FIT result of the param_tuple. This is the
            format of the default scipy.optimize.minimize functions.

        **kwargs

            Passed to the metric function.

        Returns
        -------

        fit : scipy.optimize.OptimizeResult

            The OptimizeResult object for diagnostics on the fit.
        """

        if metric is None:
            objective = lambda x: chi_sq_exp(data, x, **kwargs)

        else:
            objective = lambda x: metric(data, x, **kwargs)

        if initial_guess is None:
            if not ((len(self.exps) == 0) and (self.irf is None)):
                initial_guess = self.param_tuple
            else:
                x0 = []
                for exp in range(num_exps):
                    x0 += [60*(exp+1),1.0/num_exps]
                x0 += [40, 2.0] # tau_offset, tau_g
                initial_guess = tuple(x0) 
        
        if solver is None:
            solver = _default_solver

        # initial_guess = list(initial_guess)

        # for x in range(self.ncomponents):
        #     initial_guess[2*x]=initial_guess[2*x]/100.0

        # initial_guess[-2] = initial_guess[-2]/100.0

        fit_obj = solver(objective, initial_guess)

        fit_tuple = fit_obj.x

        # fit_tuple = list(fit_tuple)

        # for x in range(self.ncomponents):
        #     fit_tuple[2*x]=fit_tuple[2*x]*100.0

        # fit_tuple[-2] = fit_tuple[-2]*100.0
        # fit_tuple = tuple(fit_tuple)

        self.exps = [Exp(tau=fit_tuple[2*exp_idx], frac = fit_tuple[2*exp_idx + 1]) for exp_idx in range(num_exps)]
        
        self.irf = Irf(tau_offset = fit_tuple[-2], tau_g = fit_tuple[-1])

        return fit_obj

    def chi_sq(self, data : np.ndarray, negative_scope : float = 0.0)->float:
        """ Presumes all units are in ARRIVAL_BIN units. TODO make this unitful!"""
        return chi_sq_exp(data, self.param_tuple, negative_scope =negative_scope )

    @classmethod
    def from_tuple(cls, param_tuple : tuple, units : FlimUnits = FlimUnits.COUNTBINS):
        """ Instantiate a FLIMParams from the parameter tuple """
        num_components = len(param_tuple) - 2

        args = []
        args += [
            Exp(
                tau=param_tuple[comp*num_components],
                frac =param_tuple[comp*num_components + 1],
                units = units,
            )
            for comp in range(num_components)
        ]

        args += [
            Irf(
                tau_offset = param_tuple[-2],
                tau_g = param_tuple[-1],
                units = units,
            )
        ]
        return cls(*args)

    def probability_dist(self, x_range : np.ndarray, **kwargs):
        """
        Return the fit value's probability distribution. To plot against a
        data set, rescale this by the total number of photons in the data set.
        Assumes x_range is in the same units as the FLIMParams.

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
        if not len(self.exps):
            raise AttributeError("FLIMParams does not have at least one defined component.")
        if not (x_range.dtype is float):
            x_range = x_range.astype(float)
        arrival_p = np.zeros_like(x_range)
        for exp in self.exps:
            arrival_p += exp.frac * monoexponential_prob(
                x_range - self.irf.tau_offset,
                exp.tau,
                self.irf.tau_g,
                **kwargs
            )

        if self.allow_noise:
            arrival_p *= 1.0 - self.noise
        return arrival_p

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
            return super().__getattribute__(attr)

    def __eq__(self, other)->bool:
        equal = False
        if isinstance(other, FLIMParams):
            if not ((self.color_channel is None) and other.color_channel is None):
                equal *= self.color_channel == other.color_channel
            equal *= len(self.exps) == len(other.exps)
            equal *= all( # every exp has at least one match in the other FLIMParams
                (
                    any(
                        (
                            exp == otherexp
                            for otherexp in other.exps
                        )
                    )
                    for exp in self.exps
                )
            )
            equal *= self.irf == other.irf
        return equal
    
    def to_dict(self)->dict[str, Any]:
        """ Converts the FLIMParams to a JSON-compatible dictionary """
        return {
            'exps' : [exp.to_dict() for exp in self.exps],
            'irf' : self.irf.to_dict(),
            'color_channel' : self.color_channel,
            'noise' : self.noise,
            'name' : self.name,
            'units' : self.units.value,
        }

    @classmethod
    def from_dict(cls, data_dict : dict[str, Any])->'FLIMParams':
        """ Converts a JSON-compatible dictionary to a FLIMParams """
        exps = [Exp.from_dict(exp_dict) for exp_dict in data_dict['exps']]
        irf = Irf.from_dict(data_dict['irf'])
        color_channel = data_dict['color_channel']
        noise = data_dict['noise']
        name = data_dict['name']
        return cls(
            *exps,
            irf,
            color_channel = color_channel,
            noise = noise,
            name = name,
        )

    def save(self, path : PathLike):
        """ Save to a json file with different extension """
        path = Path(path)
        path = path.with_suffix(".flimparams")
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, path : PathLike):
        """ Load from a json file with different extension """
        path = Path(path)
        path = path.with_suffix(".flimparams")
        with open(path, 'r') as flim_p_file:
            flim_p_dict = json.load(flim_p_file)

        return cls.from_dict(flim_p_dict)
    
    

class FLIMParameter():
    """
    Base class for the various types of parameters.

    Doesn't do anything special, just a useful organizer
    for shared behavior.
    """
    class_params = []
    unitful_params = []

    def __init__(self, units : FlimUnits = FlimUnits.COUNTBINS, **params):
        self.units = units
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
    
    def to_dict(self)->dict:
        """ JSON-parsable string """
        return {
            **{
                param : getattr(self, param)
                for param in self.__class__.class_params
            },
            'units' : self.units.value,
        }
    
    @classmethod
    def from_dict(cls, data_dict : dict)->'FLIMParameter':
        """ Converts a JSON-compatible dictionary to a FLIMParameter """
        return cls(
            **{
                param : data_dict[param]
                for param in cls.class_params
            },
            units = FlimUnits(data_dict['units'])
        )

    def convert_units(self, to_units : FlimUnits)->None:
        """ Converts unitful params """
        for param_name in self.unitful_params:    
            setattr(
                self,
                param_name,
                convert_flimunits(
                    getattr(self, param_name),
                    self.units,
                    to_units
                )
            )
        self.units = to_units

    def redimensionalize(self, to_units : FlimUnits):
        """
        Converts UNITLESS params to to_units
        
        Alias for convert_units
        """
        self.convert_units(to_units)

    def nondimensionalize(self)->None:
        """
        Converts unitful params to nondimensional
        
        Alias for convert_units(FlimUnits.UNITLESS)
        """
        self.convert_units(FlimUnits.UNITLESS)

    def __repr__(self):
        retstr = self.__class__.__name__ + "\n"
        retstr += f"\tUNITS: {self.units}\n"
        for param in self.__class__.class_params:
            retstr += "\t" + str(param) + " : " + str(getattr(self,param)) + "\n"
        return retstr

    def __eq__(self, other)->bool:
        equal = False
        if type(self) is type(other):
            for par in self.__class__.class_params:
                equal *= getattr(self,par) == getattr(other,par)
        return equal


class Exp(FLIMParameter):
    """ Monoexponential parameter fits """
    class_params = ['tau', 'frac']
    unitful_params = ['tau']
    def __init__(self, **params):
        """ Exp(tau : float, frac : float, units : FlimUnits) """
        super().__init__(**params)


class Irf(FLIMParameter):
    """ Instrument response function """
    class_params = ['tau_offset', 'tau_g']
    unitful_params = ['tau_offset', 'tau_g']
    
    def __init__(self, **params):
        """ Irf(tau_offset : float, tau_g : float, units : FlimUnits)"""
        super().__init__(**params)


### LOCAL FUNCTIONS
def _default_solver(objective : callable, initial_guess : tuple):
    return minimize(objective, initial_guess, method='trust-constr',
            constraints=generate_linear_constraints_trust(initial_guess),
            bounds=generate_bounds_scipy(initial_guess)
        )

def generate_bounds_scipy(param_tuple : tuple)->Bounds:
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

    lin_op = np.zeros_like(param_tuple, dtype=float)
    n_exps = (len(param_tuple)-2)//2
    # Tuple looks like:
    # (tau, frac, tau, frac, ... , tau_o, tau_g)
    for exp_idx in range(n_exps):
        lin_op[(exp_idx * 2) + 1] = 1
    
    return LinearConstraint(lin_op,1.0,1.0)