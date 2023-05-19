import json
from typing import Any, Callable, TYPE_CHECKING, Optional
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, OptimizeResult
from scipy.stats import exponnorm

from siffpy.core.utils.types import PathLike
from siffpy.core.flim.flimunits import FlimUnitsLike
from siffpy.core.flim.flimunits import FlimUnits, convert_flimunits
from siffpy.core.flim.exponentials import param_tuple_to_pdf
from siffpy.core.flim.loss_functions import ChiSquared
if TYPE_CHECKING:
    from siffpy.core.flim.loss_functions import LossFunction

class FLIMParams():
    """
    A class for storing parameters related
    to fitting distributions of fluorescence
    lifetime or photon arrival data.

    Currently only implements combinations of exponentials.
    """
    
    def __init__(self,
        *args,
        color_channel : Optional[int] = None,
        noise : float = 0.0,
        name : Optional[str] = None,
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


    def convert_units(self, to_units : FlimUnitsLike, flim_info = None):
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

    @property
    def ncomponents(self)->int:
        """ Number of exponentials in this FLIMParams """
        if hasattr(self, 'exps'):
            return len(self.exps)
        return 0

    def fit_params_to_data(
            self,
            data            : np.ndarray,
            initial_guess   : tuple     = None,
            loss_function   : 'LossFunction'  = ChiSquared,
            solver          : Callable  = None,
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
        
        loss_function : LossFunction

            Defines the cost function for curve fitting. Defaults to chi-squared

            Argument 1: DATA (1d-ndarray) as above

            Argument 2: PARAMS (tuple)

            All other arguments must be KWARGS.

        solver : Callable

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
        start_units = FlimUnits(self.units) # force eval
        self.convert_units(FlimUnits.COUNTBINS)
        objective : Callable[[tuple], float] = loss_function.from_data(data)

        if initial_guess is None:
            initial_guess = self.param_tuple
        
        if solver is None:
            solver = lambda loss_func, initial_guess: minimize(
                loss_func,
                loss_function.params_transform(initial_guess),
                method = 'trust-constr',
                bounds = self.bounds,
                constraints = self.constraints,
            )

        fit_obj = solver(objective, initial_guess)

        fit_tuple = fit_obj.x

        self.param_tuple = loss_function.params_untransform(fit_tuple)
        self.convert_units(start_units)
        return fit_obj

    @classmethod
    def from_tuple(cls, param_tuple : tuple, units : FlimUnits = FlimUnits.COUNTBINS):
        """ Instantiate a FLIMParams from the parameter tuple """
        num_components = len(param_tuple) - 2

        args = []
        args += [
            Exp(
                tau=tau,
                frac =frac,
                units = units,
            )
            for tau,frac in zip(
                param_tuple[:-2:2],
                param_tuple[1:-2:2]
            )
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
        return param_tuple_to_pdf(
            x_axis = x_range,
            param_tuple = self.param_tuple,
        )

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
    
    @param_tuple.setter
    def param_tuple(self, new_params : tuple):
        """ :param new_params: a list of the new parameters for the model
        (Tau, frac, tau, frac, ... , mean, sigma) 
        """

        if len(new_params) != 2*self.n_exp + 2:
            raise ValueError(f"Incorrect number of parameters (should be {self.n_exp*2 + 2})")
        # Update the parameters
        self.exps = [
            Exp(tau=tau, frac=frac)
            for tau, frac in zip(
                new_params[:-2:2],
                new_params[1:-2:2]
            )
        ]
        self.irf = Irf(tau_offset = new_params[-2], tau_g = new_params[-1])
    
    @property
    def n_exp(self)->int:
        """ Number of exponentials in this FLIMParams """
        return len(self.exps)
    
    @property
    def bounds(self)->Bounds:
        """
        Returns a bounds object from the scipy optimization library
        that can be used to constrain the parameters of this FLIMParams
        """

        # Switch the type of bounds depending on the # of exponential
        return Bounds(
            # lower bounds
            [0, 0]*(self.n_exp + 1),
            # upper bounds
            [np.inf, 1]*self.n_exp + [100, 10],
        )
    
    @property
    def fraction_bounds(self)->Bounds:
        """ Fraction-only bounds"""
        return Bounds(
            [0.0] * self.n_exps,
            [1] * self.n_exps
        )

    @property
    def constraints(self)->list[LinearConstraint]:
        """ Exponential fractions sum to one, taus in increasing order """
        sum_exps_constraint = [
                LinearConstraint(
                A=np.array([0.0,1]*self.n_exp + [0.0,0.0]),
                lb=1,
                ub=1,
            )
        ]

        increasing_taus_constraint = [
            LinearConstraint(
                A=np.array(
                [0,0]*(exp_num-1) + # preceding exponentials
                [1,0] + [-1,0] +
                [0,0]*(self.n_exp-exp_num-1) + # tailing exponentials
                [0,0] #IRF
                ),
                ub = 0,
            )
            for exp_num in range(1, self.n_exp)
        ]
        return sum_exps_constraint + increasing_taus_constraint

    @property
    def fraction_constraints(self)->list[LinearConstraint]:
        """ For when the taus and IRF are fixed """
        return [LinearConstraint(
            A=np.array([1.0]*self.n_exp),
            lb=1,
            ub=1,
        )]
    
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
    def load(cls, path : PathLike)->'FLIMParams':
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

    def convert_units(self, to_units : FlimUnitsLike)->None:
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
        self.units = FlimUnits(to_units)

    def redimensionalize(self, to_units : FlimUnitsLike):
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
