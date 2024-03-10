import json
from typing import Any, Callable, TYPE_CHECKING, Optional, List, Dict, Tuple, Union
from pathlib import Path
from contextlib import contextmanager

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, OptimizeResult
from scipy.stats import exponnorm

from siffpy.core.utils.types import PathLike
from siffpy.core.flim.flimunits import FlimUnitsLike, FlimUnits, convert_flimunits
from siffpy.core.flim.loss_functions import ChiSquared
from siffpy.core.flim.typing import PDF_Function
if TYPE_CHECKING:
    from siffpy.core.flim.loss_functions import LossFunction


def multi_exponential_pdf_from_params(
    x_range : np.ndarray,
    params : np.ndarray,
):
    """
    Returns the probability distribution of observing a photon at each
    time in x_range given the exponential parameters and the IRF parameters.
    """

    pdist = np.zeros(x_range.shape)
    irf_mean, irf_sigma = params[-2], params[-1]
    
    # No more loop, do it all vectorized
    params.reshape(-1, 2)
    pdist = np.sum(
        params[1:-2:2] * exponnorm.pdf(
            x_range[:, np.newaxis],
            params[:-2:2]/irf_sigma,
            loc=irf_mean,
            scale=irf_sigma
        ),
        axis=1
    )

    pdist += np.sum(
        params[1:-2:2] * exponnorm.pdf(
            x_range[:, np.newaxis] + x_range[-1],
            params[:-2:2]/irf_sigma,
            loc=irf_mean,
            scale=irf_sigma
        ),
        axis=1
    )
    return pdist/pdist.sum()

class FLIMParams():
    """
    A class for storing parameters related
    to fitting distributions of fluorescence
    lifetime or photon arrival data.

    Currently only implements combinations of exponentials.

    The FLIMParams object uses units, specifically the `FlimUnits`
    enum, to keep track of the units of the parameters.
    """
    
    def __init__(self,
        *args: Union[List['FLIMParameter'], Tuple[float]],
        color_channel : Optional[int] = None,
        noise : float = 0.0,
        name : Optional[str] = None,
        ):
        """
        initialized with a list of Exp objects and an Irf object
        in the `args` parameter. Alternatively, a tuple of parameters
        can be passed and interpreted as follows:
            (tau, frac, tau, frac, ... , irf_mean, irf_sigma)

        These will be stored in the `exps` and `irf` attributes.
        """
        
        self.exps = [arg for arg in args if isinstance(arg, Exp)]
        self.irf = next((x for x in args if isinstance(x, Irf)), None)
        if len(self.exps) == 0:
            if any((isinstance(arg, tuple) for arg in args)):
                putative_param_tuple = next((x for x in args if isinstance(x, tuple)), None)
                self.exps = [
                    Exp(
                        tau=tau,
                        frac=frac,
                    )
                    for tau, frac in zip(
                        putative_param_tuple[:-2:2],
                        putative_param_tuple[1:-2:2]
                    )
                ]
                self.irf = Irf(
                    tau_offset = putative_param_tuple[-2],
                    tau_g = putative_param_tuple[-1],
                )
            else:
                raise ValueError(
                    "At least one exponential and an IRF is required."
                    + " May be specified as a tuple or as a list of `Exp` and `IRF` objects."
                )

        if np.sum([exp.frac for exp in self.exps]) != 1:
            raise ValueError("Fractions of exponentials must sum to 1.")
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
    def params(self)->List['FLIMParameter']:
        """ Returns a list of all FLIMParameter objects contained by this FLIMParams """
        retlist = [x for x in self.exps]
        retlist += [self.irf]
        return retlist

    @property
    def param_tuple(self)->Tuple:
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
            initial_guess   : np.ndarray     = None,
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
        with self.as_units(FlimUnits.COUNTBINS):
            objective : Callable[[np.ndarray], float] = loss_function.from_data(
                data, 
                multi_exponential_pdf_from_params,
            )

            initial_guess = np.array(self.param_tuple) if initial_guess is None else np.array(initial_guess)
            
            if solver is None:
                def minimize_loss(loss_func, initial_guess):
                    return minimize(
                        loss_func,
                        loss_function.params_transform(initial_guess),
                        method = 'trust-constr',
                        bounds = self.bounds,
                        constraints = self.constraints,
                    )

                solver = minimize_loss

            print("Initial guess: ", loss_function.params_transform(initial_guess))
            fit_obj = solver(objective, initial_guess)

            fit_tuple = fit_obj.x

            self.param_tuple = loss_function.params_untransform(fit_tuple)
        return fit_obj

    @classmethod
    def from_tuple(
            cls,
            param_tuple : tuple,
            units : 'FlimUnitsLike' = FlimUnits.COUNTBINS,
            noise : float = 0.0,
        ):
        """ 
        Instantiate a FLIMParams from the parameter tuple. Order is:

        (tau, frac, tau, frac, ... , mean, sigma)
        
        """
        units = FlimUnits(units)
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
        return cls(*args, noise = noise)

    def pdf(self, x_range : np.ndarray)->np.ndarray:
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
        return self.probability_dist(x_range)

    def probability_dist(self, x_range : np.ndarray)->np.ndarray:
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
        if x_range.dtype is not float:
            x_range = x_range.astype(float)

        return (1-self.noise)*multi_exponential_pdf_from_params(
            x_range,
            np.array(self.param_tuple)
        ) + self.noise/len(x_range)

    def __repr__(self):
        retstr = "FLIMParams object: \n\n"
        retstr += "\tParameters:\n"
        for exp in self.exps:
            retstr += "\t\t"+exp.__repr__() + "\n"
        retstr += "\t\t"+self.irf.__repr__() + "\n"
        return retstr
    
    @contextmanager
    def as_units(self, to_units : FlimUnitsLike):
        """
        Context manager that temporarily converts the units of this FLIMParams
        object to the units of the first parameter, and then converts them back
        to the original units when the context is exited.
        """
        start_units = self.units
        self.convert_units(to_units)
        yield
        self.convert_units(start_units)

    @property
    def T_O(self)->float:
        """ Back-compatibility """
        return self.tau_offset

    @param_tuple.setter
    def param_tuple(self, new_params : tuple):
        """ :param new_params: a list of the new parameters for the model
        (Tau, frac, tau, frac, ... , mean, sigma).
        Does NOT change current units! Be careful!
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
    def constraints(self)->List[LinearConstraint]:
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
    def fraction_constraints(self)->List[LinearConstraint]:
        """ For when the taus and IRF are fixed """
        return [LinearConstraint(
            A=np.array([1.0]*self.n_exp),
            lb=1,
            ub=1,
        )]
    
    def to_dict(self)->Dict[str, Any]:
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
    def from_dict(cls, data_dict : Dict[str, Any])->'FLIMParams':
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

        # Make sure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
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
    
    def __eq__(self, other)->bool:
        """
        Two FLIMParams objects are equal if they have the same parameters
        and the same color channel, regardless of ordering of those parameters
        """
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
        return (
            (type(self) is type(other))
            and all(
                getattr(self, par) == getattr(other, par)
                for par in self.__class__.class_params
            )
        )


class Exp(FLIMParameter):
    """
    Monoexponential parameter fits.

    Tracks the fraction of photons belonging
    to this exponential and the corresponding
    timeconstant.
    """
    class_params = ['tau', 'frac']
    unitful_params = ['tau']
    def __init__(self, **params):
        """ Exp(tau : float, frac : float, units : FlimUnits) """
        super().__init__(**params)

    @property
    def fraction(self):
        return self.frac


class Irf(FLIMParameter):
    """
    Instrument response function.
    Tracks the mean offset and its variance
    (presumes a Gaussian IRF, which is convolved
    with the exponentials to produce the estimated
    arrival time distribution).
    """
    class_params = ['tau_offset', 'tau_g']
    unitful_params = ['tau_offset', 'tau_g']
    
    def __init__(self, **params):
        """ Irf(tau_offset : float, tau_g : float, units : FlimUnits)"""
        super().__init__(**params)

    @property
    def sigma(self):
        return self.tau_g
    
    @property
    def mu(self):
        return self.tau_offset