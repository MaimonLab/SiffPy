import json
from typing import (
    Any, Callable, TYPE_CHECKING, Optional, List, Dict, Tuple, Union
)
from pathlib import Path
from contextlib import contextmanager

import numpy as np
from scipy.optimize import (
    minimize, Bounds, LinearConstraint, OptimizeResult
)
from scipy.stats import exponnorm

from siffpy.core.utils.types import PathLike
from siffpy.core.flim.flimunits import FlimUnitsLike, FlimUnits, convert_flimunits
from siffpy.core.flim.loss_functions import MSE
if TYPE_CHECKING:
    from siffpy.core.flim.loss_functions import LossFunction

## TODO : clean up the fitting procedure to make it faster
# and more customizable
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

def noiseless_objective(params, tau_axis, data):
    """
    Minimize the sum of squared errors
    with no noise term 
    """
    return np.sum(
        (
            (
                multi_exponential_pdf_from_params(tau_axis, params)[1:]
                - data[1:]
            )**2
            / data[1:]
        )
    )

def noisy_objective(params, tau_axis, data):
    """
    Minimize the sum of squared errors
    with a noise term
    """
    return np.sum(
        (
            (
                np.ones_like(tau_axis[1:])*params[-1]/len(tau_axis) # noise
                + (1-params[-1])*multi_exponential_pdf_from_params(tau_axis, params[:-1])[1:]
                - data[1:]
            )**2
            / data[1:]
        )
    ) 

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
        Initialized with a list of Exp objects and an Irf object
        in the `args` parameter. A `Noise` may also be provided
        as a member of the argument list, or the `noise` keyword
        argument may be provided.

        These will be stored in the `exps` and `irf` attributes.
        If there is noise in the model, `self._allow_noise` will
        be > 0

        See also:
        ---------
        `FlimParams.from_tuple`

        `FlimParams.from_dict`

        Examples
        ------------

        ### Using a list of `siffpy`'s `FLIMParameter` objects
        ```python
        from siffpy.core.flim import FLIMParams, Exp, Irf

        flim_params = FLIMParams(
            Exp(tau=0.5, frac=0.5, units = 'nanoseconds'),
            Exp(tau=3, frac=0.5, units = 'nanoseconds'),
            Irf(tau_offset=1.2, tau_g=0.02, units = 'nanoseconds'),
        )
        ```

        ### Using a tuple of parameters
        ```python
        from siffpy import FLIMParams

        flim_params = FLIMParams.from_tuple(
            (2, 0.5, 3, 0.5, 1, 0.1),
            units = 'nanoseconds'
        )
        ```

        ### Using a dictionary
        ```python
        from siffpy import FLIMParams

        flim_params = FLIMParams.from_dict(
            dict(
                exps = [
                    dict(tau=2, frac=0.5, units="NANOSECONDS"),
                    dict(tau=3, frac=0.5, units="NANOSECONDS"),
                ],
                irf = dict(tau_offset=1, tau_g=2),
                color_channel = 0,
                noise = 0.1,
                name = "Test FLIMParams",
            )
        )
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

        # if np.abs(np.sum([exp.frac for exp in self.exps]) -1) > 0.01:
        #     raise ValueError("Fractions of exponentials must sum to 1.")
        self.color_channel = color_channel
        noise_arg : Noise = next((x for x in args if isinstance(x, Noise)), None)
        if noise_arg is None:
            self._noise = noise
        else:
            self._noise = noise_arg.frac
        self.name = name

    tau_g = property(
        lambda self: self.irf.tau_g,
        lambda self, val: setattr(self.irf, 'tau_g', val)
    )

    tau_offset = property(
        lambda self: self.irf.tau_offset,
        lambda self, val: setattr(self.irf, 'tau_offset', val)
    )

    @property
    def ncomponents(self)->int:
        """ Number of exponentials in this FLIMParams """
        if hasattr(self, 'exps'):
            return len(self.exps)
        return 0

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
        return tuple([x for param in self.params for x in param.param_list])    

    @property
    def units(self)->FlimUnits:
        if not all((x.units == self.params[0].units for x in self.params)):
            return FlimUnits.UNKNOWN
        else:
            return self.params[0].units
        
    @units.setter
    def units(self, new_units : FlimUnitsLike):
        """
        Sets the units of all the parameters in this FLIMParams,
        aliases `convert_units`
        """
        self.convert_units(new_units)
                
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

        Example
        -------

        ```python
        import numpy as np
        from siffpy.core.flim import FLIMParams, Exp, Irf

        flim_params = FLIMParams(
            Exp(tau=0.5, frac=0.5, units='nanoseconds'),
            Exp(tau=3, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=1, tau_g=0.1, units='nanoseconds')
        )

        flim_params.convert_units('picoseconds')
        flim_params

        >>> FLIMParams object: 

            Parameters:
                Exp
            UNITS: FlimUnits.PICOSECONDS
            tau : 500.0
            frac : 0.5

                Exp
            UNITS: FlimUnits.PICOSECONDS
            tau : 3000.0
            frac : 0.5

                Irf
            UNITS: FlimUnits.PICOSECONDS
            tau_offset : 1000.0
            tau_g : 100.0

                Noise: 0.0
                Color channel: None
        ```
        """
        for param in self.params:
            param.convert_units(to_units)
  
    @contextmanager
    def as_units(self, to_units : FlimUnitsLike):
        """
        Context manager that temporarily converts the units of this FLIMParams
        object to the units of the first parameter, and then converts them back
        to the original units when the context is exited.

        Example
        -------

        ```python

        import numpy as np
        from siffpy.core.flim import FLIMParams, Exp, Irf

        flim_params = FLIMParams(
            Exp(tau=0.5, frac=0.5, units='nanoseconds'),
            Exp(tau=3, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=1, tau_g=0.1, units='nanoseconds')
        )

        with flim_params.as_units('picoseconds'):
            # do some stuff while this is in picoseconds
            print(flim_params)

        print(flim_params)
        >>> FLIMParams object: 

            Parameters:
                    Exp
                UNITS: FlimUnits.PICOSECONDS
                tau : 500.0
                frac : 0.2

                    Exp
                UNITS: FlimUnits.PICOSECONDS
                tau : 3000.0
                frac : 0.8

                    Irf
                UNITS: FlimUnits.PICOSECONDS
                tau_offset : 3000.0
                tau_g : 100.0

                    Noise: 0.0
                    Color channel: None

            FLIMParams object: 

                Parameters:
                    Exp
                UNITS: FlimUnits.NANOSECONDS
                tau : 0.5
                frac : 0.2

                    Exp
                UNITS: FlimUnits.NANOSECONDS
                tau : 3.0
                frac : 0.8

                    Irf
                UNITS: FlimUnits.NANOSECONDS
                tau_offset : 3.0
                tau_g : 0.1

                    Noise: 0.0
                    Color channel: None
        ```
        """
        start_units = self.units
        self.convert_units(to_units)
        yield
        self.convert_units(start_units)

    @property
    def T_O(self)->float:
        """ Back-compatibility """
        return self.tau_offset
    
    @property
    def noise(self)->float:
        """
        The noise parameter -- fraction of photons presumed
        to arise from uniformly distributed arrival times.
        """
        return self._noise
    
    @noise.setter
    def noise(self, new_noise : float):
        """ Sets the noise parameter """
        if new_noise < 0: 
            raise ValueError("Noise value must be >= 0.")
        self._noise = new_noise

    @property
    def allow_noise(self)->bool:
        """ Whether or not the noise parameter is allowed """
        if self.noise > 0:
            return True
        if not hasattr(self,'_allow_noise'):
            return False
        return self._allow_noise
        
    @allow_noise.setter
    def allow_noise(self, new_allow : bool):
        if not new_allow:
            self._noise = 0.0
        self._allow_noise = new_allow

    @param_tuple.setter
    def param_tuple(self, new_params : Tuple):
        """ :param new_params: a list of the new parameters for the model
        (Tau, frac, tau, frac, ... , mean, sigma).
        Does NOT change current units! Be careful!
        """

        if len(new_params) != 2*self.n_exp + 2:
            raise ValueError(f"Incorrect number of parameters (should be {self.n_exp*2 + 2})")
        # Update the parameters
        self.exps = [
            Exp(tau=tau, frac=frac, units = self.units)
            for tau, frac in zip(
                new_params[:-2:2],
                new_params[1:-2:2]
            )
        ]
        self.irf = Irf(
            tau_offset = new_params[-2],
            tau_g = new_params[-1],
            units = self.units
        )
    
    @property
    def n_exp(self)->int:
        """ Number of exponentials in this FLIMParams """
        return len(self.exps)
    
###### PDF ########
    
    def pdf(self, x_range : np.ndarray)->np.ndarray:
        """
        Return the fit value's probability distribution. To plot against a
        data set, rescale this by the total number of photons in the data set.
        Assumes x_range is in the same units as the FLIMParams.

        Alias for `probability_dist`

        ## Arguments
        
        - `x_range : np.ndarray` (1-dimensional)

            The x values you want the output probabilities of. Usually this will be something like
            np.arange(MAX_BIN_VALUE), e.g. np.arange(1024)

        ## Returns

        - `p_out : np.ndarray` (1-dimensional)
            
            The probability of observing a photon in each corresponding bin of x_range.

        ## Example
        
        Create a `FLIMParams` object and get the probability distribution of a photon
        arriving in any of the time bins provided

        ```python
        import numpy as np
        from siffpy.core.flim import FLIMParams, Exp, Irf

        flim_params = FLIMParams(
            Exp(tau=2, frac=0.5, units='nanoseconds'),
            Exp(tau=3, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=1, tau_g=0.1, units='nanoseconds')
        )

        flim_params.pdf(np.arange(0,12.5, 0.01))
        
        >>> array([4.42152784e-05, 4.40548485e-05, 4.38950189e-05, ...,
        4.51206365e-05, 4.49570342e-05, 4.47940435e-05])
        ```
        """
        return self.probability_dist(x_range)

    def probability_dist(self, x_range : np.ndarray)->np.ndarray:
        """
        Return the fit value's probability distribution. To plot against a
        data set, rescale this by the total number of photons in the data set.
        Assumes x_range is in the same units as the FLIMParams.

        ## Arguments
        
        - `x_range : np.ndarray` (1-dimensional)

            The x values you want the output probabilities of. Usually this will be something like
            np.arange(MAX_BIN_VALUE), e.g. np.arange(1024)

        ## Returns

        - `p_out : np.ndarray` (1-dimensional)
            
            The probability of observing a photon in each corresponding bin of x_range.

        ## Example
        
        Create a `FLIMParams` object and get the probability distribution of a photon
        arriving in any of the time bins provided

        ```python
        import numpy as np
        from siffpy.core.flim import FLIMParams, Exp, Irf

        flim_params = FLIMParams(
            Exp(tau=2, frac=0.5, units='nanoseconds'),
            Exp(tau=3, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=1, tau_g=0.1, units='nanoseconds')
        )

        flim_params.pdf(np.arange(0,12.5, 0.01))
        
        >>> array([4.42152784e-05, 4.40548485e-05, 4.38950189e-05, ...,
        4.51206365e-05, 4.49570342e-05, 4.47940435e-05])
        ```
        """
        if not len(self.exps):
            raise AttributeError("FLIMParams does not have at least one defined component.")
        if x_range.dtype is not float:
            x_range = x_range.astype(float)

        return (1-self.noise)*multi_exponential_pdf_from_params(
            x_range,
            np.array(self.param_tuple)
        ) + self.noise/len(x_range)

##### FITTING #####

    @property
    def bounds(self)->Bounds:
        """
        Returns a bounds object from the scipy optimization library
        that can be used to constrain the parameters of this FLIMParams
        """
        lb = [0]*(2*self.n_exp + 2) # all taus and fracs are non-negative
        # fracs are between 0 and 1, everything else
        # is non-negative
        ub = [np.inf, 1]*self.n_exp + [np.inf, np.inf]
        if self.n_exp == 1:
            lb[1] = 1
        if self.allow_noise:
            # noise is between 0 and 1
            lb.append(0)
            ub.append(1)
        return Bounds(lb, ub)
    
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
        if self.n_exp == 1:
            return []
        A_exps_sum = [0.0,1]*self.n_exp + [0.0,0.0]
        if self.allow_noise:
            A_exps_sum += [0.0]

        sum_exps_constraint = [
            LinearConstraint(
                A=np.array(A_exps_sum),
                lb=1,
                ub=1,
            )
        ]

        A_inc_tau = [
            [0,0]*(exp_num-1) + # preceding exponentials
            [-1,0] + [1,0] + # current exponential
            [0,0]*(self.n_exp-exp_num-1) + # tailing exponentials
            [0,0] #IRF
            for exp_num in range(1, self.n_exp)
        ]

        if self.allow_noise:
            for tau_const in A_inc_tau:
                tau_const += [0] # add the noise param

        increasing_taus_constraint = [
            LinearConstraint(
                A=np.array(tau_const),
                lb = 0,
                ub = np.inf,
            )
            for tau_const in A_inc_tau
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
    
    def fit_params_to_data(
            self,
            data            : np.ndarray,
            initial_guess   : Optional[np.ndarray]     = None,
            loss_function   : 'LossFunction'  = MSE,
            solver          : Optional[Callable]  = None,
            x_range         : Optional[np.ndarray] = None,
            optimization_units : FlimUnitsLike = FlimUnits.NANOSECONDS,
            **kwargs
        )->OptimizeResult:
        """
        Takes in the data and adjusts the internal
        parameters of this FLIMParams object to
        minimize the metric input. Default is CHI-SQUARED.

        Stores new parameter values IN PLACE, but will return
        the scipy OptimizeResult object.

        ACTUALLY NO LONGER USING LOSS_FUNCTION AND SOLVER ---
        FUNCTION CALL OVERHEAD WAS MAKING IT VERY SLOW. TO DO:
        FIGURE OUT A WAY TO PRESERVE THAT FLEXIBILITY!!

        Inputs
        ------
        - `data : np.ndarray`

            A numpy array of the arrival time histogram. Data[n] = number
            of photons arriving in bin n

        - `initial_guess : tuple`

            Guess for initial params in FLIMParams.param_tuple format.
            Presumed to be in the same units as the 
            `optimization_units` (if not None).
        
        - `loss_function : LossFunction`

            Defines the cost function for curve fitting. Defaults to chi-squared

            Argument 1: DATA (1d-ndarray) as above

            Argument 2: PARAMS (tuple)

            All other arguments must be KWARGS.

        - `solver : Callable`

            A function that takes the metric and an initial guess and returns
            some object that has an attribute called 'x' that is a tuple with
            the same format as the FIT result of the param_tuple. This is the
            format of the default `scipy.optimize.minimize` functions.

        - `x_range : np.ndarray`

            The range of the x-axis in the same units as the FLIMParams.
            If not provided, assumes the data is in countbins and uses
            np.arange(len(data)) as the x_range.

        - `**kwargs`

            Passed to the `scipy.optimize.minimize` `opts` kwarg.

        Returns
        -------

        - `fit : scipy.optimize.OptimizeResult`

            The OptimizeResult object for diagnostics on the fit.

        Examples
        ---------
        
        ### Create a `FLIMParams` object that generates the data,
        ### sample from it, then fit a new `FLIMParams` object to
        ### those data.


        ```python
        from siffpy.core.flim import FLIMParams, Exp, Irf
        import numpy as np

        flim_params = FLIMParams(
            Exp(tau=0.4, frac=0.5, units='nanoseconds'),
            Exp(tau=3, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=1, tau_g=0.05, units='nanoseconds')
        )

        # 12.5 nanoseconds in 10 ps increments
        arrival_time_axis = np.arange(0.01,12.5, 0.01)
        # Sample from the distribution, using a seed to make sure your result matches the example
        data = flim_params.sample(arrival_time_axis, 1000000, seed = 120, as_histogram = True)

        # A particularly bad guess
        fit_params = FLIMParams(
            Exp(tau=0.1, frac=0.5, units='nanoseconds'),
            Exp(tau=0.2, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=5, tau_g=0.1, units='nanoseconds')
        )

        fit_params.fit_params_to_data(data.astype(float), x_range = arrival_time_axis[1:])

        # A not-so-bad result
        fit_params


        >>> FLIMParams object: 

            Parameters:
                Exp
            UNITS: FlimUnits.NANOSECONDS
            tau : 0.39989360953557634
            frac : 0.5006910298045519

                Exp
            UNITS: FlimUnits.NANOSECONDS
            tau : 2.9891959985672134
            frac : 0.49930897019544795

                Irf
            UNITS: FlimUnits.NANOSECONDS
            tau_offset : 1.0102523399410983
            tau_g : 0.05007446535532903

            Noise: 0.0
            Color channel: None
        """
        optimization_units = FlimUnits(optimization_units)
        
        if (
            optimization_units is FlimUnits.COUNTBINS
            and x_range is None
        ):
            x_range = np.arange(len(data))

        assert x_range is not None, "Must provide x_range for solutions in real time units"

        # Convert units and run the solver
        with self.as_units(optimization_units):
            if initial_guess is None:
                initial_guess = self.param_tuple
            initial_guess = np.array(initial_guess)
            
            if self.allow_noise:
                if initial_guess.size == len(self.param_tuple):
                    initial_guess = np.append(initial_guess, self.noise)

            data /= np.sum(data)

            fit_obj = minimize(
                noisy_objective if self.allow_noise else noiseless_objective,
                initial_guess,
                args = (x_range, data),
                method = 'trust-constr',
                bounds = self.bounds,
                constraints = self.constraints,
                options = dict(
                    maxiter = 2000,
                    **kwargs
                )
            )

            fit_tuple = fit_obj.x
            if self.allow_noise:
                self.param_tuple = fit_tuple[:-1]
                self.noise = fit_tuple[-1]
            else:
                self.param_tuple = fit_tuple

        fit_obj.x = FlimUnits.convert_flimunits(
            fit_obj.x,
            optimization_units,
            self.units
        )
        return fit_obj
    
###### STORING ######
    
    def to_dict(self)->Dict[str, Any]:
        """ Converts the FLIMParams to a JSON-compatible dictionary """
        return {
            'exps' : [exp.to_dict() for exp in self.exps],
            'irf' : self.irf.to_dict(),
            'color_channel' : self.color_channel,
            'noise' : self.noise,
            'name' : self.name,
            'units' : self.units.value,
            'class' : self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data_dict : Dict[str, Any])->'FLIMParams':
        """
        Converts a JSON-compatible dictionary to a FLIMParams

        ## Example code:

        ```python

        flim_params = FLIMParams.from_dict(
            dict(
                exps = [
                    dict(tau=2, frac=0.5, units="NANOSECONDS"),
                    dict(tau=3, frac=0.5, units="NANOSECONDS"),
                ],
                irf = dict(tau_offset=1, tau_g=2),
                color_channel = 0,
                noise = 0.1,
                name = "Test FLIMParams",
            )
        )
        ```
        
        """
        exps = [Exp.from_dict(exp_dict) for exp_dict in data_dict['exps']]
        irf = Irf.from_dict(data_dict['irf'])
        color_channel = data_dict['color_channel'] if 'color_channel' in data_dict else None
        noise = data_dict['noise'] if 'noise' in data_dict else 0.0
        name = data_dict['name'] if 'name' in data_dict else None
        return cls(
            *exps,
            irf,
            color_channel = color_channel,
            noise = noise,
            name = name,
        )
    
    @classmethod
    def from_tuple(
            cls,
            param_tuple : Tuple,
            units : 'FlimUnitsLike' = FlimUnits.COUNTBINS,
            noise : float = 0.0,
        ):
        """ 
        Instantiate a FLIMParams from the parameter tuple. Order is:

        (tau, frac, tau, frac, ... , mean, sigma).

        Example code:
        -------------

        ```python
        flim_params = FLIMParams.from_tuple(
            (2, 0.5, 3, 0.5, 5, 1, 2),
            units = FlimUnits.PICOSECONDS,
        )
        ```

        
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

        if 'class' in flim_p_dict:
            # If the class is specified, use that --
            # old versions of the FLIMParams class
            # did not save the class name.
            if not flim_p_dict['class'] == cls.__name__:
                raise ValueError(
                    "Class name in file does not match the "
                    "class name of the object. Try using the "
                    f" `load` method of {flim_p_dict['class']}."
                )
            del flim_p_dict['class']

        return cls.from_dict(flim_p_dict)
    
    def sample(
            self,
            x_range : np.ndarray,
            n : int = 10000,
            seed : Optional[int] = None,
            as_histogram : bool = True,
        )->np.ndarray:
        """
        Sample the arrival time distribution of this FLIMParams

        Arguments
        ---------
        x_range : np.ndarray

            The range of the x-axis in the same units as the FLIMParams.
            If using `as_histogram`, this will return a histogram of the
            samples binned as in `x_range` (so the returned array
            will have shape (len(x_range)-1,) if `as_histogram` is True).

        n : int

            The number of samples to take.

        seed : int

            The seed for the `numpy` random number generator.

        as_histogram : bool

            If True, returns a histogram of the samples, binned as in
            `x_range`. If False, returns the samples themselves.

        Returns
        -------
        samples : np.ndarray

            An array of n samples from the arrival time distribution if
            `as_histogram` is False. If `as_histogram` is True, returns
            a histogram of the samples binned as in `x_range` of length
            `len(x_range) - 1`.

        Example
        -------

        ### Return a sample of 3 arrival times from the FLIMParams object.

        ```python
        import numpy as np
        from siffpy.core.flim import FLIMParams, Exp, Irf

        flim_params = FLIMParams(
            Exp(tau=2, frac=0.5, units='nanoseconds'),
            Exp(tau=3, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=1, tau_g=0.1, units='nanoseconds')
        )

        flim_params.sample(np.arange(0,12.5,1000), 3, seed = 15, as_histogram = False)

        >>> array([5.73, 6.9 , 3.95])

        samps = flim_params.sample(np.arange(0,12.5,1), 100000, seed = 15, as_histogram = True)

        print(samps.shape, np.arange(0,12.5,1).shape)

        >>> (12,) (13,)
        ```

        ### Return a histogram of 100000 samples from the FLIMParams object,
        ### binned into 1 nanosecond bins

        ```python
        import numpy as np
        from siffpy.core.flim import FLIMParams, Exp, Irf

        flim_params = FLIMParams(
            Exp(tau=0.5, frac=0.5, units='nanoseconds'),
            Exp(tau=3, frac=0.5, units='nanoseconds'),
            Irf(tau_offset=1, tau_g=0.1, units='nanoseconds')
        )

        flim_params.sample(np.arange(0,12.5,1), 100000, seed = 15, as_histogram = True)

        >>> array([  367, 46941, 23666,  9627,  5890,  4053,  2939,  2114,  1536,
        1098,   789,   980])
        ```
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        samples = rng.choice(
            x_range,
            size = n,
            p = self.pdf(x_range)
        )
        if not as_histogram:
            return samples

        return np.histogram(samples, bins = x_range)[0]
    
    def fraction_to_empirical(self, fractions : np.ndarray) -> np.ndarray:
        """
        Converts the `fractions` argument into the equivalent empirical lifetime
        using the exponential parameter fits.

        ## Arguments

        - `fractions : np.ndarray`

            An array of fractions of the total photons in each bin. Should be
            dimensions (..., self.n_exps - 1), with the presumption that the final
            bin adds up to 1 (i.e. fractions.sum(axis=-1) <= 1)

        ## Returns

        - `empirical : np.ndarray`

            An array of the empirical lifetimes corresponding to the fractions
            in the input array.
        """
        if self.n_exp == 2:
            fractions = np.expand_dims(fractions, axis=-1)

        return (
            fractions @ np.array([exp.tau for exp in self.exps[:-1]]) 
            + ((1-fractions.sum(axis=-1)) * self.exps[-1].tau)
        )


    
    def __eq__(self, other)->bool:
        """
        Two FLIMParams objects are equal if they have the same parameters
        and the same color channel, regardless of ordering of those parameters
        """
        equal = False
        if isinstance(other, FLIMParams):
            equal = True
            if (self.color_channel is not None) and (other.color_channel is not None):
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
    
    def __repr__(self):
        """ TODO: Pretty-print the equations in the repr """
        retstr = "FLIMParams object: \n\n"
        retstr += "\tParameters:\n"
        for exp in self.exps:
            retstr += "\t\t"+exp.__repr__() + "\n"
        retstr += "\t\t"+self.irf.__repr__() + "\n"
        retstr += "\t\tNoise: " + str(self.noise) + "\n"
        retstr += "\t\tColor channel: " + str(self.color_channel) + "\n"
        return retstr

    
class FLIMParameter():
    """
    Base class for the various types of parameters.

    Doesn't do anything special, just a useful organizer
    for shared behavior.

    FLIMParameters can be converted between units, and
    can be serialized to and from JSON.

    The `units` attribute is used to keep track of the units
    of the parameters. The `unitful_params` attribute is used
    to keep track of which parameters are unitful and need to
    be converted when the units are changed.

    The `aliases` attribute is used to keep track of aliases
    for the parameters, so that they can be accessed with
    different names (each of which is more intuitive to 
    certain users)
    """
    class_params = []
    unitful_params = []

    # aliases for the parameters
    # format is {true_param_name : [alias1, alias2, ...]}
    aliases : Dict[str, List[str]] = {}

    def __init__(self, units : FlimUnits = FlimUnits.COUNTBINS, **params):
        self.units = FlimUnits(units)

        # Use the aliases to build properties with getters and setters
        # for all the parameters allowing access with any aliases

        for true_param, aliases in self.aliases.items():
            for alias in aliases:
                setattr(
                    self.__class__,
                    alias,
                    property(
                        lambda self: getattr(self, true_param),
                        lambda self, val: setattr(self, true_param, val)
                    )
                )
                if alias in params:
                    params[true_param] = params[alias]
                    del params[alias]
        
        for key, val in params.items():
            if key in self.__class__.class_params:
                setattr(self, key, val)
        for param in self.__class__.class_params:
            if not hasattr(self, param):
                setattr(self, param, None)

    @property
    def param_list(self)->List:
        return [getattr(self, attr) for attr in self.__class__.class_params]

    @property
    def param_tuple(self)->Tuple:
        return tuple(self.param_list)
    
    def to_dict(self)->Dict:
        """ JSON-parsable string """
        return {
            **{
                param : getattr(self, param)
                for param in self.__class__.class_params
            },
            'units' : self.units.value,
        }
    
    @classmethod
    def from_dict(cls, data_dict : Dict)->'FLIMParameter':
        """ Converts a JSON-compatible dictionary to a FLIMParameter """
        return cls(**data_dict)

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

    Params:
    -------
    tau : float
        Time constant of the exponential
    
    frac : float
        Fraction of photons in this exponential

    Aliases:
    --------
    tau : ['lifetime']
    frac : ['fraction']
    """
    class_params = ['tau', 'frac']
    unitful_params = ['tau']
    aliases = {
        'frac' : ['fraction', 'f'],
        'tau' : ['lifetime', 'exp', 'average', 'mean', 'empirical_lifetime'],
    }

class Irf(FLIMParameter):
    """
    Instrument response function.
    Tracks the mean offset and its variance
    (presumes a Gaussian IRF, which is convolved
    with the exponentials to produce the estimated
    arrival time distribution).

    ## Params:
    tau_offset : float
        Mean offset of the IRF
    tau_g : float
        Width of the IRF

    ## Aliases:
    
    `tau_offset` : `['mu', 'mean', 'offset']`
    `tau_g` : `['sigma', 'width']`
    """
    class_params = ['tau_offset', 'tau_g']
    unitful_params = ['tau_offset', 'tau_g']

    aliases = {
        'tau_offset' : ['mu', 'mean', 'offset'],
        'tau_g' : ['sigma', 'width']
    }

class Noise(FLIMParameter): 
    """
    Background noise signal, assumes a
    uniform photon arrival time.

    Unitless

    ## Params:

    `frac : float`

    ## Aliases:

    `frac : 'fraction'`
    """
    class_params = ['frac']
    unitful_params = []

    aliases = {
        'frac' : ['fraction']
    }
