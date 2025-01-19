"""
Extends the FLIMParams class for situations
involving multiple pulses, and so multiple IRFs.
"""
from typing import (
    Any, Union, List, Tuple, Dict, Optional,
    TYPE_CHECKING, Callable, Iterator,
)

import numpy as np

from siffpy.core.flim.flimparams import (
    FLIMParams, FLIMParameter, Irf, Exp,
    multi_exponential_pdf_from_params,
    Bounds, LinearConstraint, MSE,
    OptimizeResult, minimize
)
from siffpy.core.flim.flimunits import (
    FlimUnits, FlimUnitsLike
)

if TYPE_CHECKING:
    from siffpy.core.flim.loss_functions import LossFunction


def noiseless_objective(params, tau_axis, data, n_pulses):
    """
    Minimize the sum of squared errors
    with no noise term 
    """
    PARAMS_PER_PULSE = 3
    exp_params = params[:-n_pulses*PARAMS_PER_PULSE]
    irf_params = params[-n_pulses*PARAMS_PER_PULSE:].reshape(
        (n_pulses, PARAMS_PER_PULSE)
    )

    irf_contributions = np.array([
        multi_exponential_pdf_from_params(
            tau_axis,
            np.append(exp_params, irf_params[i, :-1])
        ) for i in range(n_pulses)
    ])

    total_irf_contribution = np.sum(irf_contributions * irf_params[:, -1][:, None], axis=0)
    squared_error = (total_irf_contribution[1:] - data[1:]) ** 2
    normalized_error = squared_error / data[1:]

    # Sum up normalized errors
    return np.sum(normalized_error)

def noisy_objective(params, tau_axis, data, n_pulses):
    """
    Minimize the sum of squared errors
    with a noise term
    """
    PARAMS_PER_PULSE = 3
    exp_params = params[:(-n_pulses*PARAMS_PER_PULSE - 1)]
    irf_params = params[-n_pulses*PARAMS_PER_PULSE-1:-1].reshape(
        (n_pulses, PARAMS_PER_PULSE)
    )

    # Sigh, back to a loop
    irf_contributions = np.array([
        multi_exponential_pdf_from_params(
            tau_axis,
            np.append(exp_params, irf_params[i, :-1])
        ) for i in range(n_pulses)
    ])

    # Weight by frac
    total_irf_contribution = np.sum(irf_contributions * irf_params[:, -1][:, None], axis=0)
    with_noise = (1-params[-1])*total_irf_contribution + params[-1]/len(tau_axis)
    squared_error = (with_noise[1:] - data[1:]) ** 2
    normalized_error = squared_error / data[1:]

    # Sum up normalized errors
    return np.sum(normalized_error)

class FractionalIrf(Irf):
    """
    Promotes an `Irf` to a
    `MultiIrf` compatible tool
    by adding a fraction parameter
    """
    class_params = Irf.class_params + ['frac']
    aliases = {
        **Irf.aliases,
        'frac' : ['f', 'fraction']
    }

class MultiIrf(FractionalIrf):
    """
    Stores a list of `FractionalIrf` objects.
    Does not really behave like a `FLIMParameter`,
    so maybe this means I'm doing this wrong...
    """

    def __init__(self, *args):
        if not all(isinstance(arg, Irf) for arg in args):
            raise ValueError(
                "MultiIrf must be initialized with Irfs"
            )
        
        if any(not isinstance(arg, FractionalIrf) for arg in args):
            SyntaxWarning(
                "`MultiIrf` should be initialized with `FractionalIrfs`"
                + " but was initialized with `Irf`s. Will assume"
                + " all Irfs have equal fraction."
            )
            frac = 1.0 / len(args)
            args = [
                FractionalIrf(
                    tau_g = irf.tau_g,
                    tau_offset = irf.tau_offset,
                    frac = frac,
                    units = irf.units
                )
                if not isinstance(irf, FractionalIrf)
                else irf
                for irf in args
            ]

        if abs(sum(irf.frac for irf in args) - 1.0) > 1e-6:
            raise ValueError(
                "Fractions of IRFs must sum to 1, but sum to "
                + str(sum(irf.frac for irf in args))
                + " instead."
            )

        self.irfs = list(args)

    def __getattribute__(self, __name: str) -> Any:
        """ If it's an attribute of a FractionalIrf, return
        a list of those attributes. """
        if __name in FractionalIrf.class_params:
            return [
                getattr(irf, __name)
                for irf in self.irfs
            ]
        return super().__getattribute__(__name)

    @property
    def units(self)->Union[FlimUnits, List[FlimUnits]]:
        """
        Units of the IRFs. Returns a list if the IRFs
        have different units, otherwise returns just a single
        `FlimUnits`.
        """
        if all(irf.units == self.irfs[0].units for irf in self.irfs):
            return self.irfs[0].units
        return [irf.units for irf in self.irfs]
    @property
    def n_irfs(self)->int:
        """
        Number of IRFs in the model
        """
        return len(self.irfs)
    
    @property
    def n_params(self)->int:
        """
        Number of parameters in the model
        """
        return len(self.irfs) * len(FractionalIrf.class_params)
    
    @property
    def param_list(self)->List:
        """ Flattened list of parameters """
        return [
            param
            for irf in self.irfs
            for param in irf.param_list
        ]

    @property
    def param_tuple(self)->Tuple[float]:
        """
        Returns a tuple of the parameters (flattened).
        """
        return tuple(
            param
            for irf in self.irfs
            for param in irf.param_tuple
        )
    
    def to_dict(self)->List:
        """ JSON-parsable string, list of dicts """
        return [
            irf.to_dict()
            for irf in self.irfs
        ]
    
    @classmethod
    def from_dict(cls, data:List[Dict[str, Any]]):
        """
        Reconstructs the MultiIrf from a list of dictionaries
        """
        return cls(
            *[FractionalIrf.from_dict(irf) for irf in data]
        )

    def convert_units(self, to_units: FlimUnitsLike):
        """
        Converts the IRF parameters to the given units
        """
        for irf in self.irfs:
            irf.convert_units(to_units)
    
    def redimensionalize(self, to_units: FlimUnitsLike):
        """
        Redimensionalizes the IRF parameters
        """
        for irf in self.irfs:
            irf.redimensionalize(to_units)

    def nondimensionalize(self) -> None:
        """
        Nondimensionalizes the IRF parameters
        """
        for irf in self.irfs:
            irf.nondimensionalize()

    def __repr__(self):
        return f"MultiIrf({self.irfs})"
    
    def __getitem__(self, index:int)->FractionalIrf:
        """
        Returns the Irf at the given index
        """
        return self.irfs[index]
    
    def __setitem__(self, index:int, value:FractionalIrf):
        """
        Sets the Irf at the given index
        """
        self.irfs[index] = value

    def __iter__(self) -> Iterator[FractionalIrf]:
        """
        Iterates over the IRFs
        """
        return iter(self.irfs)
    
    def __len__(self):
        """
        Returns the number of IRFs
        """
        return len(self.irfs)

class MultiPulseFLIMParams(FLIMParams):
    """
    Specialized FLIMParams that permits more than
    one IRF. Much of the functionality is re-written,
    which makes inheritance seem... silly. I'll have
    to think about a better way at some point.
    """    
    def __init__(
        self,
        *args : Union[List['FLIMParameter'], Tuple[float]],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # If no MultiIRFs given, assume all Irfs provided
        # are equal fraction, at least until the fitting happens.
        if not any(isinstance(arg, MultiIrf) for arg in args):
            irfs = [arg for arg in args if isinstance(arg, Irf)]
            self.irf = MultiIrf(*irfs)
        else:
            self.irf = next(
                arg for arg in args if isinstance(arg, MultiIrf)
            )

    @property
    def offsets(self) -> List[Tuple[float, float]]:
        return [
            (irf.tau_offset, irf.frac)
            for irf in self.irfs
        ]

    @property
    def irfs(self)->MultiIrf:
        """
        Returns the MultiIrf object
        """
        return self.irf
    
    @irfs.setter
    def irfs(self, new_irfs:MultiIrf):
        """
        Sets the MultiIrf object
        """
        self.irf = new_irfs
    
    @property
    def n_pulses(self)->int:
        """
        Number of laser pulses in the model
        """
        return self.irf.n_irfs
    
    @property
    def n_params(self)->int:
        """
        Number of parameters in the model
        """
        return (
            self.n_exp*2 
            + self.n_pulses*len(FractionalIrf.class_params)
            + self.allow_noise
        )
    
    @property
    def param_tuple(self)->Tuple:
        return super().param_tuple

    @param_tuple.setter
    def param_tuple(self, new_params : Tuple):
        """ :param new_params: a list of the new parameters for the model
        (Tau, frac, tau, frac, ... , mean, sigma, frac, mean, sigma, frac ...).
        Does NOT change current units! Be careful!
        """

        if len(new_params) != 2*self.n_exp + 3*self.n_pulses:
            raise ValueError(f"Incorrect number of parameters (should be {self.n_exp*2 + 3*self.n_pulses})")
        # Update the parameters
        self.exps = [
            Exp(tau=tau, frac=frac, units = self.units)
            for tau, frac in zip(
                new_params[:2*self.n_exp:2],
                new_params[1:2*self.n_exp:2]
            )
        ]
        self.irfs = MultiIrf(
            *[
                FractionalIrf(
                    tau_g = sigma,
                    tau_offset = mean,
                    frac = frac,
                    units = firf.units
                )
                for mean, sigma, frac, firf in zip(
                    new_params[2*self.n_exp::3],
                    new_params[2*self.n_exp+1::3],
                    new_params[2*self.n_exp+2::3],
                    self.irfs
                )
            ]
        )
    
######## PDF #######
    
    def pdf(self, x_range : np.ndarray)->np.ndarray:
        """
        Returns the PDF at the given x value
        """
        return self.probability_dist(x_range)
    
    def probability_dist(self, x_range : np.ndarray)->np.ndarray:
        """
        Returns the probability distribution at the given x value
        """
        if not len(self.exps):
            raise AttributeError("FLIMParams does not have at least one defined component.")
        if x_range.dtype is not float:
            x_range = x_range.astype(float)

        pdist = np.zeros(x_range.shape)
        exp_tuple = np.array(
            [param for exp in self.exps for param in exp.param_tuple]
        )
        for irf in self.irfs:
            param_tuple = np.append(
                exp_tuple, irf.param_tuple[:-1]
            )
            pdist += irf.frac * (
                multi_exponential_pdf_from_params(
                    x_range,
                    param_tuple
                )
            )

        return (1-self.noise)*pdist + self.noise/len(x_range)

######## FITTING ########
    
    @property
    def bounds(self)->Bounds:
        """
        Returns the bounds for the parameters
        """
        lb = [0]*(len(self.exps)*2 + self.irf.n_params)
        ub = (
            [np.inf, 1]*len(self.exps)
            + [np.inf, np.inf, 1]*self.n_pulses
        )
        if self.allow_noise:
            lb += [0]
            ub += [1]
        return Bounds(lb, ub)
    
    @property
    def constraints(self)->List[LinearConstraint]:
        """
        Returns the constraints for the parameters
        """
        if (self.n_exp == 1) and (self.n_pulses == 1):
            return []
        A_exps_sums = (
            [0.0, 1] * self.n_exp 
            + [0.0,0.0,0.0]*self.n_pulses
        )
        if self.allow_noise:
            A_exps_sums += [0.0]
        
        sum_exps_constraint = [ # Exponential fractions sum to 1
            LinearConstraint(
                A = A_exps_sums,
                lb = 1,
                ub = 1
            )
        ]

        A_irfs_sums = (
            [0.0, 0.0] * self.n_exp 
            + [0.0, 0.0, 1] * self.n_pulses
        )
        if self.allow_noise:
            A_irfs_sums += [0.0]

        sum_irfs_constraint = [ # IRF fractions sum to 1
            LinearConstraint(
                A = A_irfs_sums,
                lb = 1,
                ub = 1
            )
        ]

        # Taus in increasing order
        A_inc_tau = []
        if self.n_exp > 1:
            A_inc_tau = [
                [0,0] * (exp_num-1) + #preceding exponentials
                [-1, 0] + [1,0] + # current exponential
                [0,0] * (self.n_exp - exp_num -1) + # following exponentials
                [0,0,0]*self.n_pulses # irfs don't matter
                for exp_num in range(1, self.n_exp)
            ]

            if self.allow_noise:
                for tau_const in A_inc_tau:
                    tau_const += [0]
        
        increasing_taus_constraint = [
            LinearConstraint(
                A = tau_const,
                lb = 0,
                ub = np.inf
            )
            for tau_const in A_inc_tau
        ]
        
        # IRFs in increasing order
        A_inc_irf = []
        if self.n_pulses > 1:
            A_inc_irf = [
                [0,0] * self.n_exp + # exponentials don't matter
                [0,0,0] * (irf_num - 1) + # preceding irfs
                [-1,0,0] + [1,0,0] + # current irf
                [0,0,0] * (self.n_pulses - irf_num - 1) # following irfs
                for irf_num in range(1, self.n_pulses)
            ]

            if self.allow_noise:
                for irf_const in A_inc_irf:
                    irf_const += [0]

        increasing_mus_constraints = [
            LinearConstraint(
                A = mu_const,
                lb = 0,
                ub = np.inf
            )
            for mu_const in A_inc_irf
        ]

        return (
            sum_exps_constraint
            + sum_irfs_constraint
            + increasing_taus_constraint
            + increasing_mus_constraints
        )
    
    def fit_params_to_data(
        self,
        data : np.ndarray,
        initial_guess : Optional[np.ndarray] = None,
        loss_function : 'LossFunction' = MSE,
        solver : Optional[Callable] = None,
        x_range : Optional[np.ndarray] = None,
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
        data : np.ndarray

            A numpy array of the arrival time histogram. Data[n] = number
            of photons arriving in bin n

        initial_guess : tuple

            Guess for initial params in FLIMParams.param_tuple format.
            Presumed to be in the same units as the FLIMParams
            when the function is called (if not None).
        
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

        x_range : np.ndarray

            The range of the x-axis in the same units as the FLIMParams.
            If not provided, assumes the data is in countbins and uses
            np.arange(len(data)) as the x_range.

        **kwargs

            Passed to the metric function.

        Returns
        -------

        fit : scipy.optimize.OptimizeResult

            The OptimizeResult object for diagnostics on the fit.
        """
        optimization_units = FlimUnits(optimization_units)
        if  (
            optimization_units is FlimUnits.COUNTBINS
            and x_range is None
        ):
            x_range = np.arange(len(data))
        
        if x_range is None:
            raise ValueError(
                "Must provide x_range for solutions in real time units"
            )

        with self.as_units(optimization_units):
            if initial_guess is None:
                initial_guess = self.param_tuple
            initial_guess = np.array(initial_guess)
        
            if self.allow_noise and (len(initial_guess) == self.n_params - 1):
                initial_guess = np.append(initial_guess, self.noise)

            data /= data.sum()

            fit_obj = minimize(
                noisy_objective if self.allow_noise else noiseless_objective,
                initial_guess,
                args = (x_range, data, self.n_pulses),
                method = 'trust-constr',
                bounds = self.bounds,
                constraints = self.constraints,
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

######## STORING ########
    @classmethod
    def from_dict(cls, data : Dict[str, Any])->'MultiPulseFLIMParams':
        """
        Reconstructs the MultiPulseFLIMParams from a dictionary
        """
        exps = [Exp.from_dict(exp) for exp in data['exps']]
        irf = MultiIrf.from_dict(data['irf'])
        color_channel = data.get('color_channel', None)
        noise = data.get('noise', None)
        name = data.get('name', None)
        return cls(
            *exps,
            irf,
            color_channel=color_channel,
            noise=noise,
            name=name
        )
    
    @classmethod
    def from_tuple(
        cls,
        param_tuple : Tuple,
        n_pulses : int,
        units : 'FlimUnitsLike' = FlimUnits.COUNTBINS,
        noise : float = 0.0,
        ):
        """
        Instantiate a MultiPulseFLIMParams object
        from a tuple of parameters. Because this could have
        any number of pulses, the values are not uniquely
        determined simply by the length of the param_tuple
        itself, and so the number of pulses must be given.

        Tuple form required:
        (tau, frac, tau, frac, ..., mean, sigma, irf_frac, mean, sigma, irf_frac...)

        Example code:
        -------------

        ```python
        # Instantiate a MultiPulseFLIMParams object
        # with 2 pulses, 2 exps, and 2 irfs
        params = MultiPulseFLIMParams.from_tuple(
            (1.0, 0.5, 2.0, 0.5, 100.0, 10.0, 0.5, 100.0, 10.0, 0.5),
            2,
            units = 'nanoseconds',
            noise = 0.1
        )
        ```
        """
        units = FlimUnits(units)
        n_exps = len(param_tuple) - len(FractionalIrf.class_params) * n_pulses
        exps = [
            Exp(*param_tuple[i:i+2], units=units)
            for i in range(0, 2*n_exps, 2)
        ]
        irfs = [
            FractionalIrf(*param_tuple[i:i+3], units=units)
            for i in range(2*n_exps, len(param_tuple), len(FractionalIrf.class_params))
        ]
        irf = MultiIrf(*irfs)

        return cls(
            *exps,
            irf,
            noise=noise
        )
    
    def __eq__(self, other):
        """
        Equality check
        """
        equal = False
        if isinstance(other, MultiPulseFLIMParams):
            equal = all(
                getattr(self, attr) == getattr(other, attr)
                for attr in self.class_params
            )

        return equal
    
    def __repr__(self):
        retstr = "MultiPulseFLIMParams object: \n\n"
        retstr += "\tParameters:\n"
        for exp in self.exps:
            retstr += "\t\t"+exp.__repr__() + "\n"
        retstr += f"\t\t{self.n_pulses} pulses\n"
        retstr += "\t\t"+self.irf.__repr__() + "\n"
        retstr += "\t\tNoise: " + str(self.noise) + "\n"
        retstr += "\t\tColor channel: " + str(self.color_channel) + "\n"
        return retstr

