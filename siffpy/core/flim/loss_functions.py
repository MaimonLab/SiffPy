from abc import abstractmethod, ABC
from functools import partial

import numpy as np

from siffpy.core.flim.typing import PDF_Function, Objective_Function

class LossFunction(ABC):
    """ Wrapper class to keep track of allowed / implemented loss functions """

    #params_transform 

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def compute_loss(
        cls,
        params : np.ndarray,
        data : np.ndarray,
        pdf : PDF_Function,
        x_range : np.ndarray
        )->float:
        """ Returns the loss value of the model prediction """
        pass

    @classmethod
    def from_data(cls, data : np.ndarray, pdf : PDF_Function, x_range : np.ndarray, **kwargs)->Objective_Function:
        """
        Returns a function that can be used in scipy.optimize.minimize,
        meaning it accepts just the parameter tuple and executes the `LossFunction`'s
        `compute_loss` method with the given data and pdf.

        Kwargs are passed through to `compute_loss`
        """
        # def loss(params):
        #     return cls.compute_loss(cls.params_untransform(params), data, pdf)
        return partial(
            cls.compute_loss,
            data = data,
            pdf = pdf,
            x_range = x_range,
            **kwargs
        )

    @classmethod 
    def from_data_frac(cls, curr_params, data, pdf : PDF_Function)->Objective_Function:
        """
        Returns a function that can be used in scipy.optimize.minimize
        but only uses the fraction parameters (i.e. the fraction of photons
        belonging to each exponential).
        """

        def loss(frac_params):
            new_params = list(curr_params)
            #new_params[1:-2*self.n_pulses:2] = frac_params
            new_params[1:-2:2] = frac_params

            return cls.compute_loss(new_params, data, pdf)
        return loss

    @classmethod
    def params_transform(cls, param_tuple : np.ndarray)->np.ndarray:
        """
        Transforms the parameters to improve solver behavior
        """
        return param_tuple
    
    @classmethod
    def noise_transform(cls, noise : np.ndarray)->np.ndarray:
        """
        Transforms the noise to improve solver behavior
        """
        return noise

    @classmethod
    def params_untransform(cls, param_tuple : np.ndarray)->np.ndarray:
        """
        Untransforms the parameters to invert the transformation
        needed for good solver behavior
        """
        return param_tuple

    @classmethod
    def noise_untransform(cls, noise : np.ndarray)->np.ndarray:
        """
        Untransforms the noise to invert the transformation
        needed for good solver behavior
        """
        return noise

    def __call__(self, params : np.ndarray, data : np.ndarray)->float:
        return self.__class__.compute_loss(params, data)
    
class ChiSquared(LossFunction):
    """
    Slower, probably more accurate in the end??
    Considered the gold standard because it places
    equal emphasis on all data points... but in practice
    for some reason seems to behave worse than MSE
    """
    @staticmethod
    def compute_loss(
            params: np.ndarray,
            data: np.ndarray,
            pdf : PDF_Function,
            x_range : np.ndarray,
            #exclude_wraparound: bool = False
        ) -> float:
        """ Returns the chi-squared value of the model prediction vs the data """
        predicted = pdf(x_range, params)
        min_bin = 1 # eliminates zeros
        # if exclude_wraparound:
        #     min_bin = int(params[-2] - params[-1])
        return np.sum(
            ((predicted[min_bin:] - data[min_bin:]) ** 2) 
            / predicted[min_bin:]
        )
    
class MSE(LossFunction):
    """
    Minimize the mean squared error
    """
    @classmethod
    def compute_loss(
            cls,
            params: np.ndarray,
            data: np.ndarray,
            pdf : PDF_Function,
            x_range : np.ndarray,
            exclude_wraparound: bool = False
        ) -> float:
        """ Returns the mean squared error value of the model prediction vs the data """
        predicted = pdf(x_range,params)
        min_bin = 1
        if exclude_wraparound:
            min_bin = int(params[-2] - params[-1])
        return ((predicted[min_bin:] - data[min_bin:]) ** 2).sum()
