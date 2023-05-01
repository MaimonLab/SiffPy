from abc import abstractmethod, ABC
from typing import Callable

import numpy as np

from siffpy.core.flim.flimparams import param_tuple_to_pdf

class LossFunction(ABC):
    """ Wrapper class to keep track of allowed / implemented loss functions """

    @classmethod
    @abstractmethod
    def compute_loss(cls, params : tuple, data : np.ndarray)->float:
        """ Returns the loss value of the model prediction """
        pass

    @classmethod
    def from_data(cls, data)->Callable[[tuple], float]:
        """ Reeturns a function that can be used in scipy.optimize.minimize """
        def loss(params):
            return cls.compute_loss(cls.params_untransform(params), data)
        return loss

    @classmethod 
    def from_data_frac(cls, curr_params, data)->Callable[[tuple], float]:
        """
        Returns a function that can be used in scipy.optimize.minimize
        but only uses the fraction parameters
        """

        def loss(frac_params):
            new_params = list(curr_params)
            new_params[1:-2:2] = frac_params

            return cls.compute_loss(tuple(new_params), data)
        return loss

    @classmethod
    def params_transform(cls, param_tuple : tuple)->tuple:
        """
        Transforms the parameters to improve solver behavior
        """
        return param_tuple

    @classmethod
    def params_untransform(cls, param_tuple : tuple)->tuple:
        """
        Untransforms the parameters to invert the transformation
        needed for good solver behavior
        """
        return param_tuple

    def __call__(self, params : tuple, data : np.ndarray)->float:
        return self.__class__.compute_loss(params, data)
    
class ChiSquared(LossFunction):

    @classmethod
    def params_transform(cls, param_tuple : tuple)->tuple:
        """
        Transforms the parameters to improve solver behavior.
        Divides the tau parameters by 100, divides the 
        tau_offset by 1000.
        """
        ret_list = [
            param / 100 if i % 2 == 0 else param
            for i, param in enumerate(param_tuple)
        ]
        ret_list[-2] /= 10
        return tuple(ret_list)

    @classmethod
    def params_untransform(cls, transformed_tuple : tuple)->tuple:
        """
        Untransforms the parameters to invert the transformation
        needed for good solver behavior
        """
        ret_list = [
            param * 100 if i % 2 == 0 else param
            for i, param in enumerate(transformed_tuple)
        ]
        ret_list[-2] *= 10
        return tuple(ret_list)

    @classmethod
    def compute_loss(
            cls,
            params: tuple,
            data: np.ndarray,
            exclude_wraparound: bool = True
        ) -> float:
        """ Returns the chi-squared value of the model prediction vs the data """
        predicted = np.sum(data)*param_tuple_to_pdf(np.arange(data.size), params)
        min_bin = int(params[-2] - params[-1]) if exclude_wraparound else 0 # one sigma earlier than mean only
        return np.sum(((predicted[min_bin:] - data[min_bin:]) ** 2) / predicted[min_bin:])
    
class MSE(LossFunction):

    @classmethod
    def compute_loss(
            cls,
            params: tuple,
            data: np.ndarray,
            exclude_wraparound: bool = True
        ) -> float:
        """ Returns the mean squared error value of the model prediction vs the data """
        predicted = np.sum(data)*param_tuple_to_pdf(np.arange(data.size),params)
        min_bin = int(params[-2] - params[-1]) if exclude_wraparound else 0 # one sigma earlier than mean only
        return np.sqrt(((predicted[min_bin:] - data[min_bin:]) ** 2).sum())
