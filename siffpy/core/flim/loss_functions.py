from abc import abstractmethod, ABC

import numpy as np

from siffpy.core.flim.typing import PDF_Function, Objective_Function

class LossFunction(ABC):
    """ Wrapper class to keep track of allowed / implemented loss functions """

    #params_transform 

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def compute_loss(cls, params : np.ndarray, data : np.ndarray, pdf : PDF_Function)->float:
        """ Returns the loss value of the model prediction """
        pass

    @classmethod
    def from_data(cls, data : np.ndarray, pdf : PDF_Function)->Objective_Function:
        """
        Returns a function that can be used in scipy.optimize.minimize,
        meaning it accepts just the parameter tuple and executes the `LossFunction`'s
        `compute_loss` method with the given data and pdf
        """
        def loss(params):
            return cls.compute_loss(cls.params_untransform(params), data, pdf)
        return loss

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
    def params_untransform(cls, param_tuple : np.ndarray)->np.ndarray:
        """
        Untransforms the parameters to invert the transformation
        needed for good solver behavior
        """
        return param_tuple

    def __call__(self, params : np.ndarray, data : np.ndarray)->float:
        return self.__class__.compute_loss(params, data)
    
class ChiSquared(LossFunction):

    @classmethod
    def params_transform(cls, param_array : np.ndarray)->np.ndarray:
        """
        Transforms the parameters to improve solver behavior.
        Divides the tau parameters by 100, divides the 
        tau_offset by 100.
        """
        transformed = param_array.copy()
        transformed[:-2:2] = transformed[:-2:2] / 100
        transformed[-2] = transformed[-2] / 10
        return transformed

    @classmethod
    def params_untransform(cls, transformed_array : np.ndarray)->np.ndarray:
        """
        Untransforms the parameters to invert the transformation
        needed for good solver behavior
        """
        untransformed = transformed_array.copy()
        untransformed[:-2:2] = untransformed[:-2:2] * 100
        untransformed[-2] = untransformed[-2] * 10
        return untransformed

    @classmethod
    def compute_loss(
            cls,
            params: np.ndarray,
            data: np.ndarray,
            pdf : PDF_Function,
            exclude_wraparound: bool = True
        ) -> float:
        """ Returns the chi-squared value of the model prediction vs the data """
        predicted = np.sum(data)*pdf(np.arange(data.size)+0.5, params)
        if exclude_wraparound:
            min_bin = int(params[-2] - params[-1])
            return np.sum(((predicted[min_bin:] - data[min_bin:]) ** 2) / predicted[min_bin:])
        return np.sum(((predicted - data) ** 2) / predicted)
    
class MSE(LossFunction):

    @classmethod
    def compute_loss(
            cls,
            params: np.ndarray,
            data: np.ndarray,
            pdf : PDF_Function,
            exclude_wraparound: bool = True
        ) -> float:
        """ Returns the mean squared error value of the model prediction vs the data """
        predicted = np.sum(data)*pdf(np.arange(data.size)+0.5,params)
        if exclude_wraparound:
            min_bin = int(params[-2] - params[-1])
            return np.sqrt(((predicted[min_bin:] - data[min_bin:]) ** 2).sum())
        return np.sqrt(((predicted - data) ** 2).sum())
