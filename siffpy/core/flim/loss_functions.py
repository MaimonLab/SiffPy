from abc import abstractmethod, ABC

import numpy as np

from siffpy.core.flim.typing import PDF_Function, Objective_Function

class LossFunction(ABC):
    """
    Wrapper class to keep track of allowed /
    implemented loss functions
    """

    @staticmethod
    @abstractmethod
    def compute_loss(
        params : np.ndarray,
        data : np.ndarray,
        pdf : PDF_Function,
        x_range : np.ndarray
        )->float:
        """
        Returns the loss value of the model prediction.
        Must be implemented by subclasses.
        """
        ...

    @classmethod
    def from_data(
            cls,
            data : np.ndarray,
            pdf : PDF_Function,
            x_range : np.ndarray,
            **kwargs
        )->Objective_Function:
        """
        Returns a function that can be used in scipy.optimize.minimize,
        meaning it accepts just the parameter tuple and executes the `LossFunction`'s
        `compute_loss` method with the given data and pdf.

        Kwargs are passed through to `compute_loss`
        """
        return cls.compute_loss(data=data, pdf=pdf, x_range=x_range, **kwargs)

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
        ) -> float:
        """ Returns the chi-squared value of the model prediction vs the data """
        predicted = pdf(x_range, params)
        min_bin = 1 # eliminates zeros
        return np.sum(
            ((predicted[min_bin:] - data[min_bin:]) ** 2) 
            / predicted[min_bin:]
        )
    
class MSE(LossFunction):
    """
    Minimize the mean squared error
    """
    @staticmethod
    def compute_loss(
            params: np.ndarray,
            data: np.ndarray,
            pdf : PDF_Function,
            x_range : np.ndarray,
        ) -> float:
        """ Returns the mean squared error value of the model prediction vs the data """
        predicted = pdf(x_range,params)
        min_bin = 1
        return ((predicted[min_bin:] - data[min_bin:]) ** 2).sum()
