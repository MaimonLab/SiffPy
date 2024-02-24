"""
Lookup tables for FLIM data to
the variable of interest.

"""
from dataclasses import dataclass
from typing import List
from functools import partial

def hill_equation(
    x : float,
    n : float,
    k50 : float,
    zero_point : float = 0.0,
    max_point : float = 1.0
) -> float:
    """
    Hill equation for converting
    a concentration to a FLIM value
    """
    return zero_point + (max_point - zero_point)/(1 + (k50/x)**n)

def inverse_hill_equation(
    y : float,
    n : float,
    k50 : float,
    zero_point : float = 0.0,
    max_point : float = 1.0
) -> float:
    """
    Inverse Hill equation for converting
    a FLIM value to a concentration
    """
    return k50 / ( ( (max_point - zero_point) / (y - zero_point) ) - 1 )**(1/n)

@dataclass
class FluorophoreHillFit:
    """
    Hill equation for a fluorophore.

    When called, the equation will convert
    a lifetime to a concentration.

    When called with `to_lifetime`, the
    equation will convert a concentration
    to a lifetime.

    n : float
        Hill coefficient. Higher n is less linear

    k50 : float
        Half-maximal concentration

    zero_point : float
        Value of the lifetime when the input is 0

    max_point : float
        Value of the lifetime when the input is infinite

    units_in : str
        Units of the input variable

    units_out : str
        Units of the output variable

    

    TODO: more foolproof unit stuff.
    """
    n : float
    k50 : float
    zero_point : float
    max_point : float
    units_in : str = 'Undefined'
    units_out : str = 'Undefined'
    name : str = 'Unnamed'

    def __call__(self, lifetime : float) -> float:
        """
        Invert the hill equation to convert
        lifetime to concentration
        """
        return inverse_hill_equation(lifetime, self.n, self.k50, self.zero_point, self.max_point)

    def to_concentration(self, input : float) -> float:
        """
        Hill equation to convert
        concentration to lifetime
        """
        return hill_equation(input, self.n, self.k50, self.zero_point, self.max_point)

class DangerousFit(FluorophoreHillFit):
    undefined_parameters : List[str]

    def __init__(self, *args, undefined_parameters : List[str] = [], **kwargs):
        super().__init__(*args, **kwargs)
        self.undefined_parameters = undefined_parameters

    def __call__(self, *args, **kwargs):
        warnstr = (f'WARNING: Fit {self.name} as currently implemented '
            + 'has at least one parameter not defined in the literature'
            + ' (or not known to the authors of this code at present).'
        )
       
        if len(self.undefined_parameters) > 0:
            warnstr += f' Undefined parameters: {self.undefined_parameters}'
        
        print(warnstr) 
        return super().__call__(*args, **kwargs)

    def to_concentration(self, *args, **kwargs):
        warnstr = (f'WARNING: Fit {self.name} as currently implemented '
            + 'has at least one parameter not defined in the literature'
            + ' (or not known to the authors of this code at present).'
        )

        if len(self.undefined_parameters) > 0:
            warnstr += f' Undefined parameters: {self.undefined_parameters}'
        print(warnstr) 
        return super().to_concentration(*args, **kwargs)


TqCaFLITS = FluorophoreHillFit(
    n = 1.63,
    k50 = 265,
    zero_point = 1.4,
    max_point = 2.78,
    units_in = r'[Ca$2^+$] (nM)',
    units_out = 'nanoseconds',
    name = 'TqCaFLITS',
)

jRCaMP1b = DangerousFit(
        n = 1.63,
        k50 = 712,
        zero_point = 1.6,
        max_point = 2.78,
        units_in = r'[Ca$2^+$] (nM)',
        units_out = 'nanoseconds',
        name = 'jRCaMP1b',
        undefined_parameters = ['n']
    )

# WARNING FOR UNDEFINED JRCAMP1B