"""
Lookup tables for FLIM data to
the variable of interest.

"""
from dataclasses import dataclass
from typing import List, Dict

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
        return inverse_hill_equation(
            lifetime,
            self.n,
            self.k50,
            self.zero_point,
            self.max_point
        )

    def to_lifetime(self, concentration : float) -> float:
        """
        Hill equation to convert
        concentration to lifetime
        """
        return hill_equation(
            concentration,
            self.n,
            self.k50,
            self.zero_point,
            self.max_point
        )

class DangerousFit(FluorophoreHillFit):
    """
    A `FluorophoreHillFit` with at least one
    parameter not known from the literature.

    To be used with caution.


    """
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

class TemperatureSensitiveFit(FluorophoreHillFit):
    """
    A `FluorophoreHillFit` when the temperature-dependent
    changes in parameters are known.

    Initialized by a dict of temperatures whose values
    are the parameters for the fit at that temperature.

    If the temperature is out of the range of fits, uses
    the closest... or interpolates if it's between two
    known temperatures.

    e.g.

    ```
    temperature_dependent_params = {
        37 : dict(
            n = 1.63,
            k50 = 265,
            zero_point = 1.4,
            max_point = 2.78
        ),
        25 : dict(
            n = 1.33,
            k50 = 245,
            zero_point = 1.2,
            max_point = 2.5
        )
        ...
    }
    ```
    """

    def __init__(
            self,
            temperature_dependent_params : Dict[float, Dict[str, float]],
            *args,
            **kwargs
        ):
        # kwargs = {
        #     **kwargs,
        #     next(temperature_dependent_params.values()),
        # }
        self.temperature_dependent_params = temperature_dependent_params
        super().__init__(
            *args,
            n=next(iter(temperature_dependent_params.values()))['n'],
            k50=next(iter(temperature_dependent_params.values()))['k50'],
            zero_point=next(iter(temperature_dependent_params.values()))['zero_point'],
            max_point=next(iter(temperature_dependent_params.values()))['max_point'],
            **kwargs
        )

    def remap_by_temperature(self, temperature : float):
        """
        Remap the parameters according to the temperature.

        Written with CoPilot... haven't fact checked but it looks right.
        """

        if temperature in self.temperature_dependent_params:
            self.n = self.temperature_dependent_params[temperature]['n']
            self.k50 = self.temperature_dependent_params[temperature]['k50']
            self.zero_point = self.temperature_dependent_params[temperature]['zero_point']
            self.max_point = self.temperature_dependent_params[temperature]['max_point']
        else:
            # Interpolate
            temps = list(self.temperature_dependent_params.keys())
            temps.sort()

            if temperature < temps[0]:
                self.n = self.temperature_dependent_params[temps[0]]['n']
                self.k50 = self.temperature_dependent_params[temps[0]]['k50']
                self.zero_point = self.temperature_dependent_params[temps[0]]['zero_point']
                self.max_point = self.temperature_dependent_params[temps[0]]['max_point']

            elif temperature > temps[-1]:
                self.n = self.temperature_dependent_params[temps[-1]]['n']
                self.k50 = self.temperature_dependent_params[temps[-1]]['k50']
                self.zero_point = self.temperature_dependent_params[temps[-1]]['zero_point']
                self.max_point = self.temperature_dependent_params[temps[-1]]['max_point']

            else:
                for i in range(len(temps) - 1):
                    if temperature > temps[i] and temperature < temps[i + 1]:
                        break

                # Interpolate
                t1 = temps[i]
                t2 = temps[i + 1]

                n1 = self.temperature_dependent_params[t1]['n']
                k501 = self.temperature_dependent_params[t1]['k50']
                zero_point1 = self.temperature_dependent_params[t1]['zero_point']
                max_point1 = self.temperature_dependent_params[t1]['max_point']

                n2 = self.temperature_dependent_params[t2]['n']
                k502 = self.temperature_dependent_params[t2]['k50']
                zero_point2 = self.temperature_dependent_params[t2]['zero_point']
                max_point2 = self.temperature_dependent_params[t2]['max_point']

                self.n = n1 + (n2 - n1) * (temperature - t1) / (t2 - t1)
                self.k50 = k501 + (k502 - k501) * (temperature - t1) / (t2 - t1)
                self.zero_point = zero_point1 + (zero_point2 - zero_point1) * (temperature - t1) / (t2 - t1)
                self.max_point = max_point1 + (max_point2 - max_point1) * (temperature - t1) / (t2 - t1)


    def __call__(self, lifetime : float, temperature : float = 23.0):
        """
        Invert the hill equation to convert
        lifetime to concentration
        """
        self.remap_by_temperature(temperature)

        return inverse_hill_equation(
            lifetime,
            self.n,
            self.k50,
            self.zero_point,
            self.max_point
        )

    def to_lifetime(self, input : float, temperature : float = 23.0):
        """
        Hill equation to convert
        concentration to lifetime
        """
        self.remap_by_temperature(temperature)
        return hill_equation(
            input,
            self.n,
            self.k50,
            self.zero_point,
            self.max_point
        )

    def to_concentration(self, input : float, temperature : float = 23.0):
        """
        Hill equation to convert
        concentration to lifetime
        """
        self.remap_by_temperature(temperature)
        return inverse_hill_equation(
            input,
            self.n,
            self.k50,
            self.zero_point,
            self.max_point
        )

class IntensityHillFit(FluorophoreHillFit):
    """
    A non-FLIM-compatible Hill equation --
    for dF/F values
    """

    def __call__(self, lifetime : float) -> float:
        """
        Invert the hill equation to convert
        lifetime to concentration
        """
        return inverse_hill_equation(
            lifetime,
            self.n,
            self.k50,
            self.zero_point,
            self.max_point
        )

    def to_relative_intensity(self, concentration : float) -> float:
        """
        Hill equation to convert
        concentration to lifetime
        """
        return hill_equation(
            concentration,
            self.n,
            self.k50,
            self.zero_point,
            self.max_point
        )
    
# TqCaFLITS = FluorophoreHillFit(
#     n = 1.63,
#     k50 = 265,
#     zero_point = 1.4,
#     max_point = 2.78,
#     units_in = r'[Ca$2^+$] (nM)',
#     units_out = 'nanoseconds',
#     name = 'TqCaFLITS',
# )

TqCaFLITS = FluorophoreHillFit(
    n = 1.63,
    k50 = 265,
    zero_point = 1.72,
    max_point = 2.86,
    units_in = r'[Ca$2^+$] (nM)',
    units_out = 'nanoseconds',
    name = 'TqCaFLITS',
)

# TqCaFLITS = TemperatureSensitiveFit(
#     temperature_dependent_params= {
#         37: dict(
#             n = 1.63,
#             k50 = 265,
#             zero_point = 1.4,
#             max_point = 2.78,
#         ),
#         23: dict(
#             n = 1.63,
#             k50 = 265,
#             zero_point = 1.72,
#             max_point = 2.86,
#         ) # I SHOULD CHECK ABOUT THIS ONE
#     },
#     units_in = r'[Ca$2^+$] (nM)',
#     units_out = 'nanoseconds',
#     name = 'TqCaFLITS',
# )

GCaFLITS = TemperatureSensitiveFit(
    temperature_dependent_params= {
        37 : dict(
            n = 1.53,
            k50 = 220,
            zero_point = 2.9,
            max_point = 1.93
        ),
        23 : dict(
            n = 1.67,
            k50 = 356,
            zero_point = 3.52,
            max_point = 2.03
        )
    }
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

# GCaMP6s = IntensityHillFit(

# )

jGCaMP7f = IntensityHillFit(
    n = 3.10,
    k50 = 150,
    zero_point = 0.0,
    max_point = 31.0 ,
    units_in = r'[Ca$2^+$] (nM)',
    units_out = 'dF/F_0',
    name = 'jGCaMP7f',
)

jGCaMP8s = IntensityHillFit(
    n = 2.20,
    k50 = 46,
    zero_point = 0.0,
    max_point = 49.5,
    units_in = r'[Ca$2^+$] (nM)',
    units_out = 'dF/F_0',
    name = 'jGCaMP8s',
)

jGCaMP8m = IntensityHillFit(
    n = 1.92,
    k50 = 108,
    zero_point = 0.0,
    max_point = 47.5,
    units_in = r'[Ca$2^+$] (nM)',
    units_out = 'dF/F_0',
    name = 'jGCaMP8m',
)

jGCaMP8f = IntensityHillFit(
    n = 2.08,
    k50 = 334,
    zero_point = 0.0,
    max_point = 78.8,
    units_in = r'[Ca$2^+$] (nM)',
    units_out = 'dF/F_0',
    name = 'jGCaMP8f',
)
