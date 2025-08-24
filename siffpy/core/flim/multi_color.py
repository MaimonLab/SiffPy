"""
Code for the whole shebang -- multi-channel, multi-pulse, multi-fluorophore
putting it all together to figure out as much as possible from your arrival
time data.
"""
from typing import List, Union, Tuple, Optional, Callable

import numpy as np

from siffpy.core.flim.flimparams import (multi_exponential_pdf_from_params,
    FLIMParams, FLIMParameter, FlimUnits, FlimUnitsLike, Bounds, LinearConstraint)
from siffpy.core.flim.multi_pulse import MultiPulseFLIMParams, MultiIrf

def multi_channel_multi_pulse_multi_fluorophore_objective(
    params : np.ndarray,
    tau_axis : np.ndarray,
    data : np.ndarray,
    n_pulses : int,
    n_fluorophores : int,
    n_states_by_fluorophore : list[int]
    ):
    """
    Data is of form n_channels x arrival_time_Bins. Expected to already be divided by
    the entire sum.

    TODO: tidy this up and make it execute more quickly!!

    ## Arguments

    - `params` : np.ndarray
        The parameters determining the distribution of arrival times in the same
        units as `tau_axis`. The parameters are expected to be in the following order:

    ```
    [N_f1, N_f2, N_f3, ..., N_f, (fraction of signal photons for each fluorophore)
    
    chi_f1c1, chi_f1c2, chi_f1cn, chi_f2c1, ... chi_NfNc, (channel weights for each fluorophore)
    
    tau_f1i1, tau_f1i2, ..., tau_f1iN1, (exponential decay times for each state of each fluorophore)
    
    tau_f2i1, ...
    
    frac_f1i1, frac_f1i2, ..., frac_f1iN1, (fraction of signal photons for each state of each fluorophore)
    
    frac_f2i1, ...
    
    phi_f1l1, phi_f1l2, ... phi_f1Nl, (laser pulse relative weight for each pulse for each fluorophore)
    
    phi_f2l1, ... phi_f2Nl, ...
    
    mu_l1, mu_l2, ... , mu_l1, ..., mu_Nl, (mean time of each pulse)
    
    sigma_l1, sigma_l2, ... sigma_Nl, (IRF breadth for each pulse)

    delta_mu_c2, ... delta_mu_Nc, (note one fewer than num channels) (time offset for each sensor channel)
    
    noise_c1, noise_c2, ... noise_cn] (noise for each channel)
    ```

    - `tau_axis` : np.ndarray
        The array of arrival time bins in the same units as the parameters

    - `data` : np.ndarray
        Arrival time histograms of the shape (n_channels = `Nc`, len(tau_axis)). This should
        be normalized to the _total number of photons in the data_: i.e. it should be
        `data /= data.sum()`. This is not done in the function so that the operation
        is not repeated on every optimization iteration.

    - `n_pulses` : int
       The number of laser pulses in the data (determines `Nl` above)

    -  `n_fluorophores` : int
        The number of fluorophores in the data (determines `Nf` above)

    - `n_states_by_fluorophore` : list[int]
        A list of the number of states for each fluorophore (determines `Ni` for each `f` above)

    ## Returns

    - `objective` : float
        The chi-squared statistic of the data 
        given the parameters.
    """
    n_channels = data.shape[0]
    param_idx = 0

    noise = params[-n_channels:]
    param_idx += n_channels

    delta_mus = params[-(param_idx+n_channels-1):-param_idx]
    param_idx += n_channels - 1
    delta_mus = np.append(0, delta_mus)

    sigmas = params[-(param_idx+n_pulses):-param_idx]
    param_idx += n_pulses

    mus = params[-(param_idx+n_pulses):-param_idx]
    param_idx += n_pulses

    phis = params[-(param_idx+(n_pulses*n_fluorophores)):-param_idx]
    param_idx += n_pulses*n_fluorophores

    num_total_states = int(np.sum(n_states_by_fluorophore))
    fracs = params[-(param_idx + num_total_states):-param_idx]
    param_idx += num_total_states

    taus = params[-(param_idx + num_total_states):-param_idx]
    param_idx += num_total_states

    chis = params[-(param_idx + n_channels*n_fluorophores):-param_idx]
    param_idx += n_channels*n_fluorophores

    Ns = params[-(param_idx + n_fluorophores):-param_idx]
    param_idx += n_fluorophores
    
    # channelwise noise estimates
    noise_array = (
        np.expand_dims(noise, -1)
        @ np.expand_dims(np.ones_like(tau_axis), axis=0)
        /len(tau_axis)
    )

    # of len n_fluorophores, list of lists of (tau, frac) tuples
    fluorophore_params = []
    p_idx = 0
    for f_idx in range(n_fluorophores):
        fluorophore_params.append(
            [
                (tau, frac)
                for tau, frac in zip(
                    taus[p_idx:p_idx+n_states_by_fluorophore[f_idx]],
                    fracs[p_idx:p_idx+n_states_by_fluorophore[f_idx]]
                )
            ]
        )
        p_idx += n_states_by_fluorophore[f_idx]

    expected_data = np.zeros_like(data)
    for channel in range(n_channels):
        for pulse in range(n_pulses):
            for fluorophore in range(n_fluorophores):
                expected_data[channel] += Ns[fluorophore] * chis[fluorophore*n_channels + channel] * (
                    phis[fluorophore*n_pulses + pulse]*multi_exponential_pdf_from_params(
                        tau_axis - delta_mus[channel],
                        np.append(fluorophore_params[fluorophore], [mus[pulse], sigmas[pulse]])
                    )
                )

    expected_data = (expected_data + noise_array[:,np.newaxis]).astype(float)
    # return expected_data
    return np.nansum(((expected_data - data)/expected_data)**2)

class NPhotons(FLIMParameter):
    """
    Fraction of total number of photons in the data contributed by
    a fluorophore and their relative .

    Initialized like `NPhotons(n = 0.5, channel_fractions = [0.5, 0.5])`

    ## Attributes
    - `n` : float
        The fraction of total data photons contributed by the fluorophore
        (sum of `NPhotons.n` for all fluorophores should be 1)

    - `channel_fractions` : List[float]
        The fraction of the total signal photons from this fluorophore in
        each channel.

    ## Aliases

    - `n` : ['n_photons']
    - `channel_fractions` : ['channels', 'fracs', 'chis']
    """
    class_params = ['n', 'channel_fractions']
    aliases = {
        'n' : ['n_photons'],
        'channel_fractions' : ['channels', 'fracs', 'chis']
    }

class DeltaMu(FLIMParameter):
    """
    Each channel's time offset from the first channel.

    Initialized with `DeltaMu(delta_mu = 0)`
    """
    class_params = ['delta_mu']
    unitful_params = ['delta_mu']
    aliases = {
        'delta_mu' : ['delta', 'mu', 'mean', 'offset']
    }

class MPMFMCFlimParams(FLIMParams):
    """
    Multi-channel, multi-pulse, multi-fluorophore FLIM parameters object.

    TODO: Unify all of these into one interface that can be used flexibly
    and generically... i.e. `FlimParams` should handle multi-channel, multi-pulse,
    multi-fluorophore, and multi-state and just take args to figure out which
    type of model to use....
    """

    def __init__(
            self,
            *args : Union[Tuple[Union['MultiPulseFLIMParams', FLIMParameter]], Tuple[float]],
            chis : Optional[List[List[float]]] = None,
            n_colors : int = 1,
            n_fluorophores : int = 1,
            n_exps_by_fluorophore : List[int] = [2],
            n_irfs : int = 1,
            **kwargs,
        ):
        """
        Create a `MPMFMCFlimParams` object. If no `MultiPulseFLIMParams` objects are
        passed, a default set of `MultiPulseFLIMParams` objects will be created using
        the `n_fluorophores`, `n_exps_by_fluorophore`, and `n_irfs` arguments.
        """
        mpfps = [mpfp for mpfp in args if isinstance(mpfp, MultiPulseFLIMParams)]
        if len(mpfps) > 1:
            if not all(np.allclose(mpfp.irf.tau_offsets, mpfps[0].irf.tau_offsets, rtol = 1e-3) for mpfp in mpfps):
                raise ValueError("All fluorophores must have the same IRFs from the same data.")

        if len(mpfps) == 0:
            # avoid circular import
            from siffpy.core.flim import default_flimparams

            self._fluorophores = [
                default_flimparams(n_irfs = n_irfs, n_exps = exp_n)
                for _, exp_n in zip(range(n_fluorophores), n_exps_by_fluorophore)
            ]
        else:
            for fp in mpfps:
                fp.noise = 0.0 # noise is a channel-wide property in this context
            self._fluorophores = mpfps
            n_fluorophores = len(mpfps)

        self.n_fluorophores = n_fluorophores
        self.n_colors = n_colors
        if chis is not None:
            if not all([len(chi) == n_colors for chi in chis]):
                raise ValueError("All fluorophores must have the same number of channels.")
            self._n_photons = [NPhotons(n = 1/n_fluorophores, channel_fractions = chi) for n, chi in zip(range(n_fluorophores), chis)]
        else:
            self._n_photons = [NPhotons(n = 1/n_fluorophores, channel_fractions = [1/n_colors] * n_colors) for _ in range(n_fluorophores)]
        self._delta_mus = [DeltaMu(delta_mu = 0) for _ in range(n_colors)]
        self._noise = [0.01 for _ in range(n_colors)]

    @property
    def n_photons(self) -> List[NPhotons]:
        """
        One `NPhotons` object for each fluorophore
        """
        return self._n_photons

    @property
    def offsets(self) -> List[Tuple[float,float]]:
        """
        Returns each laser pulse's time offset and fractional excitation
        """
        raise NotImplementedError()
    
    @property
    def irfs(self)->MultiIrf:
        return self._fluorophores[0].irfs
    
    @property
    def fluorophores(self)->List[MultiPulseFLIMParams]:
        return self._fluorophores
    
    @fluorophores.setter
    def fluorophores(self, fluorophores : List[MultiPulseFLIMParams]):
        if not (isinstance(fluorophores, list) and all([isinstance(f, MultiPulseFLIMParams) for f in fluorophores])):
            raise TypeError("Fluorophores must be a list of `MultiPulseFLIMParams` objects.")
        
        if any (fluorophores[0].irfs != fluorophore.irfs for fluorophore in fluorophores):
            raise ValueError("All fluorophores must have the same `MultiIrf`.")
        
        self._fluorophores = fluorophores

    @property
    def n_pulses(self) -> int:
        """
        Number of laser pulses
        """
        return self.irfs.n_irfs

    @property
    def n_params(self) -> int:
        """
        Number of parameters in the model
        """
        return len(self.param_tuple)
    
    @property
    def params(self) -> List['FLIMParameter']:
        """
        Params are returned of the form:
        ```
        [
            NPhotons ... (length n_fluorophores),
            each MultiPulseFLIMParams's `Exp` parameters,
            each of `MultiIrf` parameters,
            each DeltaMu (n_channels - 1),
        ]
        ```

        ## Example

        TODO
        """
        ret_list = []
        ret_list += self._n_photons
        for fluorophore in self._fluorophores:
            ret_list += fluorophore.exps

        ret_list += [self.irfs]
        ret_list += self._delta_mus
        return ret_list
    
    @property
    def allow_noise(self) -> bool:
        """
        Whether to allow noise in the model
        """
        return True
    
    @property
    def param_tuple(self) -> Tuple:
        """
        
        Params are returned of the form:
        
        ```
        (
            
            N_f1, N_f2, N_f3, ..., N_f, # (fraction of signal photons for each fluorophore)
            chi_f1c1, chi_f1c2, chi_f1cn, chi_f2c1, ... chi_NfNc, # (channel weights for each fluorophore)
            tau_f1i1, tau_f1i2, ..., tau_f1iN1, # (exponential decay times for each state of each fluorophore)
            tau_f2i1, ...
            frac_f1i1, frac_f1i2, ..., frac_f1iN1, # (fraction of signal photons for each state of each fluorophore)
            frac_f2i1, ...
            phi_f1l1, phi_f1l2, ... phi_f1Nl, # (laser pulse relative weight for each pulse for each fluorophore)
            phi_f2l1, ... phi_f2Nl, ...
            mu_l1, mu_l2, ... , mu_l1, ..., mu_Nl, # (mean time of each pulse)
            sigma_l1, sigma_l2, ... sigma_Nl, # (IRF breadth for each pulse)
            delta_mu_c2, ... delta_mu_Nc, # (note one fewer than num channels) (time offset for each sensor channel)
        )
        ```
        
        """

        ret_list = []

        ret_list += [nph.n for nph in self._n_photons]
        
        ret_list += [frac for nph in self._n_photons for frac in nph.channel_fractions]
        
        ret_list += [exp.tau for fluorophore in self.fluorophores for exp in fluorophore.exps]

        ret_list += [exp.frac for fluorophore in self._fluorophores for exp in fluorophore.exps] 

        ret_list += [irf.frac for fluorophore in self._fluorophores for irf in fluorophore.irfs]

        ret_list += [irf.mu for irf in self._fluorophores[0].irfs]

        ret_list += [irf.sigma for irf in self._fluorophores[0].irfs]
        
        if len(self._delta_mus) > 1:
            ret_list += [dm.delta_mu for dm in self._delta_mus[1:]]
        # ret_list += [n for n in self._noise]

        return tuple(ret_list)
    
    @param_tuple.setter
    def param_tuple(self, new_params : Tuple):
        """
        Set the parameters of the model using a tuple of the form
        returned by `param_tuple`
        """

        curr_idx = 0
        Ns = new_params[curr_idx : curr_idx+self.n_fluorophores]
        curr_idx += self.n_fluorophores

        chis = []
        for _ in range(self.n_fluorophores):
            chis += [new_params[curr_idx : curr_idx + self.n_colors]]
            curr_idx += self.n_colors

        taus = []
        for fluorophore in self._fluorophores:
            taus += [new_params[curr_idx : curr_idx + len(fluorophore.exps)]]
            curr_idx += len(fluorophore.exps)
        
        fracs = []
        for fluorophore in self._fluorophores:
            fracs += [new_params[curr_idx : curr_idx + len(fluorophore.exps)]]
            curr_idx += len(fluorophore.exps)

        phis = []
        for fluorophore in self._fluorophores:
            phis += [new_params[curr_idx : curr_idx + self.n_pulses]]
            curr_idx += self.n_pulses

        mus = new_params[curr_idx : curr_idx + self.n_pulses]
        curr_idx += self.n_pulses

        sigmas = new_params[curr_idx : curr_idx + self.n_pulses]
        curr_idx += self.n_pulses

        delta_mus = new_params[curr_idx : curr_idx + self.n_colors - 1]
        curr_idx += self.n_colors - 1

        # noise = new_params[curr_idx : curr_idx + self.n_colors]
        # curr_idx += self.n_colors

        self._n_photons = [NPhotons(n = n, channel_fractions = chi) for n, chi in zip(Ns, chis)]
        for fluorophore, tau_f, frac_f in zip(self._fluorophores, taus, fracs):
            for exp, tau, frac in zip(fluorophore.exps, tau_f, frac_f):
                exp.tau = tau
                exp.frac = frac

        for fluorophore, phi_f in zip(self._fluorophores, phis):
            for irf, phi, mu, sigma in zip(fluorophore.irfs, phi_f, mus, sigmas):
                irf.frac = phi
                irf.mu = mu
                irf.sigma = sigma

        for delta_mu, dm in zip(delta_mus, self._delta_mus[1:]):
            dm.delta_mu = delta_mu
    
    @property
    def units(self) -> FlimUnits:
        if not all([f.units == self._fluorophores[0].units for f in self._fluorophores]):
            raise ValueError("All fluorophores must have the same units.")
        return self._fluorophores[0].units
    
    @units.setter 
    def units(self, new_units : FlimUnitsLike):
        for fluorophore in self._fluorophores:
            fluorophore.units = new_units

    @property
    def noise(self) -> Tuple[float]:
        return self._noise
    
    @noise.setter
    def noise(self, new_noise : Tuple[float]):
        if len(new_noise) != self.n_colors:
            raise ValueError("Noise must be a tuple of length `n_colors`.")
        self._noise = new_noise
    
    def ncomponents(self) -> int:
        """
        Number of components in the model
        """
        raise AttributeError(" `ncomponents` is not a valid attribute for this object.")

    ###### PDF #####
     
    def pdf(self, x_range : np.ndarray) -> np.ndarray:
        """
        Returns the PDF of the model over the given x values. Output
        is of shape (n_channels, len(x_range)). Each color channel is
        normalized to sum to 1. If you want the color channels to be
        normalized proportional to each other, i.e. the channel with
        more photons to contain a higher proportion of the total signal,
        use `cross_color_pdf`.

        ## Arguments

        - `x_range` : np.ndarray
            The range of x values to evaluate the PDF over

        ## Returns

        - `pdf` : np.ndarray
            The PDF of the model over the given x values. Output
            is of shape (n_channels, len(x_range)). Each color channel is
            normalized to sum to 1.

        ## Example

        ```
        # flim_params = MPMFMCFlimParams( * args )

        x_range = np.linspace(0, 12.5, 1000)

        pdf = flim_params.pdf(x_range)

        print(pdf.sum(axis = 1)) # should be 1

        >>> [1., 1.]

        ## See also

        - `MPMFMCFlimParams.probability_dist`
        """
        return self.probability_dist(x_range)
    
    def probability_dist(self, x_range : np.ndarray) -> np.ndarray:
        """
        Returns the PDF of the model over the given x values. Output
        is of shape (n_channels, len(x_range)). Each color channel is
        normalized to sum to 1. If you want the color channels to be
        normalized proportional to each other, i.e. the channel with
        more photons to contain a higher proportion of the total signal,
        use `cross_color_pdf`.

        ## Arguments

        - `x_range` : np.ndarray
            The range of x values to evaluate the PDF over

        ## Returns

        - `pdf` : np.ndarray
            The PDF of the model over the given x values. Output
            is of shape (n_channels, len(x_range)). Each color channel is
            normalized to sum to 1.

        ## Example

        ```
        # flim_params = MPMFMCFlimParams( * args )

        x_range = np.linspace(0, 12.5, 1000)

        pdf = flim_params.probability_dist(x_range)

        print(pdf.sum(axis = 1)) # should be 1

        >>> [1., 1.]
        ```
        """

        synth_data = np.zeros((self.n_colors, len(x_range)))
        for color, dm in enumerate(self._delta_mus):
            for f, n_f in zip(self.fluorophores, self.n_photons):
                synth_data[color] += n_f.n * n_f.channel_fractions[color]*f.pdf(x_range - dm.delta_mu)

        # add the noise
        for color in range(self.n_colors):
            synth_data[color] *= (1-self.noise[color])
            synth_data[color] += self.noise[color]/len(x_range)
            synth_data[color] /= np.sum(synth_data[color])

        return synth_data
    
    def cross_color_pdf(self, x_range : np.ndarray) -> np.ndarray:
        """
        Returns the PDF of the model across fluorophores over the given x values. Output
        is of shape (n_channels, len(x_range)). The whole array is normalized to sum
        to 1. If you want the color channels to be
        normalized independently of each other, i.e. the channel with
        more photons to contain a higher proportion of the total signal,
        use `pdf`.

        ## Arguments

        - `x_range` : np.ndarray
            The range of x values to evaluate the PDF over

        ## Returns

        - `pdf` : np.ndarray
            The PDF of the model over the given x values. Output
            is of shape (n_channels, len(x_range)). The entire array
            is normalized to sum to 1.

        ## Example

        ```
        # flim_params = MPMFMCFlimParams( * args )

        x_range = np.linspace(0, 12.5, 1000)

        pdf = flim_params.cross_color_pdf(x_range)

        print(pdf.sum(axis = 1))

        >>> [0.60862262 0.39137738]

        print(pdf.sum())

        >>> 1.0
        ```
        """
        synth_data = np.zeros((self.n_colors, len(x_range)))
        for color, dm in enumerate(self._delta_mus):
            for f, n_f in zip(self.fluorophores, self.n_photons):
                synth_data[color] += n_f.n * n_f.channel_fractions[color]*f.pdf(x_range - dm.delta_mu)

        # add the noise
        for color in range(self.n_colors):
            synth_data[color] *= (1-self.noise[color])
            synth_data[color] += self.noise[color]/len(x_range)

        synth_data /= np.sum(synth_data)

        return synth_data

    ###### FITTING #########

    @property
    def bounds(self) -> Bounds:
        """
        Returns the bounds for the parameters
        """
        lb, ub = [], []
        # N
        for n_photon in self._n_photons:
            lb += [0]
            ub += [1]

        for n_photon in self._n_photons:
            for c_frac in n_photon.channel_fractions:
                lb += [0]
                ub += [1]

        for fluorophore in self._fluorophores:
            # tau
            for exp in fluorophore.exps:
                lb += [0]
                ub += [np.inf]

        for fluorophore in self._fluorophores:
            # frac
            for exp in fluorophore.exps:
                lb += [0]
                ub += [1]

        for fluorophore in self._fluorophores:
            # phi
            for irf in fluorophore.irfs:
                lb += [0]
                ub += [1]

        # mu
        for irf in self._fluorophores[0].irfs:
            lb += [0]
            ub += [np.inf]

        # sigma
        for irf in self._fluorophores[0].irfs:
            lb += [0]
            ub += [np.inf]

        # delta_mu
        for delta_mu in self._delta_mus[1:]:
            lb += [-np.inf]
            ub += [np.inf]

        # noise
        for noise in self._noise:
            lb += [0]
            ub += [1]

        return Bounds(lb = lb, ub = ub)
    
    @property
    def constraints(self) -> List[LinearConstraint]:
        """
        Returns the constraints for the parameters
        """

        n_params = self.n_params + len(self.noise)
        # Sum of Ns = 1
        consts = []
        consts += [LinearConstraint(
            A = [1] * len(self._n_photons) + [0]*(n_params - len(self._n_photons)),
            lb = 1,
            ub = 1
        )]

        used_params = len(self._n_photons)

        # sum of chis < 1 per channel
        for f in self.fluorophores:
            consts += [
                LinearConstraint(
                    A = ([0] * used_params) + ([1] * self.n_colors) + ([0]*(n_params - used_params - self.n_colors)),
                    lb = 0,
                    ub = 1
                )
            ]
            used_params += self.n_colors

        # for each fluorophore, the taus must be in order
        tau_consts = []
        for f in self._fluorophores:
            if f.n_exp == 1:
                continue
            tau_consts = [
                LinearConstraint(
                   A = (
                       ([0] * (used_params + (exp_num - 1)))
                        + [-1, 1]
                        + ([0] * (n_params - used_params - (exp_num+1)))
                   ),
                   lb = 0, ub = np.inf
                )
                for exp_num in range(1, f.n_exp)
            ]
            used_params += f.n_exp
        consts += tau_consts

        # for each fluorophore, the fractions sum to 1
        for f in self._fluorophores:
            if f.n_exp == 1:
                continue
            consts += [
                LinearConstraint(
                   A = (
                       ([0] * used_params)
                        + [1]*f.n_exp
                        + ([0] * (n_params - used_params - (f.n_exp)))
                   ),
                   lb = 1, ub = 1
                )
            ]
            used_params += f.n_exp

        # For each laser pulse and for each flurophore
        # the phis must sum to 1
        for f in self._fluorophores:
            consts += [
                LinearConstraint(
                    A = (
                        ([0] * (used_params))
                        + [1] * self.n_pulses
                        + ([0] * (n_params - used_params - self.n_pulses))
                    ),
                    lb = 1, ub = 1
                )
            ]
            used_params += self.n_pulses
        
        # for each laser pulse, the mus must be in order
        consts += [
            LinearConstraint(
                A = (
                    ([0] * (used_params + (pulse_num - 1)))
                    + [-1, 1]
                    + ([0] * (n_params - used_params - (pulse_num+1)))
                ),
                lb = 0, ub = np.inf
            )
            for pulse_num in range(1, self.n_pulses)
        ]

        return consts
    
    @property
    def objective(self) -> Callable:
        """
        Returns the objective function for the model
        """
        return multi_channel_multi_pulse_multi_fluorophore_objective

    @property 
    def optimizer_args(self) -> Tuple:
        """
        (n_pulses, n_fluorophores, n_states_by_fluorophore)
        """
        return (self.n_pulses, self.n_fluorophores, [len(f.exps) for f in self._fluorophores])
    
    def __repr__(self) -> str:
        retstr = "MPMFMCFlimParams object: \n\n"
        retstr += "\tN_photons:\n"
        for n_photon in self._n_photons:
            retstr += "\t\t"+n_photon.__repr__() + "\n"
        retstr += "Fluorophores:\n"
        for fluorophore in self._fluorophores:
            retstr += "\t"+fluorophore.__repr__() + "\n"
        retstr += "\tDelta mus:\n"
        for delta_mu in self._delta_mus:
            retstr += "\t\t"+delta_mu.__repr__() + "\n"
        retstr += "\t\tNoise: " + str(self.noise) + "\n"
        return retstr