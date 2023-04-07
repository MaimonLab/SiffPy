from functools import reduce

import numpy as np
from napari.utils.events import Event, EmitterGroup
from skimage import morphology

from siffpy.siffplot.roi_protocols.protocerebral_bridge.von_mises.napari_tools.flower_plot import (
    RosettePetal,
)

class Seed():
    """ Indexed by pixel coordinates, stores a list of RosettePetals and masks."""
    def __init__(
            self,
            px: np.ndarray,
            rosette : RosettePetal,
            mask : np.ndarray,
            source_image : np.ndarray = None,
        ):
        self.px = px
        self.rosette = rosette
        self.mask = mask
        self.source_image = source_image
        self.events = EmitterGroup(
            self,
        )
        self.events.add(
            changed = Event,
        )
        self.rosette.events.changed.connect(self.update_mask)
        self.selected = True
        self.rosette.events.changed()

    @classmethod
    def flood_around_seed(
        cls,
        image_arr : np.ndarray,
        seed_idx : np.ndarray,
        central_val : np.complex128,
        tolerance : float,
        neighborhood = np.ones((3,1,3,3))
    )->np.ndarray:
        """
        Floods a mask from a seed point, taking all values
        within a certain tolerance of the central value contiguous
        with the seed point.
        """
        difference_from_seed = np.abs(image_arr - central_val)

        return morphology.flood(
            difference_from_seed,
            seed_idx,
            tolerance=tolerance,
            #footprint = footprint,    
        )

    def on_px_select(self):
        self.rosette.px_selected()
        self.selected = True

    def on_px_deselect(self):
        self.rosette.px_deselected()  
        self.selected=False

    def update_mask(self, event):
        ros : RosettePetal = event.source
        central_val = ros.mag * np.exp(1j*ros.mean)
        tolerance = ros.width/(4*np.pi)
        seed = tuple(self.px.astype(int).tolist()) # (z,y,x)
        tmp_mask = self.flood_around_seed(
            image_arr = self.source_image.squeeze(),
            seed_idx = seed,
            central_val = central_val,
            tolerance = tolerance,
        )
        self.mask[:] = tmp_mask[:]
        self.events.changed()

    def __eq__(self, other)->bool:
        if isinstance(other, Seed):
            return np.array_equal(self.px, other.px)
        else:
            return self.px == other
        
    def __hash__(self):
        return hash(tuple(self.px.flatten()))
    
    def __del__(self):
        del self.rosette

class SeedManager():
    """
    A class that keeps track of Seeds and their masks
    """
    def __init__(
            self,
            fig,
            ax,
            seeds : list[Seed] = [],
            source_image : np.ndarray = None,
        ):
        self.fig = fig
        self.ax = ax
        self.seeds = seeds
        self.source_image = source_image
        self.events = EmitterGroup(
            self,
        )
        self.events.add(
            changed = Event,
        )

    def set_source_image(self, image : np.ndarray):
        self.source_image = image
        for seed in self.seeds:
            seed.source_image = image

    def on_seed_change(self, event):
        self.events.changed()

    def create_seed(
            self,
            px_coords : np.ndarray,
            rgb_value : np.ndarray,
            fft_value : np.complex128,
        ):
        seed = Seed(
            px = px_coords,
            rosette = RosettePetal(
                fig = self.fig,
                ax = self.ax,
                mean = np.angle(fft_value),
                width = 0.1,
                mag = np.abs(fft_value),
                color = rgb_value,
            ),
            mask = np.zeros_like(self.source_image.squeeze(), dtype=bool),
            source_image = self.source_image,
        )
        self.add_seed(seed)

    def add_seed(self, seed):
        if not isinstance(seed, Seed):
            raise TypeError('SeedManager.add_seed() expects a Seed object.')
        seed.events.changed.connect(self.on_seed_change)
        self.seeds.append(seed)

    def remove_seed(self, item):
        """ By seed pixel or by seed object """
        if isinstance(item, Seed):
            self.seeds.remove(item)
        elif isinstance(item, np.ndarray):
            for seed in self.seeds:
                if np.array_equal(seed.px, item):
                    self.seeds.remove(seed)
                    return
        elif hasattr(seed, '__iter__'):
            asarray = np.array(item, dtype=int)
            for seed in self.seeds:
                if np.array_equal(seed.px, asarray):
                    self.seeds.remove(seed)
                    return
        raise KeyError(f'No seed found at {item}')
    
    def __iter__(self):
        return iter([seed.px for seed in self.seeds])

    def __len__(self):
        return len(self.seeds)
    
    def __setitem__(self, key, val):
        raise NotImplementedError(
            """SeedManager does not support item assignment
            outside of the methods add_seed() and remove_seed()."""
        )

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.seeds[item]
        elif isinstance(item, np.ndarray):
            seed_px = item.astype(int)
            for seed in self.seeds:
                if np.array_equal(seed.px,seed_px):
                    return seed
            raise KeyError(f'No seed found at {seed_px}')
        elif hasattr(item, '__iter__'):
            try:
                asarray = np.array(item, dtype=int)
                for seed in self.seeds:
                    if np.array_equal(seed.px, asarray):
                        return seed
                raise KeyError(f'No seed found at {asarray}')
            except:
                raise TypeError(f'Could not cast item {item} to a numpy array')
        else:
            raise KeyError(f'No seed found at {item}')
    
    def __contains__(self, item):
        if isinstance(item, Seed):
            return item in self.seeds
        elif isinstance(item, np.ndarray):
            return any(np.array_equal(seed.px.astype(int),item.astype(int)) for seed in self.seeds)
        elif hasattr(item, '__iter__'):
            try:
                asarray = np.array(item, dtype = int)
                return any(np.array_equal(seed.px,asarray) for seed in self.seeds)
            except:
                raise TypeError(f'Could not cast item {item} to a numpy array')
        return False
    
    @property
    def mask(self):
        return reduce(
            np.logical_or,
            [seed.mask for seed in self.seeds],
            np.zeros_like(self.source_image.squeeze(), dtype=float),
        )
    
    @property
    def selected_mask(self):
        return reduce(
            np.logical_or,
            [seed.mask for seed in self.seeds if seed.selected],
            np.zeros_like(self.source_image.squeeze(), dtype=float),
        )
