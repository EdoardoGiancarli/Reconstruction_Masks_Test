"""
@Title: URA/MURA Coded Mask Pattern Simulation
@Author: Edoardo Giancarli
@Date: 13/12/24
@Content:
    - CodedMaskInterface: Interface for the URA/MURA coded mask camera analysis.
"""

import collections.abc as c
import numpy as np
from maskpattern import URAMaskPattern, MURAMaskPattern
from scipy.signal import correlate


class CodedMaskInterface:
    """Interface for the URA/MURA coded mask camera analysis."""

    def __init__(self, pattern_type: str,
                 rank: int,
                 padding: None | tuple[int, int] = None):
        
        self.mask_type = self._get_mask_type(pattern_type, rank)
        self.mask, self.open_fraction = self._get_mask_pattern(padding)
        self.decoder = self._get_decoding_pattern()

        self.detector_image = None
        self.sky_reconstruction = None
    
    def __getattr__(self, name):
        attribute_map = {"basic_pattern": lambda: self.mask_type.basic_pattern,
                         "basic_pattern_shape": lambda: self.mask_type.basic_pattern.shape,
                         "mask_shape": lambda: self.mask.shape,
                         "decoder_shape": lambda: self.decoder.shape,
                         "sky_image_shape": lambda: self.sky_image.shape,
                         "detector_image_shape": lambda: self.detector_image.shape,
                         "sky_reconstruction_shape": lambda: self.sky_reconstruction.shape}
        
        if name not in attribute_map:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return attribute_map[name]()
    

    def encode(self,
               sky_image: c.Sequence,
               detector_background_rate: None | float = None) -> c.Sequence:
        """Returns the detector image from the simulated sky image."""

        self.sky_image = sky_image
        self.detector_image = correlate(self.mask, self.sky_image, mode='same')

        if detector_background_rate:
            self.detector_image += np.random.poisson(detector_background_rate,
                                                     self.detector_image.shape)
        
        return self.detector_image


    def decode(self) -> c.Sequence:
        """Returns the reconstructed sky image from the detector image."""
        
        rec_sky = correlate(self.decoder, self.detector_image, mode='same')
        self.sky_reconstruction = rec_sky/self.mask.sum()

        return self.sky_reconstruction
    

    def psf(self):
        """Returns the mask PSF."""

        return correlate(self.decoder, self.mask, mode='same')
    

    def snr(self) -> c.Sequence:
        """Returns the SNR for the reconstructed image."""

        return self.sky_reconstruction/np.sqrt(np.abs(self.sky_reconstruction.sum()))
    

    def _get_mask_type(self, pattern_type, rank) -> c.Callable:

        if pattern_type not in ['ura', 'mura']:
            raise ValueError(f"Invalid pattern_type = {pattern_type}, must be 'ura' or 'mura'.")
        elif pattern_type == 'ura':
            mask_pattern = URAMaskPattern(rank)
        else:
            mask_pattern = MURAMaskPattern(rank)
        
        return mask_pattern

    def _get_mask_pattern(self, padding) -> tuple[c.Sequence, float]:

        if padding is not None:
            pass

        else:
            mask = self.basic_pattern
            open_fraction = mask.sum()/(mask.shape[0]*mask.shape[1])
        
        return mask, open_fraction

    def _get_decoding_pattern(self) -> c.Sequence:
        
        G = 2*self.mask - 1

        if self.mask_type.pattern_type == 'MURA': G[0, 0] = 1

        return G


# end