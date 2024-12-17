"""
@Title: URA/MURA Coded Mask Interface
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

    def __init__(self,
                 pattern_type: str,
                 rank: int,
                 padding: bool = False):
        
        self.padding = padding
        self.mask_type = self._get_mask_type(pattern_type, rank)
        self.mask, self.open_fraction = self._get_mask_pattern(self.padding)
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

        if self.padding:
            self.detector_image = correlate(self.mask, self.sky_image, mode=self._mode)
        else:
            zero_pad_mask = self._get_padded_array(self.basic_pattern)
            self.detector_image = correlate(zero_pad_mask, self.sky_image, mode=self._mode)

        if detector_background_rate:
            self.detector_image += np.random.poisson(detector_background_rate,
                                                     self.detector_image.shape)
        
        return self.detector_image


    def decode(self) -> c.Sequence:
        """Returns the reconstructed sky image from the detector image."""
        
        if self.padding:
            rec_sky = correlate(self.decoder, self.detector_image, mode=self._mode)
        else:
            zero_pad_decoder = self._get_padded_array(self.decoder)
            rec_sky = correlate(zero_pad_decoder, self.detector_image, mode=self._mode)

        self.sky_reconstruction = rec_sky

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

        if padding:
            mask = self._get_padded_array(self.basic_pattern, pad=True)
        else:
            mask = self.basic_pattern
            
        open_fraction = mask.sum()/(mask.shape[0]*mask.shape[1])
        
        return mask, open_fraction

    def _get_decoding_pattern(self) -> c.Sequence:
        
        G = 2*self.mask - 1

        if self.mask_type.pattern_type == 'MURA': G[0, 0] = 1

        G /= self.basic_pattern.sum()

        return G
    
    def _get_padded_array(self, array, pad=False) -> c.Sequence:
        # why this complicated code when you have can achieve the same in one line?
        # TODO: pad using numpy's pad with `wrap` mode.
        n, m = array.shape
        padded_array = np.zeros((2*n - 1, 2*m - 1))
        pad_n, pad_m = (n - 1)//2, (m - 1)//2

        padded_array[pad_n : -pad_n, pad_m : -pad_m] = array
        
        if pad:
            # top-left corner
            padded_array[:pad_n, :pad_m] = array[-pad_n:, -pad_m:]
            # top-central
            padded_array[pad_n : -pad_n, :pad_m] = array[:, -pad_m:]
            # top-right corner
            padded_array[-pad_n:, :pad_m] = array[:pad_n, -pad_m:]

            # mid-left
            padded_array[:pad_n, pad_m : -pad_m] = array[-pad_n:, :]
            # mid-right
            padded_array[-pad_n:, pad_m : -pad_m] = array[:pad_n, :]

            # bottom-left corner
            padded_array[:pad_n, -pad_m:] = array[-pad_n:, :pad_m]
            # bottom-central
            padded_array[pad_n : -pad_n, -pad_m:] = array[:, :pad_m]
            # bottom-right corner
            padded_array[-pad_n:, -pad_m:] = array[:pad_n, :pad_m]
        
        return padded_array
    
    @property
    def _mode(self):
        return 'valid'


# end