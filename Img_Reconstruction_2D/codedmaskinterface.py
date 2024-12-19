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
        self.mask, self.decoder, self.open_fraction = self._get_mask_pattern(self.padding)

        self.detector_image = None
        self.sky_reconstruction = None
    
    def __getattr__(self, name):
        attribute_map = {"basic_pattern": lambda: self.mask_type.basic_pattern,
                         "basic_pattern_shape": lambda: self.mask_type.basic_pattern.shape,
                         "mask_shape": lambda: self.mask.shape,
                         "basic_decoder": lambda: self.mask_type.basic_decoder,
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
        assert self.sky_image_shape == self.basic_pattern_shape

        if self.padding:
            self.detector_image = correlate(self.mask, self.sky_image, mode=self._mode)
        else:
            zero_pad_mask = self._get_padded_array(self.basic_pattern)
            self.detector_image = correlate(zero_pad_mask, self.sky_image, mode=self._mode)

        if detector_background_rate:
            self.detector_image += np.random.poisson(detector_background_rate,
                                                     self.detector_image.shape)
        
        assert self.detector_image_shape == self.basic_pattern_shape
        
        return self.detector_image


    def decode(self) -> c.Sequence:
        """Returns the reconstructed sky image from the detector image."""
        
        if self.padding:
            self.sky_reconstruction = correlate(self.decoder, self.detector_image, mode=self._mode)
        else:
            zero_pad_decoder = self._get_padded_array(self.decoder)
            self.sky_reconstruction = correlate(zero_pad_decoder, self.detector_image, mode=self._mode)
        
        assert self.sky_reconstruction_shape == self.sky_image_shape 

        return self.sky_reconstruction
    

    def psf(self) -> c.Sequence:
        """Returns the mask PSF."""

        decoder = self.decoder if self.padding else self._get_padded_array(self.decoder, mode='wrap')

        return correlate(decoder, self.basic_pattern, mode=self._mode)
    

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
        
        assert mask_pattern.pattern_type == pattern_type.upper()
        
        return mask_pattern

    def _get_mask_pattern(self, padding) -> tuple[c.Sequence, float]:

        if padding:
            mask = self._get_padded_array(self.basic_pattern, mode='wrap')
            decoder = self._get_padded_array(self.basic_decoder, mode='wrap')
        else:
            mask = self.basic_pattern
            decoder = self.basic_decoder
            
        open_fraction = mask.sum()/np.prod(mask.shape)
        
        return mask, decoder, open_fraction
    
    def _get_padded_array(self, array, mode='zero') -> c.Sequence:

        n, m = array.shape
        pad_n, pad_m = (n - 1)//2, (m - 1)//2

        if mode not in ['zero', 'wrap']:
            raise ValueError(f"invalid mode {mode}. Mode must be 'zero' or 'wrap'.")
        
        elif mode == 'zero':
            padded_array = np.pad(array, pad_width=((pad_n, pad_n), (pad_m, pad_m)))
        
        else:
            padded_array = np.pad(array, pad_width=((pad_n, pad_n), (pad_m, pad_m)), mode='wrap')
        
        assert padded_array.shape == (2*n -1, 2*m - 1)

        # deprecated
        #flag=False
        #if flag:
        #    padded_array[pad_n : -pad_n, pad_m : -pad_m] = array
        #    # top-left corner
        #    padded_array[:pad_n, :pad_m] = array[-pad_n:, -pad_m:]
        #    # top-central
        #    padded_array[pad_n : -pad_n, :pad_m] = array[:, -pad_m:]
        #    # top-right corner
        #    padded_array[-pad_n:, :pad_m] = array[:pad_n, -pad_m:]
        #    # mid-left
        #    padded_array[:pad_n, pad_m : -pad_m] = array[-pad_n:, :]
        #    # mid-right
        #    padded_array[-pad_n:, pad_m : -pad_m] = array[:pad_n, :]
        #    # bottom-left corner
        #    padded_array[:pad_n, -pad_m:] = array[-pad_n:, :pad_m]
        #    # bottom-central
        #    padded_array[pad_n : -pad_n, -pad_m:] = array[:, :pad_m]
        #    # bottom-right corner
        #    padded_array[-pad_n:, -pad_m:] = array[:pad_n, :pad_m]
        
        return padded_array
    
    @property
    def _mode(self):
        return 'valid'


# end