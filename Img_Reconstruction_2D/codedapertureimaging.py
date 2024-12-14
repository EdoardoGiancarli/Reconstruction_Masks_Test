"""
@Title: Coded Aperture Imaging for Sky Image Analysis
@Author: Edoardo Giancarli
@Date: 14/12/24
@Content:
    - CAI_pipeline: Simulates the reconstruction of the sky for a coded mask camera.
"""

import collections.abc as c
import numpy as np


class CAI_pipeline:
    """Simulates the reconstruction of the sky for a coded mask camera."""

    def __init__(self):
        pass


    def cai_simulation(self) -> tuple[c.Sequence, c.Sequence, dict]:
        """Coded Aperture Imaging procedure simulation."""

        pass


    def sky_image_simulation(self) -> tuple[c.Sequence, None | c.Sequence]:
        """Simulates the sky image given the sources flux."""
        
        sky_image = np.zeros(sky_shape)

        # assign fluxes to point-like sources
        for i, pos in enumerate(sources_pos):
            sky_image[*pos] = sources_flux[i]

        # add sky background
        if sky_background_rate is not None:
            sky_background = np.random.poisson(sky_background_rate, sky_shape)
            sky_image += sky_background
        else:
            sky_background = None
        
        return sky_image, sky_background


    def sky_image_reconstruction(self):
        """Reconstruction of the simulated sky image."""
        
        pass


    def _wrap_reconstructed_sources(self) -> c.Sequence:
        pass


    def _print_info(self) -> None:
        pass


# end