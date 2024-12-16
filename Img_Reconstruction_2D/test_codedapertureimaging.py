"""
@Title: Test of Coded Aperture Imaging for Sky Image Analysis
@Author: Edoardo Giancarli
@Date: 13/12/24
@Content:
    - TestReconstruction: Tests the sky reconstruction based on Coded Aperture Imaging.
"""

import collections.abc as c
import numpy as np


class TestReconstruction:
    """Tests the sky reconstruction based on Coded Aperture Imaging."""

    def __init__(self, pattern_type: str):
        self.pattern_type = pattern_type.upper()

    def test_sky_reconstruction(self,
                                sky_reconstruction: c.Sequence,
                                sky_image: c.Sequence,
                                tolerance: int) -> bool:
        """Tests the whole sky image reconstruction wrt the simulated sky."""

        test = bool(np.all(self._is_close(sky_reconstruction, sky_image, tolerance)))

        print(f"#### {self.pattern_type} Coded Mask Sky Reconstruction Test ####")
        print(f"|S_hat - S| < {tolerance} : {test}\n")

        return test


    def test_sources_reconstruction(self,
                                    reconstr_sources: c.Sequence,
                                    simul_sources: c.Sequence,
                                    tolerance: int) -> bool:
        """Tests the sources reconstructed intensity wrt the simulated ones."""
        
        test = self._is_close(reconstr_sources, simul_sources, tolerance)

        print(f"#### {self.pattern_type} Coded Mask Sources Intensity Reconstruction Test ####")
        for i in range(len(reconstr_sources)):
            print(f"|S_hat[{i}] - S[{i}]| < {tolerance} : {test[i]}")

        return test


    def _is_close(self, S_hat, S, eps) -> c.Sequence[bool]:
        return np.abs(S_hat - S) <= eps


# end