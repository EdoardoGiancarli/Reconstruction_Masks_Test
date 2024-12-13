"""
@Title: Test for URA/MURA Coded Mask Pattern Simulation
@Author: Edoardo Giancarli
@Date: 13/12/24
@Content:
    - TestCodedMaskInterface: Tests the CodedMaskInterface class in codedmaskinterface.py.
"""

import unittest
from unittest import TestCase
import numpy as np
from codedmaskinterface import CodedMaskInterface
from maskpattern import URAMaskPattern, MURAMaskPattern


class TestCodedMaskInterface(TestCase):
    """Tests the CodedMaskInterface class in codedmaskinterface.py."""

    def setUp(self):
        self.cmi_ura = CodedMaskInterface('ura', 0)
        self.cmi_mura = CodedMaskInterface('mura', 0)
        self.ura_cmi_dummy = CodedMaskInterface('ura', 0)
        self.mura_cmi_dummy = CodedMaskInterface('mura', 0)
        self.sky_image_dummy = np.random.randint(1, 11, (5, 3))
    

    def test_initialization(self):
        self.assertEqual(self.cmi_ura.mask_type.pattern_type, "URA")
        self.assertEqual(self.cmi_mura.mask_type.pattern_type, "MURA")

    def test_get_mask_type(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.cmi_ura._get_mask_type('ura', -2))
            self.assertEqual(self.cmi_ura._get_mask_type('mura', -2))

    def test_get_mask_pattern(self):
        np.testing.assert_array_equal(self.cmi_ura.mask, URAMaskPattern(0).basic_pattern, strict=True)
        self.assertTrue(0 < self.cmi_ura.open_fraction and self.cmi_ura.open_fraction < 1)
        np.testing.assert_array_equal(self.cmi_mura.mask, MURAMaskPattern(0).basic_pattern, strict=True)
        self.assertTrue(0 < self.cmi_mura.open_fraction and self.cmi_mura.open_fraction < 1)

    def test_get_mask_pattern_padding(self):
        pass

    def test_get_decoding_pattern(self):
        G_ura = 2*URAMaskPattern(0).basic_pattern - 1
        G_mura = 2*MURAMaskPattern(0).basic_pattern - 1; G_mura[0, 0] = 1
        np.testing.assert_array_equal(self.cmi_ura.decoder, G_ura, strict=False)
        np.testing.assert_array_equal(self.cmi_mura.decoder, G_mura, strict=False)

    def test_cmi_properties(self):
        np.testing.assert_array_equal(self.cmi_ura.basic_pattern, URAMaskPattern(0).basic_pattern)
        np.testing.assert_array_equal(self.cmi_mura.basic_pattern, MURAMaskPattern(0).basic_pattern)

        test_attr = ["basic_pattern", "basic_pattern_shape", "mask_shape", "decoder_shape",
                     "sky_image_shape", "detector_image_shape", "sky_reconstruction_shape"]
        
        _ = [getattr(self.cmi_ura, attr) for attr in test_attr[:4]]
        _ = [getattr(self.cmi_mura, attr) for attr in test_attr[:4]]

        with self.assertRaises(AttributeError):
            _ = [getattr(self.cmi_ura, attr) for attr in test_attr[-3:]]
            _ = [getattr(self.cmi_mura, attr) for attr in test_attr[-3:]]

    def test_psf(self):
        self.cmi_ura.psf()
        self.cmi_mura.psf()

    def test_SNR(self):
        with self.assertRaises(AttributeError):
            self.cmi_ura.snr()
            self.cmi_mura.snr()
    
    def test_cai(self):
        # encoding
        self.ura_cmi_dummy.encode(self.sky_image_dummy)
        self.mura_cmi_dummy.encode(self.sky_image_dummy)
        # decoding
        self.ura_cmi_dummy.decode()
        self.mura_cmi_dummy.decode()
        # SNR
        self.ura_cmi_dummy.snr()
        self.mura_cmi_dummy.snr()
        # attributes
        test_attr = ["sky_image_shape", "detector_image_shape", "sky_reconstruction_shape"]
        _ = [getattr(self.ura_cmi_dummy, attr) for attr in test_attr]
        _ = [getattr(self.mura_cmi_dummy, attr) for attr in test_attr]




if __name__ == "__main__":
    unittest.main()


# end