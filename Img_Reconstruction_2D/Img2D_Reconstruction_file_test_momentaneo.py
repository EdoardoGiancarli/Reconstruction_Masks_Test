import numpy as np
from matplotlib.colors import ListedColormap as lc

import plot_module as plot
import maskpattern as mp
import codedmaskinterface as cmi
import codedapertureimaging as cai
from test_codedapertureimaging import TestReconstruction


def _normalize(array):
    return (array - array.min())/(array.max() - array.min())



pattern_type = 'mura'
rank = 0
sources_flux = np.array([30, 50, 70])
mask_padding = True
sky_background_rate = 5
sources_pos = None
detector_background_rate = None
print_info = True

cai_args_mura = (pattern_type, rank, sources_flux, mask_padding, sky_background_rate,
                 sources_pos, detector_background_rate, print_info)

detector_image, sky_reconstruction, source_info = cai.cai_simulation(*cai_args_mura)


mask = source_info['mask_pattern']
decoder = source_info['mask_decoder']
psf = source_info['mask_PSF']


# plot sky image, sky reconstruction and SNR
sky_image = source_info['sky_image']
snr = source_info['sky_reconstruction_SNR']



# plot reconstr_sky slices
idx = 1
pos = source_info['sources_pos'][idx]
S_hat_slicex = sky_reconstruction[pos[0], :]
S_hat_slicey = sky_reconstruction[:, pos[1]]



flag = False
if flag:
    plot.image_plot([mask, decoder, psf],
                    ["MURA Mask Pattern", "Mask Decoder", "MURA Mask PSF"],
                    cbarlabel=["Aperture", None, "Counts"],
                    cbarvalues=[[0, 1], [decoder.min(), decoder.max()], None],
                    cbarcmap=[lc(["DodgerBlue", "DeepSkyBlue"]), lc(["DeepPink", "Orange"]), "inferno"])
    
    plot.image_plot([_normalize(sky_image), _normalize(sky_reconstruction), snr],
                ["Simulated Sky Image", "Reconstructed Sky Image", "SNR for Reconstructed Sky"],
                cbarlabel=["Counts"]*2 + ["SNR"],
                cbarvalues=[None]*3,
                cbarcmap=["inferno"]*2 + ["rainbow"])
    
    plot.sequence_plot([S_hat_slicex, S_hat_slicey],
                   [f"$\\hat{{S}}_{idx}$ x-axis slice", f"$\\hat{{S}}_{idx}$ y-axis slice"],
                   xlabel=["$\\hat{{S}}$ x-axis", "$\\hat{{S}}$ y-axis"],
                   ylabel=["$\\hat{{S}}$ y-axis", "$\\hat{{S}}$ x-axis"],
                   style=["bar"]*2,
                   simulated_sources=[(pos[1], *pos, -5), (pos[0], *pos, -5)])



a = source_info["coded_mask_interface"]

g = a.basic_decoder*a.basic_pattern.sum()
print(g)

n, m = g.shape
pad_n, pad_m = (n - 1)//2, (m - 1)//2

print(np.pad(g, pad_width=((pad_n, pad_n), (pad_m, pad_m))))


padded_g = np.zeros((2*n - 1, 2*m - 1))
padded_g[pad_n : -pad_n, pad_m : -pad_m] = g

print(padded_g)

print(padded_g == np.pad(g, pad_width=((pad_n, pad_n), (pad_m, pad_m))))



print(decoder*a.basic_pattern.sum() == np.pad(g, pad_width=((pad_n, pad_n), (pad_m, pad_m)), mode='wrap'))

# end