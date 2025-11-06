import numpy as np

def mags_to_flux(mag):
    return 10 ** (-0.4 * mag)

def flux_to_mag(flux):
    return -2.5 * np.log10(flux)