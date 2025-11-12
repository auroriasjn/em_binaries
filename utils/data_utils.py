import numpy as np

def mags_to_flux(mag):
    return 10 ** (-0.4 * mag)

def flux_to_mag(flux):
    return -2.5 * np.log10(flux)

def distance_modulus(distance):
    return 5.0 * np.log10(distance / 10.0)