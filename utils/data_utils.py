import numpy as np

def mags_to_flux(mag):
    return 10 ** (-0.4 * mag)

def flux_to_mag(flux):
    return -2.5 * np.log10(flux)

def distance_modulus(distance):
    return 5.0 * np.log10(distance / 10.0)

# Obviously.
def pc_to_ly(pc):
    return 3.26 * pc

def safe_loc(df, feh):
    levels = df.index.levels[1].to_numpy()
    feh_fixed = levels[np.argmin(np.abs(levels - feh))]
    return df.xs(feh_fixed, level=1)
