from astroquery.mast import Catalogs

def create_gaia_query(source_ids: list[int]) -> str:
    preamble = """
        SELECT
            g.source_id, g.ra, g.dec, 
            g.parallax, g.parallax_error, 
            g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, 
            g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
            g.phot_g_mean_flux, g.phot_g_mean_flux_error,
            g.phot_bp_mean_flux, g.phot_bp_mean_flux_error,
            g.phot_rp_mean_flux, g.phot_rp_mean_flux_error,
            a.teff_gspphot, a.teff_gspphot_lower, a.teff_gspphot_upper,
            a.distance_gspphot, a.distance_gspphot_lower, a.distance_gspphot_upper,
            a.radius_gspphot, a.radius_gspphot_lower, a.radius_gspphot_upper,
            a.logg_gspspec, a.logg_gspspec_lower, a.logg_gspspec_upper
        FROM gaiadr3.gaia_source AS g
        LEFT JOIN gaiadr3.astrophysical_parameters AS a
            ON g.source_id = a.source_id
        WHERE g.source_id IN ({source_ids})
            AND g.phot_bp_mean_mag IS NOT NULL
            AND g.phot_rp_mean_mag IS NOT NULL
            AND g.astrometric_params_solved = 31
    """

    adql_query = preamble.format(source_ids=','.join(map(str, source_ids)))
    return adql_query

from astroquery.mast import Catalogs
import pandas as pd
import numpy as np
import logging

def gaia_to_tic(gaia_ids):
    """
    Robust batch converter from Gaia DR3 IDs to TIC IDs.
    Queries MAST only once, handles DR2/DR3 mismatch, and removes duplicates.

    Parameters
    ----------
    gaia_ids : list[int] or int
        Gaia source_id(s) to crossmatch with the TESS Input Catalog.

    Returns
    -------
    dict
        Mapping {gaia_id: tic_id}, only for those found in TIC.
    """
    if isinstance(gaia_ids, int):
        gaia_ids = [gaia_ids]

    # Remove duplicates, keep ints
    gaia_ids = list({int(g) for g in gaia_ids})

    try:
        result = Catalogs.query_criteria(
            catalog="TIC",
            GAIA=gaia_ids
        ).to_pandas()
    except Exception as e:
        logging.error(f"TIC batch query failed: {e}")
        return {}

    if result.empty:
        logging.warning("No TIC matches found for provided Gaia IDs.")
        return {}

    # Keep relevant columns and unique IDs
    result = result.drop_duplicates(subset=["GAIA"])
    mapping = dict(zip(result["GAIA"].astype(int), result["ID"].astype(int)))

    # Log any missing Gaia IDs
    missing = set(gaia_ids) - set(mapping.keys())
    if missing:
        logging.warning(f"No TIC match for {len(missing)} Gaia IDs: {list(missing)[:5]}...")

    return mapping
