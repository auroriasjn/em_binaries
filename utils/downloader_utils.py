import os
import pandas as pd
import numpy as np
import logging
import requests

from astroquery.mast import Catalogs

EB_URL = (
    "https://archive.stsci.edu/hlsps/tess-ebs/"
    "hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
)

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

def gaia_to_tic(gaia_ids):
    """
    Robust batch converter from Gaia DR3 IDs to TIC IDs.
    Queries MAST only once, handles DR2/DR3 mismatch, and removes duplicates.

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

def download_eb_catalog(path="data/tess_eb_catalog.csv"):
    """Download TESS EB catalog if not present."""
    if os.path.exists(path):
        logging.info(f"Using cached TESS EB catalog at {path}")
        return path

    logging.info("Downloading TESS Eclipsing Binary catalog...")
    r = requests.get(EB_URL)
    r.raise_for_status()

    with open(path, "wb") as f:
        f.write(r.content)

    logging.info(f"Saved TESS EB catalog to {path}")
    return path


def load_eb_catalog(path="data/tess_eb_catalog.csv"):
    """Load EB catalog as DataFrame."""
    path = download_eb_catalog(path)
    df = pd.read_csv(path)

    # Standardize TIC column name
    possible_tic_cols = ["TIC", "tic_id", "tic", "tess_id"]
    for col in possible_tic_cols:
        if col in df.columns:
            tic_col = col
            break
    else:
        raise ValueError("No TIC column found in EB catalog.")

    df["tic"] = df[tic_col].astype(str).str.strip()
    return df
