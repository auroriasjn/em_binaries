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