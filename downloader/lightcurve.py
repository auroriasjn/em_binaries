import lightkurve as lk

class LightCurveExtractor:
    def __init__(
        self,
        mission_name: str = "TESS",
        reduction_author: str = "SPOC",
    ):
        self.mission_name = mission_name
        self.reduction_author = reduction_author

    def query_source(self, source_id: str):
        # Query MAST
        search_results = lk.search_lightcurve(source_id, mission=self.mission_name, author=self.reduction_author)
        lcs = search_results.download_all()

        # Rebin to longest cadence
        results_table = search_results.table
        exp_times = results_table["exptime"]

        # stitch light curves together
        lc = lcs.stitch()

        # Get rid of outliers
        mask = (lc.quality == 0)
        lc = lc[mask].remove_outliers(sigma=5).flatten(window_length=25)

        return lc.time.value, lc.flux.value, lc

    def extract_lightcurves(self, source_ids: list[str]) -> dict[str, tuple]:
        lightcurves = {}
        for source_id in source_ids:
            time, flux, lc_obj = self.query_source(source_id)
            lightcurves[source_id] = (time, flux, lc_obj)
        return lightcurves