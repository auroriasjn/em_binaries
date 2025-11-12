import logging
import lightkurve as lk

from typing import List, Dict
from tqdm import tqdm
from utils import gaia_to_tic
from lightkurve.search import SearchError

class LightCurveExtractor:
    def __init__(self, mission_name="TESS", reduction_author=None, cadence="short", download_dir=None):
        self.mission_name = mission_name
        self.reduction_author = reduction_author
        self.cadence = cadence

        if download_dir is not None:
            lk.conf.download_dir = download_dir
            logging.info(f"Set Lightkurve download directory to {download_dir}")

    def query_source(self, source_id: str):
        try:
            search_results = lk.search_lightcurve(
                f"TIC {source_id}", 
                mission=self.mission_name, 
                author=self.reduction_author,
                exptime=self.cadence
            )
            lcs = search_results.download_all()

        except (RuntimeError, SearchError, ValueError) as e:
            logging.error(f"Search/download failed for {source_id}: {e}")
            return None

        if lcs is None:
            logging.error(f"No LCs returned for {source_id}")
            return None

        # Stitch segments together
        try:
            lc = lcs.stitch()
        except Exception as e:
            logging.error(f"Error stitching LC for {source_id}: {e}")
            return None

        # Clean data
        try:
            lc = lc.remove_outliers(sigma=5).flatten(window_length=25)
        except Exception as e:
            logging.error(f"Error cleaning LC for {source_id}: {e}")
            return None

        return lc.time.value, lc.flux.value, lc

    def extract_gaia_lightcurves(self, source_ids: List[str]) -> Dict[str, tuple]:
        lightcurves = {}
        gaia_to_tic_map = gaia_to_tic(source_ids)

        for source_id in tqdm(source_ids):
            tess_id = gaia_to_tic_map.get(int(source_id))
            logging.info(f"Querying source_id {source_id} -> tess_id {tess_id}")
            if tess_id is None:
                continue

            result = self.query_source(str(tess_id))
            if result is None:
                continue

            lightcurves[source_id] = result

        return lightcurves
