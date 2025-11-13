import logging
import lightkurve as lk

from typing import List, Dict
from tqdm import tqdm
from utils import gaia_to_tic, download_eb_catalog, load_eb_catalog
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
            lc = lc.remove_nans().remove_outliers(sigma=5)

            # CBV systematics correction (if available)
            try:
                lc = lc.correct()
            except Exception:
                pass

            lc = lc.normalize()

            # Flatten sector by sector
            lc = lc.flatten(window_length=401, break_tolerance=10)

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
    
    def extract_eb_lightcurves(self, source_ids: List[str]) -> Dict[str, tuple]:
        # Load EB catalog
        eb_df = load_eb_catalog()

        # Standardize TIC column
        if "tic" in eb_df.columns:
            tic_col = "tic"
        else:
            # autodetect + rename to "tic"
            for col in ["TIC", "tic_id", "tess_id"]:
                if col in eb_df.columns:
                    eb_df["tic"] = eb_df[col].astype(str).str.strip()
                    tic_col = "tic"
                    break
            else:
                raise ValueError("EB catalog missing TIC column!")
        
        eb_tics = set(eb_df[tic_col].astype(str))

        # Map Gaia → TIC
        gaia_to_tic_map = gaia_to_tic(source_ids)

        # Collect TICs that are in the EB catalog
        matched_tics = []
        for source_id in tqdm(source_ids):
            tess_id = gaia_to_tic_map.get(int(source_id))
            logging.info(f"Inspecting for EB membership: Gaia {source_id} -> TIC {tess_id}")
            
            if tess_id is None:
                continue
            
            # normalize
            tess_id_str = str(tess_id).strip()

            if tess_id_str in eb_tics:
                matched_tics.append(source_id)   # note: use Gaia IDs for extraction

        logging.info(f"Found {len(matched_tics)} EB matches.")

        # Re-use your existing method to pull light curves normally
        return self.extract_gaia_lightcurves(matched_tics)


