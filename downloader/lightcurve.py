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

"""
From Camille

def query_data(sourceid, mission_name = "TESS", reduction_author = "QLP", plot_lc = 1):
    # query MAST for data
    search_results = lk.search_lightcurve(sourceid, mission = mission_name, author = reduction_author)
    print(search_results)

    # download all light curves
    lcs = search_results.download_all()

    # rebin to longest cadence
    results_table = search_results.table # get search results table
    exp_times = results_table["exptime"] # pull exposure times

    max_exp = np.max(exp_times) # find longest cadence
    max_exp_days = (max_exp * u.s).to(u.day).value # convert to days for rebinning

    rebin_indexes = [ii for ii, exp in enumerate(exp_times) if (exp != max_exp)] # indexes to rebin
    failed = []

    for ii in rebin_indexes:
        # lcs[ii].plot()
        try:
            lcs[ii] = lcs[ii].bin(time_bin_size = max_exp_days)
        except:
            print("{} could not be rebinned.".format(ii))
            failed.append(ii)
        # lcs[ii].plot()
    
    lcs = lk.LightCurveCollection([lc for ff, lc in enumerate(lcs) if ff not in failed])

    # stitch light curves together
    lc = lcs.stitch()

    # get rid of outliers
    m = (lc.quality == 0)
    lc = lc[m].remove_outliers(sigma = 5).flatten(window_length = 25) # high-pass filter

    # plot all light curves
    if plot_lc:
        lc.plot()
        plt.show()

    # get time, flux data
    time, flux = lc.time.value, lc.flux.value

    return time, flux, lc
"""