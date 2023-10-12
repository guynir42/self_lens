# calculates the number of detections per survey in the WD and BH/NS cases
# uses the range of BH mass indices that is expected ("low", "mid", "high")
# to get the range of estimates based on that uncertainty.
# We can leave the temperature range for WDs at "mid", I think that's fine.

import os

import numpy as np
import xarray as xr

from src.distributions import fetch_distributions
from src.utils import fetch_volumes

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


########################### start the script ###########################

CLEAR_CACHE = False
INCLUDE_FOLLOWUP = False

survey_names = ["TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]
# survey_names = ["ZTF"]  # debug only!

n_wd = []  # number of detections for WDs
nf_wd = []  # number of detections with followup for WDs

# white dwarfs:
for survey in survey_names:
    ds = fetch_distributions(survey=survey, obj="WD", space_density_pc3=1 / 1818, clear_cache=CLEAR_CACHE)
    # print(f"Number of detections for {survey} WDs: {ds['detections'].sum().values}")
    n_wd.append(float(ds.detections.sum()))
    nf_wd.append(float(ds.detections_followup.sum()))

n_bh = []  # number of detections for BHs
nf_bh = []  # number of detections with followup for BHs

# black holes, using three different BH mass distributions:
indices = ["low", "mid", "high"]

for idx in indices:
    n_bh.append([])
    nf_bh.append([])

    for survey in survey_names:
        ds = fetch_distributions(
            survey=survey, obj="BH", space_density_pc3=10**-5, mass_index=idx, clear_cache=CLEAR_CACHE
        )
        n_bh[-1].append(float(ds.detections.sum()))
        nf_bh[-1].append(float(ds.detections_followup.sum()))

# print the results in a latex table format:
for i, survey in enumerate(survey_names):
    s = survey.replace("_", "\\_")
    if INCLUDE_FOLLOWUP:
        print(
            f"{s} & {n_wd[i]:.1f} ({nf_wd[i]:.1f}) & "
            f"{n_bh[0][i]:.1f} ({nf_bh[0][i]:.1f}) & "
            f"{n_bh[1][i]:.1f} ({nf_bh[1][i]:.1f}) & "
            f"{n_bh[2][i]:.1f} ({nf_bh[2][i]:.1f}) ",
            end="",
        )
    else:
        print(f"{s} & {n_wd[i]:.1f} & {n_bh[0][i]:.1f} & {n_bh[1][i]:.1f} & {n_bh[2][i]:.1f} ", end="")
    if i < len(survey_names) - 1:
        print(" \\\\")
    else:
        print("")
