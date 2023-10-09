import os

import numpy as np
import xarray as xr

from src.distributions import marginalize_declinations
from src.utils import fetch_volumes

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


########################### start the script ###########################

survey_names = ["TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]

survey_names = ["TESS"]  # debug only!

for survey in survey_names:
    filename = os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_WD.nc")
    ds = xr.load_dataset(filename, decode_times=False)

    # finer grid for interpolations
    dm = 0.01
    dT = 100
    lens_masses = np.arange(ds.lens_mass.min(), ds.lens_mass.max(), dm)
    star_masses = np.arange(ds.star_mass.min(), ds.star_mass.max(), dm)
    lens_temps = np.arange(ds.lens_temp.min(), ds.lens_temp.max(), dT)
    star_temps = np.arange(ds.star_temp.min(), ds.star_temp.max(), dT)

    ev = marginalize_declinations(ds)  # get the effective volume, marginalized over declinations (uniform probability)
    for a in ev.semimajor_axis:
        ev2 = ev.isel(semimajor_axis=a).interp(
            lens_mass=lens_masses, star_mass=star_masses, lens_temp=lens_temps, star_temp=star_temps, method="cubic"
        )
        ev2[ev2 < 0] = 0  # remove negative values (that could arise from interpolation)
