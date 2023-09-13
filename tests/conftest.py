import sys
import os
import pytest
import time
import requests

import pickle
import numpy as np
import xarray as xr
from timeit import default_timer as timer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.transfer_matrix import TransferMatrix
from src.grid_scan import Grid
from src.simulator import Simulator
from src.survey import Survey


ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def grid():
    g = Grid()
    g.setup_demo_scan(wd_lens=True)
    # g.surveys = [Survey("TESS")]  # debugging only!
    # g.semimajor_axes = np.arange(0.001, 0.3, 0.05)
    # g.semimajor_axes = g.semimajor_axes[39:47]
    # g.declinations = [0.0005]
    always_recalculate = True
    filename = os.path.join(ROOT_FOLDER, "tests/test_data/test_dataset.nc")
    if always_recalculate or not os.path.isfile(filename):
        g.run_simulation(keep_systems=True)
        g.dataset.to_netcdf(filename)
        with open(os.path.join(ROOT_FOLDER, "tests/test_data/test_systems.pickle"), "wb") as file:
            pickle.dump(g.systems, file)
    else:
        g.dataset = xr.load_dataset(filename)

    return g


@pytest.fixture(scope="module")
def wd_ds():
    """
    This will lazy-download the big simulation files if they don't exist in "saved"
    """
    if not os.path.isdir(os.path.join(ROOT_FOLDER, "saved")):
        os.mkdir(os.path.join(ROOT_FOLDER, "saved"))

    survey_names = ["ZTF", "TESS", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]

    # TODO: replace these with permanent storage on e.g., Zenodo
    links = {
        "ZTF": "https://www.dropbox.com/s/cu0qpy2w7cyvmav/simulate_ZTF_WD.nc?dl=1",
        "TESS": "https://www.dropbox.com/s/1oybcd15rsaaaqo/simulate_TESS_WD.nc?dl=1",
        "LSST": "https://www.dropbox.com/s/fhmayvx7o2m9ery/simulate_LSST_WD.nc?dl=1",
        "DECAM": "https://www.dropbox.com/s/wxhm3ogiz91kwjk/simulate_DECAM_WD.nc?dl=1",
        "CURIOS": "https://www.dropbox.com/s/5dys94f12ud6jhm/simulate_CURIOS_WD.nc?dl=1",
        "CURIOS_ARRAY": "https://www.dropbox.com/scl/fi/kyfowkp8w01r7jgdcec5e/simulate_CURIOS_ARRAY_WD.nc?rlkey=p1k1a2pmv09gmd60f1ytexq0f&dl=1",
        "LAST": "https://www.dropbox.com/s/kijsjwgm3rijd1f/simulate_LAST_WD.nc?dl=1",
    }

    for survey in survey_names:
        if not os.path.isfile(os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_WD.nc")):
            print(f"Downloading WD simulation results for {survey}...")
            r = requests.get(links[survey], allow_redirects=True)
            open(os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_WD.nc"), "wb").write(r.content)

    ds = None
    for survey in survey_names:
        filename = os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_WD.nc")
        new_ds = xr.load_dataset(filename, decode_times=False)
        for name in ["lens_temp", "star_temp"]:
            new_ds[name] = np.round(new_ds[name])

        if ds is None:
            ds = new_ds
        else:
            ds = xr.concat([ds, new_ds], dim="survey")

    if ds is None:
        raise ValueError("No data loaded!")

    return ds


@pytest.fixture(scope="module")
def bh_ds():
    survey_names = ["ZTF", "TESS", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]
    # TODO: replace these with permanent storage on e.g., Zenodo
    links = {
        "ZTF": "https://www.dropbox.com/scl/fi/jzzwuub7925veavn777q3/simulate_ZTF_BH.nc?rlkey=c1ys6cq9vuj2q4m6y8jqo3jps&dl=1",
        "TESS": "https://www.dropbox.com/scl/fi/91wbqzlcq498486l8dcy5/simulate_TESS_BH.nc?rlkey=921lpijdhncn9ryr0ot1nw6qs&dl=1",
        "LSST": "https://www.dropbox.com/scl/fi/50g806rerxrawqgfv5ac6/simulate_LSST_BH.nc?rlkey=rbxeu7oummqe1g5dkvgpb87y9&dl=1",
        "DECAM": "https://www.dropbox.com/scl/fi/mvbcb7i361o2gqkbh5z44/simulate_DECAM_BH.nc?rlkey=npanms5tly3gqbeevtkurx9ep&dl=1",
        "CURIOS": "https://www.dropbox.com/scl/fi/f97fgm6b20wajnlk8znsi/simulate_CURIOS_BH.nc?rlkey=1stl4ox794n1kktb7o7807myw&dl=1",
        "CURIOS_ARRAY": "https://www.dropbox.com/scl/fi/jxzvvha1t0tezvoyy0ri1/simulate_CURIOS_ARRAY_BH.nc?rlkey=aa2tw4lhkxntg6z3mjy0dtjgl&dl=1",
        "LAST": "https://www.dropbox.com/scl/fi/2idhlgxcv1o47twryme95/simulate_LAST_BH.nc?rlkey=qlhgzxubiiq5y4xrnk8epm0ai&dl=1",
    }

    for survey in survey_names:
        if not os.path.isfile(os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_BH.nc")):
            print(f"Downloading BH simulation results for {survey}...")
            r = requests.get(links[survey], allow_redirects=True)
            open(os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_BH.nc"), "wb").write(r.content)

    ds = None
    for survey in survey_names:
        filename = os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_BH.nc")
        new_ds = xr.load_dataset(filename, decode_times=False)
        for name in ["lens_temp", "star_temp"]:
            new_ds[name] = np.round(new_ds[name])

        if ds is None:
            ds = new_ds
        else:
            ds = xr.concat([ds, new_ds], dim="survey")

    if ds is None:
        raise ValueError("No data loaded!")

    return ds


@pytest.fixture
def ztf():
    return Survey("ZTF")


@pytest.fixture
def sim():
    return Simulator()


@pytest.fixture
def matrix():
    return TransferMatrix.from_file(os.path.join(ROOT_FOLDER, "tests/test_data/matrix.npz"))


@pytest.fixture
def matrix_large():
    return TransferMatrix.from_file(os.path.join(ROOT_FOLDER, "matrices/matrix_SR1.000-5.000_D0.000-20.000.npz"))


@pytest.fixture
def counting_matrix_factory():
    def factory(
        min_dist=0,
        max_dist=0.5,
        step_dist=0.1,
        min_source=0.1,
        max_source=0.3,
        step_source=0.1,
        max_occulter=0.3,
        step_occulter=0.3,
    ):
        T = TransferMatrix()
        T.min_dist = min_dist
        T.max_dist = max_dist
        T.step_dist = step_dist
        T.min_source = min_source
        T.max_source = max_source
        T.step_source = step_source
        T.max_occulter = max_occulter
        T.step_occulter = step_occulter
        T.update_axes()
        T.allocate_arrays()

        # fill with mock data
        t0 = timer()
        for i in range(T.data_size()):
            T.flux[np.unravel_index(i, T.flux.shape)] = float(i)
            T.input_flux[np.unravel_index(i, T.flux.shape)] = float(i)
            T.moment[np.unravel_index(i, T.moment.shape)] = float(i)
        zero_flux = T.input_flux == 0
        input_flux_nans = T.input_flux
        input_flux_nans[zero_flux] = np.nan
        if np.any(T.flux[zero_flux] != 0):
            raise ValueError("There are non-zero fluxes where the input flux is zero!")
        T.magnification = T.flux / input_flux_nans
        T.magnification[zero_flux] = 0

        T.update_notes()
        T.complete = True
        T.calc_time = timer() - t0

        return T

    return factory


@pytest.fixture
def counting_matrix(counting_matrix_factory):
    return counting_matrix_factory()


@pytest.fixture
def counting_matrix_dist1(counting_matrix_factory):
    return counting_matrix_factory(min_dist=0.5, max_dist=1.0)


@pytest.fixture
def counting_matrix_dist2(counting_matrix_factory):
    return counting_matrix_factory(min_dist=1.0, max_dist=2.0)
