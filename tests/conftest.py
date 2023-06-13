import sys
import os
import pytest
import time

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


@pytest.fixture
def simulation_dataset():
    CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    survey_names = ["ZTF", "TESS", "LSST", "CURIOS", "CURIOS_ARRAY", "LAST"]

    ds = None
    for survey in survey_names:
        filename = os.path.join(CODE_ROOT, f"saved/simulate_{survey}_WD.nc")
        new_ds = xr.load_dataset(filename, decode_times=False)
        for name in ["lens_temp", "star_temp"]:
            new_ds[name] = np.round(new_ds[name])

        if ds is None:
            ds = new_ds
        else:
            ds = xr.concat([ds, new_ds], dim="survey")


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
