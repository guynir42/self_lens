import sys
from os import path
import pytest

import pickle
import numpy as np
import xarray as xr
from timeit import default_timer as timer

sys.path.append(path.dirname(path.abspath(__file__)))
from src.transfer_matrix import TransferMatrix
from src.grid_scan import Grid
from src.simulator import Simulator
from src.survey import Survey


@pytest.fixture(scope="module")
def grid():
    g = Grid()
    g.setup_demo_scan(wd_lens=True)
    always_recalculate = True
    if always_recalculate or not path.isfile("test_dataset.nc"):
        g.run_simulation(keep_systems=True)
        g.dataset.to_netcdf("test_dataset.nc")
        with open("test_systems.pickle", "wb") as file:
            pickle.dump(g.systems, file)
    else:
        g.dataset = xr.load_dataset("test_grid_scan.nc")

    return g


@pytest.fixture
def ztf():
    return Survey("ZTF")


@pytest.fixture
def sim():
    return Simulator()


@pytest.fixture
def matrix():
    return TransferMatrix.from_file("matrix.npz")


@pytest.fixture
def matrix_large():
    return TransferMatrix.from_file("saved/matrix_SR1.000-5.000_D0.000-20.000.npz")


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

        T.magnification = T.flux / T.input_flux

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
