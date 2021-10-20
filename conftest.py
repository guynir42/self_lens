import sys
from os import path
import pytest

import numpy as np
from timeit import default_timer as timer

sys.path.append(path.dirname(path.abspath(__file__)))
import transfer_matrix


@pytest.fixture
def matrix():
    T = transfer_matrix.TransferMatrix()
    T.load('matrix.npz')
    return T


@pytest.fixture
def matrix_high_res():
    T = transfer_matrix.TransferMatrix()
    T.load('saved/matrix_high_res.npz')
    return T


@pytest.fixture
def counting_matrix_factory():
    def factory(min_dist=0,
                max_dist=0.5,
                step_dist=0.1,
                min_source=0.1,
                max_source=0.3,
                step_source=0.1,
                max_occulter=0.3,
                step_occulter=0.3,
                ):
        T = transfer_matrix.TransferMatrix()
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
            T.moment[np.unravel_index(i, T.moment.shape)] = float(i)

        for i in range(T.input_flux.size):
            T.input_flux[i] = float(i)

        T.magnification = T.flux / np.expand_dims(T.input_flux, axis=[0, 2])

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
