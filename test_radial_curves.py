import numpy as np
from timeit import default_timer as timer

import pytest
import matplotlib.pyplot as plt
import transfer_matrix


def test_matrix_source_radius(matrix):
    # get a reference radial curve from direct calculation:
    # source_radii = [0.5, 1.0, 1.5]
    source_radii = [1.2]
    d = matrix.distances
    f1 = np.zeros((len(source_radii), d.size))
    f2 = np.zeros((len(source_radii), d.size))

    for i, sr in enumerate(source_radii):
        f1[i, :] = transfer_matrix.radial_lightcurve(sr, distances=d, occulter_radius=0, plotting=False)

    for i, sr in enumerate(source_radii):
        f2[i, :] = matrix.radial_lightcurve(source=sr, distances=d, occulter_radius=0, get_offsets=False)

    plt.cla()
    for i, sr in enumerate(source_radii):
        plt.plot(d, f2[i, :], '-o', label=f'source radius= {sr} (matrix)')
        plt.plot(d, f1[i, :], '-x', label=f'source radius= {sr} (direct)')


    plt.legend()
    plt.show(block=True)



def test_matrix_interpolation(matrix):
    pass


def test_star_profiles(matrix):
    pass