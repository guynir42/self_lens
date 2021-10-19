import numpy as np
from timeit import default_timer as timer
import copy

import pytest
import matplotlib.pyplot as plt
import transfer_matrix


def test_matrix_vs_direct_calculation(matrix):

    source_radii = matrix.source_radii[[3, 9, 15]]

    d = matrix.distances
    mu1 = np.zeros((len(source_radii), d.size))
    mu2 = np.zeros((len(source_radii), d.size))

    # get a reference radial curve from direct calculation:
    for i, sr in enumerate(source_radii):
        mu1[i, :] = transfer_matrix.radial_lightcurve(sr, distances=d, occulter_radius=0, plotting=False)

    # get the same curves from the transfer matrix
    for i, sr in enumerate(source_radii):
        mu2[i, :] = matrix.radial_lightcurve(source=sr, distances=d, occulter_radius=0, get_offsets=False)

    if not np.all(abs(mu1-mu2) < 0.01):  # maintain precision to within 1%
        plt.legend()
        plt.cla()
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu2[i, :], '-o', label=f'source radius= {sr} (matrix)')
            plt.plot(d, mu1[i, :], '-x', label=f'source radius= {sr} (direct)')
        plt.legend()
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mu1-mu2))}')


def test_matrix_interpolation(matrix):

    source_radii = matrix.source_radii[[3, 9, 15]] + np.random.rand() * matrix.step_source  # offset a little bit

    d = matrix.distances + np.random.rand() * matrix.step_dist  # offset a little bit
    mu1 = np.zeros((len(source_radii), d.size))
    mu2 = np.zeros((len(source_radii), d.size))

    # get a reference radial curve from direct calculation:
    for i, sr in enumerate(source_radii):
        mu1[i, :] = transfer_matrix.radial_lightcurve(sr, distances=d, occulter_radius=0, plotting=False)

    # get the same curves from the transfer matrix
    for i, sr in enumerate(source_radii):
        mu2[i, :] = matrix.radial_lightcurve(source=sr, distances=d, occulter_radius=0, get_offsets=False)

    if not np.all(abs(mu1 - mu2) < 0.01):  # maintain precision to within 1%
        plt.legend()
        plt.cla()
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu2[i, :], '-o', label=f'source radius= {sr} (matrix)')
            plt.plot(d, mu1[i, :], '-x', label=f'source radius= {sr} (direct)')
        plt.legend()
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mu1 - mu2))}')


def test_matrix_vs_analytic_solution(matrix):
    # refer to: https://academic.oup.com/mnras/article/411/3/1863/972908
    pass


def test_matrix_dwindled(matrix):
    matrix2 = copy.deepcopy(matrix)
    matrix2.dwindle_data()

    # get the radii not covered in the "dwindled" matrix
    source_radii = matrix.source_radii[::2]
    source_radii = [1.2]
    d = matrix.distances

    # preallocate
    mu1 = np.zeros((len(source_radii), d.size))
    mu2 = np.zeros((len(source_radii), d.size))

    # these source radii were directly calculated in the matrix
    for i, sr in enumerate(source_radii):
        mu1[i, :] = matrix.radial_lightcurve(source=sr, distances=d, occulter_radius=0, get_offsets=False)

    # these need to be interpolated in the "dwindled" matrix
    for i, sr in enumerate(source_radii):
        mu2[i, :] = matrix2.radial_lightcurve(source=sr, distances=d, occulter_radius=0, get_offsets=False)

    if not np.all(abs(mu1-mu2) < 0.01):  # maintain precision to within 1%
        plt.legend()
        plt.cla()
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu2[i, :], '-o', label=f'source radius= {sr} (matrix)')
            plt.plot(d, mu1[i, :], '-x', label=f'source radius= {sr} (direct)')
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mu1-mu2))}')

def test_star_profiles(matrix):
    pass