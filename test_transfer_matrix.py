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


def test_addition(counting_matrix, counting_matrix_dist1, counting_matrix_dist2):
    # check the null combination of matrix with new (empty) matrix
    new_matrix = counting_matrix + transfer_matrix.TransferMatrix()

    assert new_matrix.min_dist == counting_matrix.min_dist
    assert new_matrix.max_dist == counting_matrix.max_dist

    assert np.array_equal(
        new_matrix.distances,
        counting_matrix.distances,
    )

    assert np.array_equal(
        new_matrix.flux[0, 0, :],
        np.array([0, 1, 2, 3, 4, 5])
    )

    # check the same works for other order of addition
    new_matrix = transfer_matrix.TransferMatrix() + counting_matrix

    assert new_matrix.min_dist == counting_matrix.min_dist
    assert new_matrix.max_dist == counting_matrix.max_dist

    assert np.array_equal(
        new_matrix.distances,
        counting_matrix.distances,
    )

    assert np.array_equal(
        new_matrix.flux[0, 0, :],
        np.array([0, 1, 2, 3, 4, 5])
    )

    new_matrix = counting_matrix + counting_matrix_dist1

    print(f'matrix1.flux= {counting_matrix.flux[0, 0, :]}')
    print(f'matrix2.flux= {counting_matrix_dist1.flux[0, 0, :]}')
    print(f'new_matrix.flux= {new_matrix.flux[0, 0, :]}')
    print()
    print(f'matrix1.distances= {counting_matrix.distances}')
    print(f'matrix2.distances= {counting_matrix_dist1.distances}')
    print(f'new_matrix.distances= {new_matrix.distances}')

    assert new_matrix.min_dist == counting_matrix.min_dist
    assert new_matrix.max_dist == counting_matrix_dist1.max_dist

    assert np.array_equal(
        new_matrix.distances,
        np.append(counting_matrix.distances[:-1], counting_matrix_dist1.distances)
    )

    assert np.array_equal(
        new_matrix.flux[0, 0, :],
        np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5])
    )

    new_matrix = counting_matrix + counting_matrix_dist2

    assert new_matrix.min_dist == counting_matrix.min_dist
    assert new_matrix.max_dist == counting_matrix_dist2.max_dist

    assert np.array_equal(
        new_matrix.distances,
        np.append(counting_matrix.distances, counting_matrix_dist2.distances)
    )

    assert np.array_equal(
        new_matrix.flux[0, 0, :],
        np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )

    # check that inconsistent parameters (except distances) cannot be added
    counting_matrix.occulter_radii[-1] += 0.1
    with pytest.raises(ValueError, match='Values for "occulter_radii" are inconsistent!'):
        counting_matrix + counting_matrix_dist2



def test_star_profiles(matrix):
    pass