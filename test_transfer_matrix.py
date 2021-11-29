import sys
from os import path

import numpy as np
from timeit import default_timer as timer
import copy

import pytest
import matplotlib.pyplot as plt
sys.path.append(path.dirname(path.abspath(__file__)))

import transfer_matrix


def test_convergence():
    p1 = 1e3
    p2 = 1e4
    p3 = 1e6
    d = np.linspace(1, 2, 5)
    r = 0.005
    mag1 = transfer_matrix.radial_lightcurve(distances=d, source_radius=r, pixels=p1)
    mag2 = transfer_matrix.radial_lightcurve(distances=d, source_radius=r, pixels=p2)
    mag3 = transfer_matrix.radial_lightcurve(distances=d, source_radius=r, pixels=p3, circle_points=p3)
    mag_ps = transfer_matrix.point_source_approximation(d)

    try:
        assert np.all(abs(mag2 - mag3) < 0.01), f'mag of at least one point between med res and high res ' \
                                                f'({np.max(abs(mag2 - mag3))}) exceeds threshold'
        assert np.all(abs(mag2 - mag3) < 0.01), f'mag of at least one point between high res and point source' \
                                                f'({np.max(abs(mag3 - mag_ps))}) exceeds threshold'

        assert np.any(abs(mag1 - mag3) > 0.01), 'low res mag and high res mag should be quite different.'

    except:
        plt.plot(d, mag1, '-x', label=f'pixels= {p1}')
        plt.plot(d, mag2, '-+', label=f'pixels= {p2}')
        plt.plot(d, mag3, '-o', label=f'pixels= {p3}')
        plt.plot(d, mag_ps, '-', label='point source')
        plt.yscale('log')
        plt.legend()
        plt.show(block=True)
        raise


def test_matrix_vs_direct_calculation(matrix):

    source_radii = matrix.source_radii[[1, 5, 15]]

    d = matrix.distances[::10]
    mu1 = np.zeros((len(source_radii), d.size))
    mu2 = np.zeros((len(source_radii), d.size))

    # get a reference radial curve from direct calculation:
    for i, sr in enumerate(source_radii):
        mu1[i, :] = transfer_matrix.radial_lightcurve(distances=d, source_radius=sr, occulter_radius=0)

    # get the same curves from the transfer matrix
    for i, sr in enumerate(source_radii):
        mu2[i, :] = matrix.radial_lightcurve(source=sr, distances=d, occulter_radius=0, get_offsets=False)

    if not np.all(abs(mu1-mu2)/ mu1 < 0.01):  # maintain precision to within 1%
        plt.legend()
        plt.cla()
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu2[i, :], '-o', label=f'source radius= {sr} (matrix)')

        plt.gca().set_prop_cycle(None)
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu1[i, :], '--x', label=f'source radius= {sr} (direct)')

        plt.xlabel('Distance [Einstein radii]')
        plt.ylabel('Magnification')
        plt.legend()
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mu1-mu2) / mu1)}')

    # now do the same test with non-zero occulter radius
    occulter = 0.8
    # get a reference radial curve from direct calculation:
    for i, sr in enumerate(source_radii):
        mu1[i, :] = transfer_matrix.radial_lightcurve(distances=d, source_radius=sr, occulter_radius=occulter)

    # get the same curves from the transfer matrix
    for i, sr in enumerate(source_radii):
        mu2[i, :] = matrix.radial_lightcurve(source=sr, distances=d, occulter_radius=occulter, get_offsets=False)

    if not np.all(abs(mu1-mu2) / mu1 < 0.01):  # maintain precision to within 1%
        plt.legend()
        plt.cla()
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu2[i, :], '-o', label=f'source radius= {sr} (matrix)')

        plt.gca().set_prop_cycle(None)
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu1[i, :], '--x', label=f'source radius= {sr} (direct)')

        plt.xlabel('Distance [Einstein radii]')
        plt.ylabel('Magnification')
        plt.legend()
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mu1-mu2) / mu1)}')


def test_matrix_interpolation(matrix):

    source_radii = np.copy(matrix.source_radii[[7, 14]])
    source_radii += np.random.random(source_radii.shape) * matrix.step_source  # offset a little bit

    d = np.copy(matrix.distances[0:30])
    d += np.random.random(d.shape) * matrix.step_dist  # offset a little bit
    mu1 = np.zeros((len(source_radii), d.size))
    mu2 = np.zeros((len(source_radii), d.size))

    # get a reference radial curve from direct calculation:
    for i, sr in enumerate(source_radii):
        mu1[i, :] = transfer_matrix.radial_lightcurve(distances=d, source_radius=sr, occulter_radius=0)

    # get the same curves from the transfer matrix
    for i, sr in enumerate(source_radii):
        mu2[i, :] = matrix.radial_lightcurve(source=sr, distances=d, occulter_radius=0, get_offsets=False)

    if not np.all(abs(mu1 - mu2) / mu1 < 0.05):  # maintain precision to within 5%
        plt.cla()

        for i, sr in enumerate(source_radii):
            plt.plot(d, mu2[i, :], '-o', label=f'source radius= {sr} (matrix)')

        plt.gca().set_prop_cycle(None)

        for i, sr in enumerate(source_radii):
            plt.plot(d, mu1[i, :], '-x', label=f'source radius= {sr} (direct)')

        plt.xlabel('Distance [Einstein radii]')
        plt.ylabel('Magnification')
        plt.legend()
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mu1 - mu2) / mu1)}')


def test_analytic_solution():
    # refer to: https://academic.oup.com/mnras/article/411/3/1863/972908

    # start with the simple case of point-lens (case III in that paper):
    # assume source is bigger than lens by 3:
    source = 3
    occulter = 0

    # analytical formula:
    expected_magnification = np.sqrt(1 + 4 / source ** 2)

    # using our code:
    calculated_magnification = transfer_matrix.single_geometry(
        distance=0,
        source_radius=source,
        occulter_radius=occulter,
    )

    print(f'Source= {source} | occulter= {occulter} | '
          f'Magnification: analytical= {expected_magnification}, calculated= {calculated_magnification}')

    assert abs(calculated_magnification - expected_magnification) / expected_magnification < 0.01  # 1% accuracy

    # assume source is smaller than lens by 2.5:
    source = 0.4
    occulter = 0

    # analytical formula
    expected_magnification = np.sqrt(1 + 4 / source ** 2)

    # using our code:
    calculated_magnification = transfer_matrix.single_geometry(
        distance=0,
        source_radius=source,
        occulter_radius=occulter,
    )

    print(f'Source= {source} | occulter= {occulter} | '
          f'Magnification: analytical= {expected_magnification}, calculated= {calculated_magnification}')

    assert abs(calculated_magnification - expected_magnification) / expected_magnification < 0.01  # 1% accuracy

    # now lets do the case where there is a non-zero sized occulter (case II):
    source = 3
    occulter = 0.6

    # analytical formula:
    beta_L = occulter - 1 / occulter
    expected_magnification = 0.5 * (1 - (beta_L / source) ** 2
                                    + np.sqrt(1 + 4 / source ** 2)
                                    - (beta_L / source) * np.sqrt((beta_L / source) ** 2 + 4 / source ** 2))

    # using our code:
    calculated_magnification = transfer_matrix.single_geometry(
        distance=0,
        source_radius=source,
        occulter_radius=occulter,
    )

    print(f'Source= {source} | occulter= {occulter} '
          f'Magnification: analytical= {expected_magnification} | calculated= {calculated_magnification}')

    assert abs(calculated_magnification - expected_magnification) / expected_magnification < 0.01  # 1% accuracy

    # now lets do the case where the occulter is larger than the lens (case III):
    source = 3
    occulter = 1.8

    # analytical formula:
    beta_L = occulter - 1 / occulter
    expected_magnification = 0.5 * (1 - (beta_L / source) ** 2
                                    + np.sqrt(1 + 4 / source ** 2)
                                    - (beta_L / source) * np.sqrt((beta_L / source) ** 2 + 4 / source ** 2))

    # using our code:
    calculated_magnification = transfer_matrix.single_geometry(
        distance=0,
        source_radius=source,
        occulter_radius=occulter,
    )

    print(f'Source= {source} | occulter= {occulter} '
          f'Magnification: analytical= {expected_magnification} | calculated= {calculated_magnification}')

    assert abs(calculated_magnification - expected_magnification) / expected_magnification < 0.01  # 1% accuracy


def test_numerical_calculation():
    # ref: https://ui.adsabs.harvard.edu/abs/2002A%26A...394..489B/abstract
    # looking at equation 4 for a point source we should get:
    distance = 0.5
    magnification_point_source = 0.5 * (np.sqrt(1 + 4 / distance ** 2) + 1 / np.sqrt(1 + 4 / distance ** 2))
    mag = transfer_matrix.single_geometry(distance=distance, source_radius=0.005, pixels=1e6)

    assert abs(mag-magnification_point_source) < 0.01  # 1% precision

    # now asses equation 6
    distance = 0.8
    source = 0.8
    beta = 1 / source
    magnification_disk_touching = 2 / np.pi * np.sqrt(1 + 4 * beta ** 2)
    mag = transfer_matrix.single_geometry(distance=distance, source_radius=source)

    assert abs(mag - magnification_disk_touching) < 0.25  # close enough


def test_elliptical_large_source(matrix_large):

    # test a large source against the direct calculation
    source_radius = 10.0
    occulter_radius = 0.0
    d = np.linspace(0, source_radius * 1.5, 18)
    mag_approx = transfer_matrix.large_source_approximation(d, source_radius, occulter_radius)
    mag_direct = transfer_matrix.radial_lightcurve(d, source_radius, occulter_radius)

    if not np.all(np.abs(mag_direct - mag_approx) < 0.01):
        plt.cla()
        plt.plot(d, mag_approx, '-o', label='approx')
        plt.plot(d, mag_direct, '-x', label='direct')
        plt.legend()
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mag_direct - mag_approx))}')

    # add an occulter
    occulter_radius = 2.0
    mag_approx = transfer_matrix.large_source_approximation(d, source_radius, occulter_radius)
    mag_direct = transfer_matrix.radial_lightcurve(d, source_radius, occulter_radius)

    if not np.all(np.abs(mag_direct - mag_approx) < 0.01):
        plt.cla()
        plt.plot(d, mag_approx, '-o', label='approx')
        plt.plot(d, mag_direct, '-x', label='direct')
        plt.legend()
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mag_direct - mag_approx))}')

    # test a medium source against the matrix
    source_radius = 2.0
    occulter_radius = 0.0
    d = np.linspace(0, source_radius * 1.5, 18)
    mag_approx = transfer_matrix.large_source_approximation(d, source_radius, occulter_radius)
    mag_direct = matrix_large.radial_lightcurve(source_radius, d, occulter_radius)

    # still ballpark but not very accurate for small sources
    if not np.all(np.abs(mag_direct - mag_approx) < 0.3) or not np.all(np.abs(mag_direct - mag_approx) > 0.01):
        plt.cla()
        plt.plot(d, mag_approx, '-o', label='approx')
        plt.plot(d, mag_direct, '-x', label='direct')
        plt.legend()
        plt.show(block=True)
        if not np.all(np.abs(mag_direct - mag_approx) < 0.3):
            raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mag_direct - mag_approx))}')
        if not np.all(np.abs(mag_direct - mag_approx) > 0.01):
            raise ValueError(f'At least one point shows a different magnification of {np.min(abs(mag_direct - mag_approx))}')

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


# this doesn't work and I don't think we need it
def test_matrix_dwindled(matrix):
    matrix2 = copy.deepcopy(matrix)
    matrix2.dwindle_data()

    # get the radii not covered in the "dwindled" matrix
    source_radii = matrix.source_radii[::2]
    source_radii = [0.8]
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

    if not np.all(abs(mu1-mu2) < 0.1):  # maintain precision to within 10%
        plt.legend()
        plt.cla()
        for i, sr in enumerate(source_radii):
            plt.plot(d, mu2[i, :], '-o', label=f'source radius= {sr} (matrix)')
            plt.plot(d, mu1[i, :], '-x', label=f'source radius= {sr} (direct)')
        plt.show(block=True)

        raise ValueError(f'At least one point shows a different magnification of {np.max(abs(mu1-mu2))}')

