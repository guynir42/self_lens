import sys
from os import path

import numpy as np
from timeit import default_timer as timer
import copy
import scipy.integrate as integ

import pytest
import matplotlib.pyplot as plt

from src.simulator import default_filter


def test_black_body(sim):
    # make sure the total black body over all frequencies is consistent with Stephan Boltzmann
    sim.calculate(star_mass=1, star_size=1, star_type="MS", star_temp=5778, lens_mass=30)
    c = 299792458
    f1 = c / 5000e-9  # 5000nm is the low frequency edge
    f2 = c / 50e-9  # 50nm is the high frequency edge
    total_flux = integ.quad(sim.syst.black_body, f1, f2, args=(5778, False))[0]
    total_flux *= 4 * np.pi * 696e8**2  # surface area of the star/sun in cm^2
    print(f"integral= {total_flux} | Stephan Boltzmann= {sim.star_flux} | ratio= {sim.star_flux / total_flux}")
    assert abs((total_flux - sim.star_flux) / (total_flux + sim.star_flux)) < 0.01


# @pytest.mark.xfail
def test_apparent_magnitudes(sim):
    # make a sun-like star and check we get the right magnitude (add BH companion that doesn't radiate)
    sim.calculate(star_mass=1, star_size=1, star_type="MS", star_temp=5778, lens_mass=30)
    filter_pars = default_filter("V")
    # filter_pars = (550, 300)

    # magnitudes at 10pc
    mag_bol = sim.syst.bolometric_mag()
    mag_ab = sim.syst.ab_mag(*filter_pars)

    # print(f"Bol mag= {mag_bol:.4g} | AB mag= {mag_ab:.4g}")
    assert abs(mag_bol - 4.74) < 0.1
    assert abs(mag_ab - 4.83) < 0.1

    # now try an example WD from the Gaia data (sample 2)
    sim.calculate(star_mass=0.64, star_size=None, star_type="WD", star_temp=8800, lens_mass=30)
    filter_pars = (625, 450)  # Gaia G band

    mag_bol = sim.syst.apply_distance(sim.syst.bolometric_mag(), 195)  # magnitude at 195pc
    mag_ab = sim.syst.apply_distance(sim.syst.ab_mag(*filter_pars), 195)  # magnitude at 195pc
    # print(f"Bol mag= {mag_bol:.4g} | AB mag= {mag_ab:.4g}")
    assert abs(mag_ab - 19.36) < 0.2  # from Gaia we get mag_g is 19.36

    # try another, brighter WD
    sim.calculate(star_mass=0.59, star_size=None, star_type="WD", star_temp=7900, lens_mass=30)

    mag_bol = sim.syst.apply_distance(sim.syst.bolometric_mag(), 22)
    mag_ab = sim.syst.apply_distance(sim.syst.ab_mag(*filter_pars), 22)
    # print(f"Bol mag= {mag_bol:.4g} | AB mag= {mag_ab:.4g}")
    assert abs(mag_ab - 14.85) < 0.2  # from Gaia we get mag_g is 14.85

    # try the examples from https://ui.adsabs.harvard.edu/abs/2021ApJ...918L..14K/abstract
    # note that the magnitudes of SkyMapper are very different from "standard filters"
    # ref: https://skymapper.anu.edu.au/filter-transformations/
    # so maybe that explains the discrepancy?
    # example 1: J0338
    sim.calculate(
        star_mass=0.23,
        star_temp=18100,
        orbital_period=1836.1 / 3600,
        lens_mass=0.38,
        lens_temp=10000,
    )

    filter_pars = default_filter("R")
    mag_bol = sim.syst.apply_distance(sim.syst.bolometric_mag(), 533)
    mag_ab = sim.syst.apply_distance(sim.syst.ab_mag(*filter_pars), 533)
    print(f"Bol mag= {mag_bol:.4g} | AB mag= {mag_ab:.4g}")

    assert abs(mag_ab - 17.4) < 1  # why is this example off by almost 1 mag??

    # example 2: J0634
    sim.calculate(
        star_mass=0.452,
        star_temp=27300,
        orbital_period=1591.4 / 3600,
        lens_mass=0.209,
        lens_temp=10500,
    )

    mag_bol = sim.syst.apply_distance(sim.syst.bolometric_mag(), 435)
    mag_ab = sim.syst.apply_distance(sim.syst.ab_mag(*filter_pars), 435)
    print(f"Bol mag= {mag_bol:.4g} | AB mag= {mag_ab:.4g}")

    assert abs(mag_ab - 17.3) < 0.5  # why is this example off by almost 0.5 mag??


def test_max_magnification(sim):
    """
    Use the examples in https://ui.adsabs.harvard.edu/abs/1997ChPhL..14..155Q/abstract (page 4)
    to test if the peak magnification is consistent.
    """

    sim.star_size = 6400 / 700000  # source is Earth-size, in units of solar radii
    sim.lens_mass = 1  # solar mass
    sim.declination = 0
    sim.lens_size = 0
    sim.semimajor_axis = 700000 / 150e6  # distance is Solar radius (in units of AU)
    sim.calculate()

    assert np.abs(np.max(sim.magnifications) - 1.2) < 0.02

    sim.semimajor_axis = 1  # one AU
    sim.calculate()

    assert np.abs(np.max(sim.magnifications) - 9.4) < 0.02


def test_smoothness(sim):
    sim.star_mass = 0.6
    sim.lens_mass = 1.0
    sim.declination = 0

    sma = np.linspace(0.020, 0.025, 100)
    mag = []
    for a in sma:
        sim.semimajor_axis = a
        mag.append(max(sim.calculate()))

    # plt.plot(sma, mag, "-x")
    # plt.yscale("log")
    # plt.show(block=True)

    # making the semimajor axis larger should
    # consistently make the magnifications larger
    assert all(np.diff(mag) > 0)
