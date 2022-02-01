import sys
from os import path

import numpy as np
from timeit import default_timer as timer
import copy

import pytest
import matplotlib.pyplot as plt

sys.path.append(path.dirname(path.abspath(__file__)))

import simulator


def test_apparent_magnitudes(sim):
    # make a sun-like star and check we get the right magnitude (add BH companion that doesn't radiate)
    sim.calculate(star_mass=1, star_size=1, star_type='MS', star_temp=5778, lens_mass=30)
    filter_pars = simulator.default_filter('V')
    corr = sim.syst.bolometric_correction(*filter_pars)[0]
    mag = sim.syst.bolometric_mag(10)  # magnitude at 10pc

    assert abs(mag - 4.74) < 0.1

    # what about the bolometric corrections?

    # now try an example WD from the Gaia data (sample 2)
    sim.calculate(star_mass=0.64, star_size=None, star_type='WD', star_temp=8800, lens_mass=30)
    mag = sim.syst.bolometric_mag(200)  # magnitude at 200pc

    assert abs(mag - 19.36) < 0.5  # from Gaia we get mag_g is 19.36

    # try another, brighter WD
    sim.calculate(star_mass=0.59, star_size=None, star_type='WD', star_temp=7900, lens_mass=30)
    mag = sim.syst.bolometric_mag(22)  # magnitude at 22pc

    assert abs(mag - 14.85) < 0.5  # from Gaia we get mag_g is 14.85


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
