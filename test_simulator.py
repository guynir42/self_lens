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
