import sys
from os import path

import numpy as np
from timeit import default_timer as timer
import copy

import pytest
import matplotlib.pyplot as plt

sys.path.append(path.dirname(path.abspath(__file__)))

import simulator
import survey


def test_back_of_the_envelope(sim, ztf):
    ztf.distances = np.array([100])  # limit to a single distance to keep it simple

    # an object with a period less than 2 hours
    sim.star_mass = 0.6
    sim.star_temp = 10000
    sim.star_type = 'WD'
    sim.lens_mass = 3.0
    sim.declination = 0.0
    sim.semimajor_axis = 0.005

    (peak_prob, total_prob) = ztf.visit_prob_all_declinations(sim)

    # get a back of the envelope estimate for the total probability
    total_prob_estimate = sim.visit_prob_all_declinations_estimate()

    print(f'max_dec= {sim.declination:.3f} | P(best)= {peak_prob:.2e} | P(total)= {total_prob:.2e} | P(total, est)= {total_prob_estimate:.2e}')

    assert abs(np.log10(total_prob / total_prob_estimate)) < 1  # should be within an order of magnitude

    # an object with a period of about 2 days
    sim.star_mass = 0.6
    sim.star_temp = 10000
    sim.star_type = 'WD'
    sim.lens_mass = 3.0
    sim.declination = 0.0
    sim.semimajor_axis = 0.05

    (peak_prob, total_prob) = ztf.visit_prob_all_declinations(sim)

    # get a back of the envelope estimate for the total probability
    total_prob_estimate = sim.visit_prob_all_declinations_estimate()

    print(f'max_dec= {sim.declination:.3f} | P(best)= {peak_prob:.2e} | P(total)= {total_prob:.2e} | P(total, est)= {total_prob_estimate:.2e}')
    assert abs(np.log10(total_prob / total_prob_estimate)) < 1  # should be within an order of magnitude

    # an object with a period of about 1/2 year
    sim.star_mass = 0.6
    sim.star_temp = 10000
    sim.star_type = 'WD'
    sim.lens_mass = 3.0
    sim.declination = 0.0
    sim.semimajor_axis = 1

    (peak_prob, total_prob) = ztf.visit_prob_all_declinations(sim)

    # get a back of the envelope estimate for the total probability
    total_prob_estimate = sim.visit_prob_all_declinations_estimate()

    print(f'max_dec= {sim.declination:.3f} | P(best)= {peak_prob:.2e} | P(total)= {total_prob:.2e} | P(total, est)= {total_prob_estimate:.2e}')
    assert abs(np.log10(total_prob / total_prob_estimate)) < 1  # should be within an order of magnitude

