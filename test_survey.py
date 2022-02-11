import sys
from os import path

import numpy as np
from timeit import default_timer as timer
import copy
import scipy

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

    (best_prob, total_prob) = ztf.visit_prob_all_declinations(sim)

    # get a back of the envelope estimate for the total probability
    total_prob_estimate = sim.visit_prob_all_declinations_estimate()

    print(f'max_dec= {sim.declination:.3f} | P(best)= {best_prob:.2e} | P(total)= {total_prob:.2e} | P(total, est)= {total_prob_estimate:.2e}')

    assert abs(np.log10(total_prob / total_prob_estimate)) < 0.2  # should be within a few percent

    # an object with a period of about 2 days
    sim.star_mass = 0.6
    sim.star_temp = 10000
    sim.star_type = 'WD'
    sim.lens_mass = 3.0
    sim.declination = 0.0
    sim.semimajor_axis = 0.05

    # syst = sim.syst  # this system is at dec=0
    # ztf.apply_detection_statistics(syst)
    # flare_to_visit_ratio = syst.flare_prob['ZTF']

    (best_prob, total_prob) = ztf.visit_prob_all_declinations(sim)

    # get a back of the envelope estimate for the total probability
    total_prob_estimate = sim.visit_prob_all_declinations_estimate()

    print(f'max_dec= {sim.declination:.3f} | prob(dec)= {np.sin(np.deg2rad(sim.declination))} | '
          f'P(best)= {best_prob:.2e} | P(total)= {total_prob:.2e} | P(total, est)= {total_prob_estimate:.2e}')
    assert abs(np.log10(total_prob / total_prob_estimate)) < 0.2  # should be within a few percent

    # an object with a period of about 1/2 year
    sim.star_mass = 0.6
    sim.star_temp = 10000
    sim.star_type = 'WD'
    sim.lens_mass = 3.0
    sim.declination = 0.0
    sim.semimajor_axis = 1

    (best_prob, total_prob) = ztf.visit_prob_all_declinations(sim)

    # get a back of the envelope estimate for the total probability
    total_prob_estimate = sim.visit_prob_all_declinations_estimate()

    print(f'max_dec= {sim.declination:.3f} | P(best)= {best_prob:.2e} | P(total)= {total_prob:.2e} | P(total, est)= {total_prob_estimate:.2e}')
    assert abs(np.log10(total_prob / total_prob_estimate)) < 1  # should be within an order of magnitude


def test_individual_systems(sim, ztf):
    ztf.distances = np.array([100])  # limit to a single distance to keep it simple

    # a system with a fairly bright peak magnification
    sim.star_mass = 0.6
    sim.star_temp = 10000
    sim.star_type = 'WD'
    sim.lens_mass = 30.0
    sim.declination = 0.0
    sim.semimajor_axis = 0.001  # about 180s orbit

    sim.calculate()
    s1 = sim.syst
    print(f'period= {s1.orbital_period * 3600}s | fwhm= {sim.fwhm}s | source_size= {s1.source_size} ')

    ztf.apply_detection_statistics(s1)

    print(f'peak magnification= {np.max(s1.magnifications)} | flare_prob= {s1.flare_prob["ZTF"]} '
          f'| visit_prob= {s1.visit_prob["ZTF"]} | num_detections= {s1.num_detections["ZTF"]}')

    # probability to hit the flare during a single visit
    timing_prob = sim.fwhm / (s1.orbital_period * 3600)
    assert abs(np.log10(timing_prob / s1.visit_prob['ZTF'])) < 1  # an estimate to within order of magnitude

    # assume flare duration is shorter than exposure time
    diluted_flare_magnification = sim.fwhm / ztf.exposure_time * (np.max(s1.magnifications) - 1) + 1
    flare_snr = diluted_flare_magnification / ztf.precision
    flare_prob = 0.5 * (1 + scipy.special.erf(flare_snr - ztf.threshold))
    print(f'flare_prob= {flare_prob} | full calculation= {s1.flare_prob["ZTF"]}')
    assert abs(flare_prob - s1.flare_prob['ZTF']) < 0.01

    # with more exposures in a series we get a much better chance to detect any flares, and also
    ztf.series_length = 10
    sim.calculate()  # this will regenerate the same system but in a new object
    s2 = sim.syst
    ztf.apply_detection_statistics(s2)

    print(f'peak magnification= {np.max(s2.magnifications)} | flare_prob= {s2.flare_prob["ZTF"]} '
          f'| visit_prob= {s2.visit_prob["ZTF"]} | num_detections= {s2.num_detections["ZTF"]}')

    assert 1 - s2.visit_prob['ZTF'] < 0.01
    assert s2.num_detections['ZTF'] > 1


def test_probabilities(sim, ztf):
    sim.calculate(
        lens_mass=2.0,
        star_mass=0.5,
        star_temp=7000,
        declination=0.0,
        semimajor_axis=0.001,
    )

    # the exposure time is much shorter than the flare time
    ztf.exposure_time = sim.fwhm / 5

    # fit the survey precision to this event's magnification at peak
    ztf.precision = (max(sim.magnifications) - 1) / 3
    ztf.threshold = 3
    ztf.distances = ztf.distances[0:1]

    ztf.apply_detection_statistics(sim.syst)
    # the S/N should be similar to the threshold, so 50% detection rate
    assert abs(sim.syst.flare_prob['ZTF'][0] - 0.5) < 0.01

    # the chance of hitting the flare in the total event time is much smaller
    duty_cycle = sim.fwhm / (sim.orbital_period * 3600)
    print(f'precision= {ztf.precision:.4f} | exposure_time= {ztf.exposure_time:.1f} | duty_cycle= {duty_cycle:.2e}')
    assert abs(sim.syst.visit_prob['ZTF'][0] - 0.5 * duty_cycle) < 0.01

    # now what happens when we use multiple exposures in a row
    ztf.series_length = 5
    dt = np.diff(sim.timestamps)[0]
    oversampling = int(ztf.exposure_time / dt)  # how badly is the LC sampled?
    mag = sim.magnifications - 1
    if oversampling > 1:
        mag = mag[oversampling//2:-1:oversampling]  # closer to the native sampling

    # matched filtering on the full LC should give this much more S/N
    increased_snr = np.sqrt(np.sum(mag ** 2)) / np.max(mag)
    ztf.threshold *= increased_snr  # should turn it back to borderline detection

    ztf.apply_detection_statistics(sim.syst)
    print(f'series= {ztf.series_length} | '
          f'flare_prob= {sim.syst.flare_prob["ZTF"][0]} | '
          f'visit_prob= {sim.syst.visit_prob["ZTF"][0]} | '
          f'num_detections= {sim.syst.num_detections["ZTF"][0]}')

    assert abs(sim.syst.flare_prob['ZTF'][0] - 0.5) < 0.2  # close to 50% detection
    assert abs(sim.syst.visit_prob['ZTF'][0] - 0.5 * duty_cycle) < 0.2
    assert sim.syst.visit_prob['ZTF'][0] == sim.syst.num_detections['ZTF'][0]

    # test very long series that cover entire orbit
    num_exposures_per_orbit = int(sim.orbital_period * 3600 / ztf.exposure_time)
    ztf.series_length = 5 * num_exposures_per_orbit  # can now see up to 5 flares
    ztf.apply_detection_statistics(sim.syst)
    print(f'series= {ztf.series_length} | '
          f'flare_prob= {sim.syst.flare_prob["ZTF"][0]} | '
          f'visit_prob= {sim.syst.visit_prob["ZTF"][0]} | '
          f'num_detections= {sim.syst.num_detections["ZTF"][0]}')

    assert abs(sim.syst.flare_prob['ZTF'][0] - 0.5) < 0.2  # close to 50% detection
    assert 1 - sim.syst.visit_prob['ZTF'][0] < 0.2
    assert sim.syst.num_detections['ZTF'][0] > 1  # should be able to see multiple flares


