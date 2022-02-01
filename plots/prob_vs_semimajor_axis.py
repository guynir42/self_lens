import os
import sys
from os import path

import numpy as np
from timeit import default_timer as timer
import copy

import pytest
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import simulator
import survey

sim = simulator.Simulator()
sim.star_mass = 0.8
sim.star_temp = 10000
sim.star_type = 'WD'
sim.lens_mass = 5.0
sim.declination = 0.0

ztf = survey.Survey('ZTF')
ztf.distances = np.array([100])
ztf.exposure_time = 1

sma = np.geomspace(0.005, 1, 100)
prob_peak = np.zeros(sma.shape)
prob_total = np.zeros(sma.shape)
prob_est = np.zeros(sma.shape)
t_flare = np.zeros(sma.shape)

for i, a in enumerate(sma):
    sim.calculate(semimajor_axis=a, declination=0)
    ztf.apply_detection_statistics(sim.syst)
    t_flare[i] = sim.syst.flare_durations['ZTF'][0]
    (prob_peak[i], prob_total[i]) = ztf.visit_prob_all_declinations(sim)
    prob_est[i] = sim.visit_prob_all_declinations_estimate()


plt.plot(sma, prob_total)
plt.plot(sma, prob_est)
plt.yscale('log')
plt.xlabel('Semimajor axis [AU]')
plt.ylabel('Object detection probability')
plt.show()

