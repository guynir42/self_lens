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
sim.star_mass = 0.6
sim.star_temp = 10000
sim.star_type = 'WD'
sim.lens_mass = 3.0
sim.declination = 0.0
sim.semimajor_axis = 0.1

ztf = survey.Survey('ZTF')
ztf.distances = np.array([100])

dec = np.linspace(0, 0.1, 1000)
t_flare = np.zeros(dec.shape)
prob = np.zeros(dec.shape)

for i, d in enumerate(dec):
    sim.timestamps = None
    sim.calculate(declination=d)
    ztf.apply_detection_statistics(sim.syst)
    prob[i] = sim.syst.visit_prob['ZTF'][0]
    t_flare[i] = sim.syst.flare_durations['ZTF'][0]

plt.plot(dec, prob, '-*')
plt.show()

