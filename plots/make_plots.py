import os
import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt

import transfer_matrix
import simulator
import survey

"""
This script should be used to produce all the plots needed for the paper.  
"""
# find the folder of the module
folder = path.dirname(path.abspath(transfer_matrix.__file__))
sys.path.append(path.dirname(folder))
os.chdir(folder)

## start by loading a matrix:
T = transfer_matrix.TransferMatrix.from_file(folder+'/matrix.npz')

## plot some example images

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, num=10)

# TODO: need to get unified single map for both images together, and pass the legend argument into single_geometry.

plt.sca(ax1)
transfer_matrix.single_geometry(source_radius=0.8, distance=0, occulter_radius=0, plotting=True, legend=False)

plt.sca(ax2)
transfer_matrix.single_geometry(source_radius=0.8, distance=0.4, occulter_radius=0, plotting=True, legend=False)

plt.sca(ax3)
transfer_matrix.single_geometry(source_radius=0.8, distance=1.6, occulter_radius=0, plotting=True, legend=True)


## setup a simulator
import simulator
sim = simulator.Simulator()

## peak magnification vs. orbital period

plt.rc('font', size=16)
sim.star_mass = 0.6
sim.star_temp = 7000
sim.declination = 0

fig = plt.figure(num=2, figsize=[18, 10])
plt.clf()
# ax = fig.add_axes([0.08, 0.08, 0.65, 0.87])
ax = fig.add_axes([0.08, 0.08, 0.87, 0.87])

thresholds = [0.03, 0.3, np.inf]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# for WD lenses
masses = np.array([0.3, 0.6, 1.0, 1.2])
styles = ['*', 'v', 's', 'p']
orbits = np.geomspace(0.0001, 1, 60)
peak_mags = np.zeros((len(masses), len(orbits)))
periods = np.zeros(peak_mags.shape)
prob_low = np.zeros(peak_mags.shape)
prob_high = np.zeros(peak_mags.shape)
roche = np.zeros(peak_mags.shape)

for i, m in enumerate(masses):
    t = 0
    sim.lens_mass = m
    for j, a in enumerate(orbits):
        sim.semimajor_axis = a
        mag = sim.calculate()
        periods[i, j] = sim.orbital_period * 3600
        peak_mags[i, j] = np.max(mag - 1)
        prob_low[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[0], threshold=1)
        prob_high[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[1], threshold=1)
        roche[i, j] = sim.roche_lobe / sim.star_size

        if peak_mags[i, j] > thresholds[t]:
            vertical = 'top' if i % 2 == 0 else 'bottom'
            horizontal = 'center' if i % 2 == 1 else 'left'
            prob = sim.best_prob_all_declinations_estimate(precision=thresholds[t], threshold=1)
            # text_str = f'P={prob:.2g}'
            text_str = f'1/{int(np.round(1 / prob, -1))}'
            ax.text(periods[i, j], peak_mags[i, j], text_str, va=vertical, ha=horizontal,
                    color='k', bbox=dict(facecolor=colors[i], alpha=0.8))
            t += 1

    ax.plot(periods[i], peak_mags[i], '-',
            markersize=10, marker=styles[i], color=colors[i],
            label=f'WD-WD ({sim.star_mass}$M_\u2609$, {sim.lens_mass}$M_\u2609$)')

    idx = roche[i] < 1
    ax.plot(periods[i, idx], peak_mags[i, idx], 'bo', fillstyle='none', markersize=12)

# NS and BH lenses
masses = np.array([1.5, 3, 10.0, 30.0])
colors = [np.ones(3) * c for c in [0.6, 0.4, 0.2, 0.0]]
styles = ['X', 'D', 'o', '^']
peak_mags = np.zeros((len(masses), len(orbits)))
periods = np.zeros(peak_mags.shape)
prob_low = np.zeros(peak_mags.shape)
prob_high = np.zeros(peak_mags.shape)
roche = np.zeros(peak_mags.shape)

for i, m in enumerate(masses):
    t = 0
    sim.lens_mass = m
    for j, a in enumerate(orbits):
        sim.semimajor_axis = a
        mag = sim.calculate()
        periods[i, j] = sim.orbital_period * 3600
        peak_mags[i, j] = np.max(mag - 1)
        prob_low[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[0], threshold=1)
        prob_high[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[1], threshold=1)
        roche[i, j] = sim.roche_lobe / sim.star_size

        if peak_mags[i, j] > thresholds[t]:
            prob = sim.best_prob_all_declinations_estimate(precision=thresholds[t], threshold=1)
            # text_str = f'P={prob:.2g}'
            text_str = f'1/{int(np.round(1 / prob))}'
            ax.text(periods[i, j], peak_mags[i, j], text_str, ha='right',
                    color='k', bbox=dict(facecolor='w', alpha=0.8))

            t += 1

    ax.plot(periods[i], peak_mags[i], '-',
        markersize=10, marker=styles[i], color=colors[i],
        label=f'WD-{sim.lens_type} ({sim.star_mass}$M_\u2609$, {sim.lens_mass}$M_\u2609$)')

    idx = roche[i] < 1
    ax.plot(periods[i, idx], peak_mags[i, idx], 'bo', fillstyle='none', markersize=12)

ax.set_xlabel('Orbital period [s]')
ax.set_ylabel('Max magnification')

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot([0], [0], 'bo', markersize=12, fillstyle='none', label='Roche lobe overflow')

ax_min = 2e-3
ax_max = 50

ax.set_ylim(ax_min, ax_max)
ax.set_xlim(0.45, 1.5e8)

ax.plot(ax.get_xlim(), np.ones(2) * thresholds[0], '--k', label=f'{int(thresholds[0]*1000)}mmag threshold', alpha=0.8)
ax.plot(ax.get_xlim(), np.ones(2) * thresholds[1], '--k', label=f'{int(thresholds[1]*1000)}mmag threshold', alpha=0.5)

ax.plot(np.ones(2) * 60, ax.get_ylim(), ':k', alpha=0.3)
ax.text(60*1.1, ax_max / 2, 'one minute')
ax.plot(np.ones(2) * 3600, ax.get_ylim(), ':k', alpha=0.3)
ax.text(3600*1.1, ax_max / 2, 'one hour')
ax.plot(np.ones(2) * 3600 * 24, ax.get_ylim(), ':k', alpha=0.3)
ax.text(3600 * 24 * 1.1, ax_max / 2, 'one day')
ax.plot(np.ones(2) * 3600 * 24 * 365, ax.get_ylim(), ':k', alpha=0.3)
ax.text(3600 * 24 * 365 * 1.1, ax_max / 2, 'one year')

ax.plot([0], [0], linestyle='none', label='"1/10": Duty cycle')

# ax.legend(loc="lower right")
# ax.set_position([0.08, 0.08, 0.64, 0.9])
# ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
ax.legend()

## save the plot

plt.savefig('plots/mag_vs_period.pdf')

## show blackbody

la = np.linspace(200, 1200, 1000)
f = simulator.black_body(la, 5778)

plt.plot(la, f)