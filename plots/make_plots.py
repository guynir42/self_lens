import os
import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt

import xarray as xr

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
orbits = np.geomspace(0.0001, 1, 120)
peak_mags = np.zeros((len(masses), len(orbits)))
source_sizes = np.zeros(peak_mags.shape)
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
        source_sizes[i, j] = sim.source_size
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
orbits = np.geomspace(0.0001, 1, 60)
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

ax_min = 2.5e-3
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
plt.savefig('plots/mag_vs_period.png')


## show a single system with the S/N results for ZTF

## show a single system with the S/N results for LSST

## show a single system with the S/N results for TESS


## show the WD sma models for different masses, with uncertainty ranges


## show the precision as function of mag for surveys

import survey

survey_names = ['TESS', 'CURIOS', 'LSST', 'ZTF']
markers = ['x', 'v', '*', '^', 's']

fig = plt.figure(num=5).clf()
fig, axes = plt.subplots(num=5, figsize=[12, 6])

for i, s in enumerate(survey_names):
    new_survey = survey.Survey(s)
    if new_survey.prec_list is not None:
        m = new_survey.mag_list
        p = new_survey.prec_list
    else:
        m = np.linspace(12, new_survey.limmag)
        p = np.ones(m.shape) * new_survey.precision
    axes.plot(m, p, marker=markers[i], label=s)

axes.legend(loc='upper left', fontsize=14)
axes.set_xlabel('Magnitude', fontsize=14)
axes.set_ylabel('Photometric precision', fontsize=14)
axes.set_yscale('log')

## save the plot

plt.savefig('plots/photometric_precision.pdf')
plt.savefig('plots/photometric_precision.png')


## load the grid scan results for all surveys

survey_names = ['ZTF', 'TESS', 'LSST', 'CURIOS', 'CURIOS_ARRAY', 'LAST']

ds = None
for survey in survey_names:
    filename = f'/home/guyn/timeseries/self_lens/saved/simulate_{survey}.nc'
    new_ds = xr.load_dataset(filename, decode_times=False)
    for name in ['lens_temp', 'star_temp']:
        new_ds[name] = np.round(new_ds[name])

    if ds is None:
        ds = new_ds
    else:
        ds = xr.concat([ds, new_ds], dim='survey')

## show how the declination dies out at two distances for different surveys

ds.flare_duration.sel(
    lens_mass=0.6,
    star_mass=0.6,
    star_temp=8000,
    lens_temp=8000,
    survey=['TESS', 'LSST', 'CURIOS']
).isel(
    semimajor_axis=[50, 65],
    declination=slice(0, 30)
).plot(col='semimajor_axis', hue='survey', marker='*', figsize=[14, 6])

# plt.yscale('log')

## save the plot

plt.savefig('plots/duration_vs_declination.pdf')
plt.savefig('plots/duration_vs_declination.png')


## show the simulation results for lens mass

## show the simulation results for star mass

## show the simulation results for lens temp

## show the simulation results for star temp

## show the simulation results for semimajor axis using a flat WD model and compare to real sma WD model


## show how the declination dies out at two distances for different surveys

import grid_scan

# mass = 0.6
# temp = 16000

plt.clf()
fig, axes = plt.subplots(num=10, figsize=[10, 6])

surveys = ['TESS', 'CURIOS', 'LSST', 'CURIOS_ARRAY']
markers = ['x', 'v', '*', '^', 's']
sens = 1 / (ds.marginalized_volume * ds.probability_density_flat).sum(dim=['lens_mass', 'star_mass', 'lens_temp', 'star_temp'])
# sens = 1 / ds.marginalized_volume

# sens = sens.sel(
#     lens_mass=mass,
#     star_mass=0.6,
#     star_temp=temp,
#     lens_temp=8000,
#     survey=surveys,
# ).isel(
# sens = sens.isel(
#     semimajor_axis=slice(0, 65)
# )

sma = sens.semimajor_axis.values

sens_curve = {}
min_sma = np.inf
for i, s in enumerate(surveys):
    sens_curve[s] = sens.sel(survey=s).values
    min_sma = min(min_sma, sma[np.argmin(np.isinf(sens_curve[s]))])
    axes.plot(sma, sens_curve[s], label=s, marker=markers[i])

g = grid_scan.Grid()
g.dataset = ds

space_density = 0.0055  # WDs per pc^3
binary_fraction = 0.1
prob = {}
for m in ['mid', 'low', 'high']:
    # model = g.get_default_probability_density(temp=m, mass=m, sma=m).sel(
    #     lens_mass=mass,
    #     star_mass=0.6,
    #     star_temp=temp,
    #     lens_temp=8000,
    model = g.get_default_probability_density(temp=m, mass=m, sma=m).sum(
        dim=[
            'star_mass',
            'lens_mass',
            'star_temp',
            'lens_temp',
        ])
    # ).isel(
    #     semimajor_axis=slice(0, 65)
    # )
    prob[m] = model.values * space_density * binary_fraction

axes.fill_between(sma, prob['low'], prob['high'], color='k', alpha=0.3, label=f'WD model')
# axes.plot(sma, prob['mid'], color='k', label=f'{mass}M$_\odot$, {temp}$^\circ$K')
axes.plot(sma, prob['mid'], color='k')

axes.legend(loc='upper right', fontsize=14)

axes.set_xlim((min_sma / 2, 0.4))
# plt.ylim((float(sens.min() * 0.5), max(prob['mid']) * 1e2))

axes.set_yscale('log')
axes.set_xscale('log')

axes.set_xlabel('Semimajor axis [AU]', fontsize=14)
axes.set_ylabel('WD binary density [pc$^{-3}$]', fontsize=14)

period = lambda x: x ** (2 / 3) * 365.25
ax2 = axes.twiny()
ax2.set_xlabel('Period [days]', fontsize=14)
ax2.set_xlim([period(x) for x in axes.get_xlim()])
ax2.set_xscale('log')

plt.show()

## save the plot

plt.savefig('plots/sensitivity_vs_model.pdf')
plt.savefig('plots/sensitivity_vs_model.png')

## show the effective volume vs the density (reciprocal of the sensitivity)

fig, axes = plt.subplots(num=11, figsize=[10, 6])
fig.clf()
fig, axes = plt.subplots(num=11, figsize=[10, 6])

surveys = ['TESS', 'CURIOS', 'LSST', 'CURIOS_ARRAY']
markers = ['x', 'v', '*', '^', 's']
ev = (ds.marginalized_volume * ds.probability_density).sum(dim=['lens_mass', 'star_mass', 'lens_temp', 'star_temp'])
sma = ev.semimajor_axis.values

ev_curve = {}
min_sma = np.inf
for i, s in enumerate(surveys):
    ev_curve[s] = ev.sel(survey=s).values
    min_sma = min(min_sma, sma[np.argmin(ev_curve[s] > 0)])
    axes.plot(sma, ev_curve[s], label=s, marker=markers[i])


axes.set_yscale('log')
axes.set_xscale('log')

axes.set_xlabel('Semimajor axis [AU]', fontsize=14)
axes.set_ylabel('effective volume [pc$^3$]', fontsize=14)

axes.set_ylim((1e-3, 2e4))
axes.set_xlim((5e-3, 10))

period = lambda x: x ** (2 / 3) * 365.25
ax2 = axes.twiny()
ax2.set_xlabel('Period [days]', fontsize=14)
ax2.set_xlim([period(x) for x in axes.get_xlim()])
ax2.set_xscale('log')

space_density = 0.0055
binary_fraction = 0.1

need_volume = 1 / (space_density * binary_fraction)
axes.plot(sma, np.ones(sma.shape) * need_volume, '--', label=f'1 DWD per {int(need_volume)} pc$^3$')
axes.legend(loc='upper right', fontsize=14, framealpha=1)

plt.show()


## save the plot

plt.savefig('plots/effective_volume.pdf')
plt.savefig('plots/effective_volume.png')


