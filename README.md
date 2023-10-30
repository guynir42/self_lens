# Self Lens

Simulate lightcurves from self lensing of binaries.
This repository has been used to simulate and plot the results
for an accompanying paper:
"Detectability of Self-Lensing Flares of White Dwarfs with Compact Companions"
by Guy Nir and Joshua S. Bloom (currently in preparation).

## Installation

This repository is currently not pip-installable.
Please download the files directly from github.
Running pytest in the main directory should run all tests
and download any missing data files.

```commandline
git clone git@github.com:guynir42/self_lens.git
cd self_lens
pytest
```

## Overview

There are a few modules in the `src` folder that are worth mentioning:

- `src/transfer_matrix.py` contains the `TransferMatrix` class that contains the core code
  for running self-lensing simulations. It can be used to generate matrices that translate
  the light from an annulus in the source plane to the observer plane. It can also be used
  load the pre-computed matrices and use those the quickly generate light curves.
  The `matrices` folder contains some saved matrices for a few ranges of the source radius and
  distance between lens and source.
- `src/simulator.py` contains a high-level class (the `Simulator`) that automatically loads
  a few transfer matrices and uses them to produce light curves for systems with a given set
  of physical parameters. It will output a `System` object that contains the light curves,
  some other information about the object (e.g., the orbital period), and also has some methods
  for plotting the results in a nice way.
- `src/survey.py` contains the `Survey` class that interacts with a `System` object and gets the
  probability to detect the system. Each `Survey` is initialized using parameters that describe
  a particulat sky survey. Some useful default surveys are avaiable, e.g., ZTF, TESS and LSST.
- `src/grid_scan.py` contains the `GridScan` class which is used to produce a grid of `System`
  objects and run a list of `Survey` objects against them, which produces a large xarray of
  probabilities and effective volumes (among other parameters).
  These are the core results of the paper. The specific results from the simulations used in the
  paper are saved on Zenodo and are loaded automatically when running the plotting tests.
- `src/distributions.py` contain some tools to estimate the distributions of systems with
  various parameters (e.g., the distribution of semimajor axis of double white dwarf binaries).
- `src/utils.py` contain some useful functions used throughout the code.

## Running simulations

To produce a single system, make a simulator object and run `calculate`:

```python
from src.simulator import Simulator
sim = Simulator()
sim.calculate(
    star_mass=0.3,  # in solar masses
    lens_mass=0.9,  # make the lens more massive
    star_temp=20000,  #  in Kelvin
    lens_temp=4000,  # the lens is smaller and cooler
    declination=0.001,  # in degrees, this is nearly edge-on, as declination is 90-inclination
    semimajor_axis=0.1,  # in AU
)

syst = sim.system

syst.plot(
    detection_limit=0.01,
    distance_pc=10,
    filter_list=["R", "V", "B"],
    fig=None,
    font_size=14,
)
```

The call to `syst.plot()` will show the light curve, a small cartoon of the system,
and some additional information. Choose the detection limit (the precision cutoff)
and the distance to the system (in parsec) to get the apparent magnitude in each filter.
The `fig` input can be used to plot into an existing figure.

Another option is to interactively use the simulator with the built-in GUI.

```python
from src.simulator import Simulator
sim = Simulator()
sim.make_gui()
```

This should open up a GUI that produces a system and plots it, based on
the parameters chosen using the buttons and inputs around the plot.

## Running a grid scan

To calculate the effective volume for each survey,
use the `GridScan` class. Examples for using this class
can be found at the end of the `src/grid_scan.py` module.
This module can be used as a script, which is useful for running
multiple simulations in parallel.

```commandline
python -m src/grid_scan.py <survey> <lens type> <demo/full>
```

Choose "all_surveys" to run all the surveys at once (very slow),
or produce a separate result for each survey (recommended).
Lens types can be "WD" (for double white dwarfs),
or "BH" (which includes white dwarfs in binaries with a neutron star or a black hole).
Use "DEMO" to run a small scan, and not save the result.
Use "FULL" to run a full scan of the parameter space and save it into the `saved` folder.

## Producing plots for publications

The `tests/test_produce_plots.py` module contains functions that produce the plots used in the paper.
Each test produces an individual figure, that are saved into `tests/plots` in both `pdf` and `png` formats.
This folder also includes many examples for how to use the code
and the pre-calculated result files that come with it.
