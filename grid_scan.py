import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import psutil
from collections import defaultdict

import xarray as xr

import simulator
import survey


class Grid:

    def __init__(self, wd_lens=True):
        self.star_masses = None
        self.star_temperatures = None
        self.star_type = 'WD'  # default is white dwarfs, can also choose 'MS' for main sequence
        self.lens_masses = None
        self.lens_temperatures = None
        self.semimajor_axes = None
        self.declinations = None

        # objects
        self.simulator = simulator.Simulator()  # use this to make lightucrves
        self.surveys = None  # a list of survey.Survey objects, applied to each system
        self.systems = None  # a list of systems to save the results of each survey
        self.dataset = None  # an xarray dataset with the summary of the results

        self.setup_default_scan(wd_lens=wd_lens)
        self.setup_default_surveys()

        self.timing = None

    def setup_demo_scan(self, wd_lens=True):
        """
        Make a very small parameter set,
        just to see that the code works.

        :param wd_lens:
            If True, will scan only white dwarf lenses,
            including a mass range up to 1.4,
            and multiple temperatures.
            If False, will run only neutron stars and black holes,
            with a single (irrelevant) temperature and larger
            masses and mass steps.
        """
        self.star_masses = [0.3, 0.6, 0.9, 1.2]
        self.star_masses = [0.6]
        self.star_temperatures = [4000, 8000, 16000]
        self.star_temperatures = [8000]

        if wd_lens:
            self.lens_masses = self.star_masses
            self.lens_temperatures = self.star_temperatures
            self.semimajor_axes = np.geomspace(1e-2, 10, 100)
            # self.declinations = np.linspace(0, 1.0, 201)
            self.declinations = np.geomspace(1e-3, 5, 100)

        else:
            self.lens_masses = np.arange(5.0, 30, 5)
            self.lens_temperatures = [0]
            self.semimajor_axes = np.geomspace(1e-4, 1, 100)
            # self.declinations = np.linspace(0, 3.0, 501)
            self.declinations = np.geomspace(1e-3, 45, 100)

        # print(f'Total number of parameters: {self.get_num_parameters()}')

    def setup_default_scan(self, wd_lens=True):
        """
        Make a small parameter set,
        with low resolution on most parameters.

        :param wd_lens:
            If True, will scan only white dwarf lenses,
            including a mass range up to 1.4,
            and multiple temperatures.
            If False, will run only neutron stars and black holes,
            with a single (irrelevant) temperature and larger
            masses and mass steps.
        """

        self.star_masses = np.round(np.arange(0.2, 1.3, 0.2), 2)
        # self.star_temperatures = np.arange(5000, 45000, 5000)
        # self.star_temperatures = np.array([5000, 10000, 20000, 30000])
        self.star_temperatures = np.round(np.geomspace(4000, 32000, 4))
        if wd_lens:
            self.lens_masses = self.star_masses
            self.lens_temperatures = self.star_temperatures
            self.semimajor_axes = np.geomspace(1e-2, 10, 100)
            self.declinations = np.geomspace(1e-3, 5, 200)
        else:
            self.lens_masses = np.round(np.arange(1.5, 30, 0.5), 2)
            self.lens_temperatures = np.array([0])
            self.semimajor_axes = np.geomspace(1e-4, 1, 100)
            self.declinations = np.geomspace(1e-3, 45, 200)

        # self.semimajor_axes = np.geomspace(1e-3, 10, 100)
        # self.declinations = np.linspace(0, 90, 90001)
        # self.declinations = np.linspace(0, 1.0, 201)
        # self.declinations = np.geomspace(1e-5, 5, 10000)

        # print(f'Total number of parameters: {self.get_num_parameters()}')

    def setup_default_surveys(self):
        """
        Make a short list of several surveys that are commonly used in the analysis.
        """

        self.surveys = [
            survey.Survey('ZTF'),
            survey.Survey('LSST'),
            survey.Survey('TESS'),
            survey.Survey('CURIOS'),
            survey.Survey('CURIOS_ARRAY'),
            # survey.Survey('LAST'),
        ]

    def get_num_parameters(self):
        num_pars = len(self.star_masses) * len(self.star_temperatures)
        num_pars *= len(self.lens_masses) * len(self.lens_temperatures)
        num_pars *= len(self.semimajor_axes) * len(self.declinations)
        return num_pars

    def memory_size(self):

        coord_size = self.get_num_parameters()
        num_arrays = 11
        num_small_arrays = 3
        total_size_bytes = (num_arrays * len(self.surveys) + num_small_arrays) * coord_size * 4

        return total_size_bytes / 1024 ** 3  # return GBs

    def run_simulation(self, keep_systems=False, **kwargs):
        """
        Apply the probability estimates from
        self.surveys to all the systems in the parameter grid.
        All the results are saved in  self.systems.

        :param kwargs:
            Optional parameters are:
            ...

        """
        self.timing = defaultdict(float)
        self.systems = []
        num = int(self.get_num_parameters() / len(self.declinations))
        div = 10 ** int(np.log10(num)-1)

        # build data structures to contain the results
        memory_gb = self.memory_size()
        print(f'Memory footprint of array is {memory_gb:.2f} GB')

        if memory_gb > psutil.virtual_memory().available / 1024 ** 3:
            raise MemoryError(f'Array size required ({memory_gb:.2f} GB) exceeds available memory')

        coords = {
            'lens_temp': self.lens_temperatures,
            'star_temp': self.star_temperatures,
            'lens_mass': self.lens_masses,
            'star_mass': self.star_masses,
            'semimajor_axis': self.semimajor_axes,
            'declination': self.declinations,
        }
        survey_names = [s.name for s in self.surveys]
        coord_names = list(coords.keys())
        coord_lengths = tuple(len(v) for v in coords.values())
        nan_array = np.empty(coord_lengths, dtype=np.float32)
        nan_array[:] = np.nan
        zeros_array = np.zeros((len(self.surveys),) + coord_lengths, dtype=np.float32)

        data_vars = {
            'system_index': (coord_names, nan_array.copy()),
            'orbital_period': (coord_names, nan_array.copy()),
            'peak_magnification': (coord_names, nan_array.copy()),
            'fwhm': (coord_names, nan_array.copy()),
            'distance': (['survey'] + coord_names, zeros_array.copy()),
            'volume': (['survey'] + coord_names, zeros_array.copy()),
            'total_volume': (['survey'] + coord_names, zeros_array.copy()),
            'flare_duration': (['survey'] + coord_names, zeros_array.copy()),
            'flare_prob': (['survey'] + coord_names, zeros_array.copy()),
            'duty_cycle': (['survey'] + coord_names, zeros_array.copy()),
            'visit_prob': (['survey'] + coord_names, zeros_array.copy()),
            'total_prob': (['survey'] + coord_names, zeros_array.copy()),
            'visit_detections': (['survey'] + coord_names, zeros_array.copy()),
            'total_detections': (['survey'] + coord_names, zeros_array.copy()),
            'effective_volume': (['survey'] + coord_names, zeros_array.copy()),
        }

        # add the survey coordinate as well
        coords['survey'] = survey_names

        print(f'Running a grid with {len(self.surveys)} surveys and {num} parameters (not including dec.). ')
        t0 = timer()
        count = 0
        for lt, lens_temp in enumerate(self.lens_temperatures):
            for st, star_temp in enumerate(self.star_temperatures):
                for ml, lens_mass in enumerate(self.lens_masses):
                    for ms, star_mass in enumerate(self.star_masses):
                        for a, sma in enumerate(self.semimajor_axes):
                            for d, dec in enumerate(self.declinations):
                                try:
                                    t1 = timer()
                                    self.simulator.timestamps = None
                                    self.simulator.calculate(lens_temp=lens_temp,
                                                             lens_mass=lens_mass,
                                                             star_temp=star_temp,
                                                             star_mass=star_mass,
                                                             semimajor_axis=sma,
                                                             declination=dec)
                                    self.timing['make lcs'] += timer() - t1
                                except ValueError as e:
                                    if 'requested occulter radius' in str(e):
                                        continue  # very large occulters generally don't make a flare
                                    else:
                                        raise e

                                t1 = timer()
                                flare_probs = []
                                for survey in self.surveys:
                                    try:
                                        survey.apply_detection_statistics(self.simulator.syst)
                                    except Exception as e:
                                        print(e)
                                        raise(e)
                                    if len(self.simulator.syst.flare_prob[survey.name]):
                                        flare_probs.append(max(self.simulator.syst.flare_prob[survey.name]))
                                self.timing['apply stats'] += timer() - t1

                                if len(flare_probs) == 0 or np.all(np.array(flare_probs) == 0):
                                    break  # don't keep scanning declinations after all surveys can't detect anything

                                t1 = timer()
                                s = self.simulator.syst  # shorthand
                                data_vars['orbital_period'][1][lt, st, ml, ms, a, d] = float(s.orbital_period)
                                data_vars['peak_magnification'][1][lt, st, ml, ms, a, d] = max(s.magnifications)
                                data_vars['fwhm'][1][lt, st, ml, ms, a, d] = float(s.fwhm)

                                for i, sur in enumerate(survey_names):

                                    if len(s.flare_prob[sur]):  # if not, leave zeros for all
                                        data_vars['distance'][1][i, lt, st, ml, ms, a, d] = max(s.distances[sur])
                                        data_vars['volume'][1][i, lt, st, ml, ms, a, d] = np.sum(s.volumes[sur])
                                        data_vars['total_volume'][1][i, lt, st, ml, ms, a, d] = np.sum(s.total_volumes[sur])
                                        data_vars['flare_duration'][1][i, lt, st, ml, ms, a, d] = max(s.flare_durations[sur])
                                        data_vars['flare_prob'][1][i, lt, st, ml, ms, a, d] = max(s.flare_prob[sur])
                                        duty_cycle = max(s.flare_durations[sur]) / (s.orbital_period * 3600)
                                        data_vars['duty_cycle'][1][i, lt, st, ml, ms, a, d] = duty_cycle
                                        data_vars['visit_prob'][1][i, lt, st, ml, ms, a, d] = max(s.visit_prob[sur])
                                        data_vars['total_prob'][1][i, lt, st, ml, ms, a, d] = max(s.total_prob[sur])
                                        data_vars['visit_detections'][1][i, lt, st, ml, ms, a, d] = max(s.visit_detections[sur])
                                        data_vars['total_detections'][1][i, lt, st, ml, ms, a, d] = max(s.total_detections[sur])
                                        data_vars['effective_volume'][1][i, lt, st, ml, ms, a, d] = s.effective_volumes[sur]

                                if keep_systems:
                                    # self.simulator.syst.magnifications = None
                                    # self.simulator.syst.timestamps = None
                                    self.systems.append(self.simulator.syst)  # add this system to the list
                                    data_vars['system_index'][1][lt, st, ml, ms, a, d] = len(self.systems) - 1

                                self.timing['store data'] += timer() - t1

                            count += 1  # number of parameters already covered, not including declination
                            if count > 0 and count % div == 0:
                                current_time = timer() - t0
                                total_time = current_time / count * num
                                print(f'count= {count:10d} / {num} | '
                                      f'time= {self.human_readable_time(current_time)} / '
                                      f'{self.human_readable_time(total_time)}')

        print(f'Successfully generated {count} systems in {timer() - t0:.1f}s.')
        self.make_xarray(coords, data_vars)

    def make_xarray(self, coords, data_vars):
        """
        Setup an xarray holding the simulation results

        :param coords:
            coordinates dictionary
        :param data_vars:
            data variables dictionary
        """

        # coords = {
        #     'lens_temp': self.lens_temperatures,
        #     'star_temp': self.star_temperatures,
        #     'lens_mass': self.lens_masses,
        #     'star_mass': self.star_masses,
        #     'semimajor_axis': self.semimajor_axes,
        #     'declination': self.declinations,
        # }
        # survey_names = [s.name for s in self.surveys]
        # coord_names = list(coords.keys())
        # coord_lengths = tuple(len(v) for v in coords.values())
        #
        # coords['survey'] = survey_names  # add another coordinate to keep track of different survey's results
        #
        # nan_array = np.empty(coord_lengths)
        # nan_array[:] = np.nan
        # zeros_array = np.zeros(coord_lengths + (len(self.surveys),))
        #
        # data_vars = {
        #     'system_index': (coord_names, nan_array.copy()),
        #     'orbital_period': (coord_names, nan_array.copy()),
        #     'peak_magnification': (coord_names, nan_array.copy()),
        #     'distance': (coord_names + ['survey'], zeros_array.copy()),
        #     'volume': (coord_names + ['survey'], zeros_array.copy()),
        #     'total_volume': (coord_names + ['survey'], zeros_array.copy()),
        #     'flare_duration': (coord_names + ['survey'], zeros_array.copy()),
        #     'flare_prob': (coord_names + ['survey'], zeros_array.copy()),
        #     'duty_cycle': (coord_names + ['survey'], zeros_array.copy()),
        #     'visit_prob': (coord_names + ['survey'], zeros_array.copy()),
        #     'total_prob': (coord_names + ['survey'], zeros_array.copy()),
        #     'visit_detections': (coord_names + ['survey'], zeros_array.copy()),
        #     'total_detections': (coord_names + ['survey'], zeros_array.copy()),
        #     'effective_volume': (coord_names + ['survey'], zeros_array.copy()),
        # }

        ds = xr.Dataset(coords=coords, data_vars=data_vars)

        # edit the coordinate attributes
        ds.lens_mass.attrs = dict(units='Solar mass', long_name='Lens mass')
        ds.star_mass.attrs = dict(units='Solar mass', long_name='Source mass')
        ds.lens_temp.attrs = dict(units='K', long_name='Lens temperature')
        ds.star_temp.attrs = dict(units='K', long_name='Source temperature')
        ds.declination.attrs = dict(units='deg', long_name='Declination (90-i)')
        ds.semimajor_axis.attrs = dict(units='AU', long_name='Semimajor axis')

        # edit the dataset attributes
        ds.system_index.attrs = dict(
            long_name='System internal index',
            doc='The index of this system in self.systems list.'
        )
        ds.orbital_period.attrs = dict(
            long_name='Orbital period',
            units='hours',
            doc='The orbital period of the system given its semimajor axis and masses.'
        )
        ds.peak_magnification.attrs = dict(
            long_name='Peak magnification',
            doc='The maximum flare magnification relative to the baseline flux.'
        )
        ds.fwhm.attrs = dict(
            long_name='Full Width at Half Maximum',
            units='seconds',
            doc='The flare full width at half maximum (FWHM).'
        )
        ds.distance.attrs = dict(
            long_name='Maximal visible distance',
            units='pc',
            doc='The largest distance at which this system can be detected.'
        )
        ds.volume.attrs = dict(
            long_name='Total accessible volume.',
            units='pc^3',
            doc='The space volume around observer where this system can be detected.'
        )
        ds.total_volume.attrs = dict(
            long_name='Total volume covered.',
            units='pc^3',
            doc='The space volume covered by all fields in the survey.'
        )
        ds.flare_duration.attrs = dict(
            long_name='Flare duration',
            units='s',
            doc='The maximal duration of the flare for all distances/precisions.'
        )
        ds.flare_prob.attrs = dict(
            long_name='Flare probability',
            doc='The best probability to detect a flare for all distances/precisions, '
                'assuming best possible timing.'
        )
        ds.duty_cycle.attrs = dict(
            long_name= 'Duty cycle',
            doc='What fraction of the orbital period is the flare detectable, '
                'for the best distance/precision of this survey. '
        )
        ds.visit_prob.attrs = dict(
            long_name='Visit probability',
            doc='The best probability to detect a flare for all distances/precisions, '
                'assuming a single visit to this system.'
        )
        ds.total_prob.attrs = dict(
            long_name='Total probability',
            doc='The best probability to detect a flare for all distances/precisions, '
                'assuming multiple visits to this system over the duration of the survey.'
        )
        ds.visit_detections.attrs = dict(
            long_name='Number of detections per visit',
            doc='The best expected number of detections for all distances/precisions, '
                'assuming a single visit to this system.'
        )
        ds.total_detections.attrs = dict(
            long_name='Number of detections total',
            doc='The best expected number of detections for all distances/precisions, '
                'assuming multiple visits to this system over the duration of the survey.'
        )
        ds.effective_volume.attrs = dict(
            long_name='Effective volume',
            units='pc^3',
            doc='Total volume for systems of this kind in this survey. '
                'Should be multiplied by the space density of objects'
                'to get the total expected number of detections.'
        )

        self.dataset = ds

    def marginalize_declinations(self):
        if self.dataset is None:
            raise ValueError('Must have a dataset first!')
        ds = self.dataset
        dec = ds.declination * np.pi / 180  # convert to radians
        # dec_step = dec[1] - dec[0]  # do something more complicated if uneven steps
        dec_step = np.diff(dec)  # each dec value accounts for the range of declinations back down to previous point
        new_ev = ds.effective_volume.isel(declination=slice(1, None))
        new_ev = (new_ev * np.cos(dec[1:]) * dec_step).sum(dim='declination')
        new_ev.attrs['name'] = 'effective_volume'
        new_ev.attrs['long_name'] = 'Effective volume'
        new_ev.attrs['units'] = 'pc^3'
        ds['marginalized_volume'] = new_ev

        return new_ev

    def get_probability_density(self, lens_temp=0, star_temp=0, lens_mass=0, star_mass=0, sma=0):
        """
        generate a DataArray with the same dimensions
        as the grid scan results dataset,
        (without survey and declination axes)
        which contains the probability to find
        a system with the given parameters.
        The total probability over all parameters
        in range sums to one.
        Multiply this model by the effective
        volume (marginalized over declinations)
        and sum to get the reciprocal to the
        space density of binaries required
        to make a detection.
        Alternatively, multiply the sum
        by a model space density to get
        the total number of expected
        detections.

        Each parameter can be a scalar,
        in which case it is used as a power law index,
        or a two-element list/array, in which case the
        elements are treated as mu and sigma of a
        gaussian distribution,
        or as a vector of the same length as the
        data coordinates, in which case the values
        are taken as proportional to the
        probability density.

        :param lens_temp: float scalar or array
            distribution or power law index for the lens temperature axis probabilities.
        :param star_temp: float scalar or array
            distribution or power law index for the star temperature axis probabilities.
        :param lens_mass: float scalar or array
            distribution or power law index for the lens mass axis probabilities.
        :param star_mass: float scalar or array
            distribution or power law index for the star mass axis probabilities.
        :param sma: float scalar or array
            distribution or power law index for the semimajor axis probabilities.
        
        :return:
            a DataArray with the same dims as the main result dataset,
            excluding "survey" and "declinations".
            Multiply this by the declination marginalized
            effective volume and sum all axes to get
            the reciprocal of the space density of objects.
        """
        if self.dataset is None:
            raise ValueError('Must have a dataset first!')

        def parse_par(par, data):
            """
            Interpret the incoming parameter.
            If scalar, interpret as power law.
            If two-element, treat as gaussian mean/std.
            If length is equal to data, use it
            directly as the relative probability density.
            """
            if par is None:
                par = 0  # convert to flat density

            if not hasattr(par, '__len__'):
                if np.all(data == 0):
                    return xr.ones_like(data.astype(float))
                return data.astype(float) ** float(par)  # use a power law
            if type(par) == xr.DataArray:
                return par  # return this directly
            if len(par) == 2:
                return np.exp(-0.5 * (data - par[0]) ** 2 / par[1] ** 2)
            if len(par) != len(data):
                raise ValueError(f'Mismatch of parameter ({len(par)}) and data ({len(data)}).')

            # if none of the above, just use the "par" values
            # as the new data in an array the same size as
            # the relevant coordinate
            new_array = xr.full_like(data)
            new_array.data = par
            return par  # the parameter values are taken as the density values

        sma_dist = parse_par(sma, self.dataset.semimajor_axis)
        lens_mass_dist = parse_par(lens_mass, self.dataset.lens_mass)
        star_mass_dist = parse_par(star_mass, self.dataset.star_mass)
        lens_temp_dist = parse_par(lens_temp, self.dataset.lens_temp)
        star_temp_dist = parse_par(star_temp, self.dataset.star_temp)

        prob = lens_temp_dist * star_temp_dist * lens_mass_dist * star_mass_dist * sma_dist
        prob = prob / np.sum(prob)

        new_arr = xr.full_like(self.dataset.effective_volume, 0)
        new_arr = new_arr.isel(survey=0, declination=0)  # get rid of unneeded coordinates
        new_arr = new_arr.drop('survey')
        new_arr = new_arr.drop('declination')
        new_arr.name = 'probability_density'
        new_arr.attrs['long_name'] = 'WD probability density'
        new_arr.attrs['doc'] = 'Probability to find each type of white dwarf system.'
        if 'units' in new_arr.attrs:
            del new_arr.attrs['units']
        new_arr.data = prob

        self.dataset['probability_density'] = new_arr

        return new_arr

    def get_default_probability_density(self, temp='mid', mass='mid', sma='mid'):
        """
        Use the parameters in reference Maoz et al 2018
        (https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2584M/abstract)
        to generate a probability density for white dwarfs.
        Each parameter (temp, mass, sma->semimajor axis)
        controls of we want to get the "mid" (best estimate
        value) or "low" or "high" for the lower or higher
        models.
        """

        # less negative slope of the temperature power law
        # indicates more hot WDs and so, more detections.
        if temp == 'mid':
            temp = -2
        elif temp == 'low':
            temp = -3
        elif temp == 'high':
            temp = -1
        else:
            raise ValueError(f'Unknown option "{temp}". Use "low", "high" or "mid".')

        # right now, only have one mass model
        if mass == 'mid':
            mass = (0.6, 0.2)
        elif mass == 'low':
            mass = (0.6, 0.2)
        elif mass == 'high':
            mass = (0.6, 0.2)
        else:
            raise ValueError(f'Unknown option "{mass}". Use "low", "high" or "mid".')

        if sma == 'mid':
            sma = -1.3
        elif sma == 'low':
            sma = -1.5
        elif sma == 'high':
            sma = -1.0
        else:
            raise ValueError(f'Unknown option "{sma}". Use "low", "high" or "mid".')

        sma = self.semimajor_axis_distribution(sma)

        return self.get_probability_density(lens_temp=temp, star_temp=temp, lens_mass=mass, star_mass=mass, sma=sma)

    def semimajor_axis_distribution(self, alpha=-1.3, sma=None, star_masses=None, lens_masses=None):
        """
        The parametrization given in reference Maoz et al 2018
        (https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2584M/abstract)
        This tells us the present day distribution of semimajor axis
        for binary WDs with an initial sma distribution with power law
        index alpha.

        :param alpha: scalar float
            The power law index of the zero age, semimajor axis power law index
            for binary white dwarfs that emerge from the comme envelope phase.

        :param sma: array-like
            The semimajor axis values to evaluate the distribution at.
            If None, use the values in the dataset or the values of the object.

        :param star_masses: array-like
            The masses of the stars in solar masses.
            If None, use the dataset masses or the masses on the object.
        :param lens_masses: array-like
            The masses of the lenses in solar masses.
            If None, use the dataset masses or the masses on the object.

        :return: xarray data array
            The distribution of the number density (not normalized to anything)
            of the binary white dwarfs, based on their masses and semimajor axis.
        """
        # G_over_c = 1.2277604981899157e-73  # = G**3/c**5 = (6.6743e-11) ** 3 / (299792458) ** 5 in MKS!
        # const = 6.286133750732368e-72  # = 256 / 5 * G_over_c
        # const = const / (1.496e11) ** 4 * 3.15576e16 * (1.989e30) ** 3  # unit conversions
        const = 3.116488722396829e-09  # in units of AU^4 Gyr^-1 solar_mass ^-3

        if sma is None:
            if self.dataset is not None:
                sma = self.dataset.semimajor_axis
            else:
                sma = np.array(self.semimajor_axes)

        if star_masses is not None:
            m1 = star_masses
        elif self.dataset is not None:
            m1 = self.dataset.lens_mass
        else:
            m1 = np.array(self.lens_masses)

        if lens_masses is not None:
            m2 = lens_masses
        elif self.dataset is not None:
            m2 = self.dataset.lens_mass
        else:
            m2 = np.array(self.lens_masses)

        t0 = 13.5  # age of the galaxy in Gyr

        x = sma / (const * m1 * m2 * (m1 + m2) * t0) ** (1/4)

        if alpha == -1:
            return x ** 3 * np.log(1 + x ** (-4))
        else:
            return - x ** (4 + alpha) * ((1 + x ** (-4)) ** ((alpha + 1) / 4) - 1)

    def gravitational_semi_major_axis(self, lens_mass, star_mass):
        """
        The semi-major axis at which the systems will decay over the age of the galaxy.
        """
        # G = 6.6743e-11  # m^3 kg^-1 s^-2
        # c = 299792458  # m/s
        # t0 = 13.5e9 * 365.25 * 24 * 3600  # age of the galaxy in seconds
        # Msun = 1.989e30  # kg
        # AU = 1.496e11  # m
        # Kt = 256/5* G**3 / c**5 * t0 = 2.6780664751035093e-54 # in units of m^4 kg^-3
        # Kt = Kt * Msun^3 / AU^4 = 4.207259775235719e-08 solar_mass^-3

        return np.power(4.207259775235719e-08 * lens_mass * star_mass * (lens_mass + star_mass), 0.25)

    def get_total_volume(self):

        if 'probability_density' not in self.dataset:
            raise KeyError('Need to first calculate a probability model!')

        eff = self.marginalize_declinations()
        prob = self.dataset.probability_density

        return (eff * prob).sum()

    def total_detections(self, priors=None):
        """
        Calculate the total number of detections expected,
        given a set of prior probabilities or number of systems
        with the given prameters.

        :param priors: dict or set
            For each key in priors, use the given value
            as a function that gets a single value (the
            system parameter named by this key) and outputs
            either the probability of finding such a system
            (normalized to unity when marginalizing over all
            other parameters) or the total number of systems
            with those parameters.
            If any value is given as None, the default prior
            for that parameter is used instead.
            (inspect the defaults for examples on how to write
            the prior functions).
            Basically each system's probabilities / number of
            detections are multiplied by the prior, then all the
            products are summed to give the marginalized total
            probability/detections over that parameter.
            If a set is given instead of a dictionary, the result
            is the same as giving a dictionary where all values
            are None (use defaults for all parameters in the set).
            Parameters that do not have a matching key in the dict
            or set are left free, so that the return values are
            given as a function of the remaining parameters,
            e.g., if not given "lens_mass" the result will be
            prob/numbers as a function of lens_mass.

        :return:
            Pandas dataframe with all the columns for parameters
            that were not marginalized over, and output columns
            for total_detections and total_volume.
            Need to think about this...
        """

    @staticmethod
    def human_readable_time(time_seconds):
        hours, rem = divmod(time_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f'{int(hours):02d}:{int(minutes):02d}:{seconds:.1f}'


if __name__ == "__main__":

    if 0:
        if len(sys.argv) > 1:
            survey_name = sys.argv[1].upper()
        else:
            survey_name = 'all_surveys'
            # survey_name = 'ZTF'

        if len(sys.argv) > 2:
            lens_type = sys.argv[2]
        else:
            lens_type = 'BH'

        if len(sys.argv) > 3:
            demo_mode = sys.argv[3].upper()
        else:
            demo_mode = 'DEMO'

        if lens_type not in ['WD', 'BH']:
            raise ValueError('Value of "lens_type" must be "WD" or "BH". '
                             f'Instead got "{lens_type}".')

        if demo_mode not in ['DEMO', 'FULL']:
            raise ValueError('Value of "demo_mode" must be "DEMO" or "REAL". '
                             f'Instead got "{demo_mode}".')

        print(f'Running {demo_mode} simulation for survey: {survey_name}')

        g = Grid(wd_lens=lens_type == 'WD')
        if demo_mode == 'DEMO':
            g.setup_demo_scan(wd_lens=lens_type == 'WD')

        if survey_name != 'all_surveys':
            g.surveys = [survey.Survey(survey_name)]

        g.run_simulation(keep_systems=demo_mode == 'DEMO')

        ev = g.marginalize_declinations()

        # make a probability distribution that is flat in semimajor axis space
        prob_flat = g.get_probability_density(lens_temp=-2, star_temp=-2, lens_mass=(0.6, 0.2), star_mass=(0.6, 0.2))
        g.dataset['probability_density_flat'] = prob_flat
        prob = g.get_default_probability_density()
        total_vol = float(g.get_total_volume())
        print(f'total volume: {total_vol:.1f}pc^3')
        # g.datset = xr.load_dataset('saved/grid_data.nc')
        print(g.timing)

        ds = g.dataset

        if demo_mode == 'FULL':
            try:
                os.mkdir('saved')
            except FileExistsError:
                pass
            g.dataset.to_netcdf(f'saved/simulate_{survey_name}_{lens_type}.nc')

    else:
        g = Grid()
        g.setup_demo_scan(wd_lens=True)
        g.star_masses = [0.6]
        print(g.semimajor_axis_distribution())
        plt.plot(g.semimajor_axes, g.semimajor_axis_distribution())
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
