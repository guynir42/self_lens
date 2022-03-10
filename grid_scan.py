import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import xarray as xr

import simulator
import survey

class Grid:

    def __init__(self):
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

        self.setup_small_scan()
        self.setup_default_surveys()

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
        self.star_masses = np.arange(0.2, 1.3, 0.5)
        self.star_temperatures = np.array([5000, 10000])

        if wd_lens:
            self.lens_masses = np.arange(0.2, 1.3, 0.5)
            self.lens_temperatures = np.array([5000, 10000])
        else:
            self.lens_masses = np.arange(1.0, 30, 3)
            self.lens_temperatures = np.array([5000])

        self.semimajor_axes = np.geomspace(0.0001, 10, 30)
        self.declinations = np.linspace(0, 90, 10000)

        num_pars = len(self.star_masses) * len(self.star_temperatures)
        num_pars *= len(self.lens_masses) * len(self.lens_temperatures)
        num_pars *= len(self.semimajor_axes)

        print(f'Total number of parameters (excluding dec): {self.get_num_parameters()}')

    def setup_small_scan(self, wd_lens=True):
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

        self.star_masses = np.arange(0.2, 1.4, 0.2)
        self.star_temperatures = np.array([5000, 7500, 10000])

        if wd_lens:
            self.lens_masses = np.arange(0.2, 1.4, 0.2)
            self.lens_temperatures = np.array([5000, 7500, 10000])
        else:
            self.lens_masses = np.arange(1.5, 30, 0.5)
            self.lens_temperatures = np.array([5000])

        self.semimajor_axes = np.geomspace(0.0001, 10, 100)
        self.declinations = np.linspace(0, 90, 10000)

        num_pars = len(self.star_masses) * len(self.star_temperatures)
        num_pars *= len(self.lens_masses) * len(self.lens_temperatures)
        num_pars *= len(self.semimajor_axes)

        print(f'Total number of parameters (excluding dec): {self.get_num_parameters()}')

    def get_num_parameters(self):
        num_pars = len(self.star_masses) * len(self.star_temperatures)
        num_pars *= len(self.lens_masses) * len(self.lens_temperatures)
        num_pars *= len(self.semimajor_axes)
        return num_pars

    def setup_default_surveys(self):
        """
        Make a short list of several surveys that are commonly used in the analysis.
        """

        self.surveys = [survey.Survey('ZTF'), survey.Survey('CURIOS')]

    def run_simulation(self, **kwargs):
        """
        Apply the probability estimates from
        self.surveys to all the systems in the parameter grid.
        All the results are saved in  self.systems.

        :param kwargs:
            Optional parameters are:
            ...

        """
        self.systems = []
        num = self.get_num_parameters()
        div = 10 ** int(np.log10(num)-1)

        print(f'Running a grid with {len(self.surveys)} surveys and {num} parameters (not including dec.). ')
        t0 = timer()
        count = 0
        for lt, lens_temp in enumerate(self.lens_temperatures):
            for st, star_temp in enumerate(self.star_temperatures):
                for ms, star_mass in enumerate(self.star_masses):
                    for ml, lens_mass in enumerate(self.lens_masses):
                        for a, sma in enumerate(self.semimajor_axes):
                            for d, dec in enumerate(self.declinations):
                                try:
                                    self.simulator.timestamps = None
                                    self.simulator.calculate(lens_temp=lens_temp,
                                                             lens_mass=lens_mass,
                                                             star_temp=star_temp,
                                                             star_mass=star_mass,
                                                             semimajor_axis=sma,
                                                             declination=dec)
                                except ValueError as e:
                                    if 'requested occulter radius' in str(e):
                                        continue  # very large occulters generally don't make a flare
                                    else:
                                        raise e

                                flare_probs = []
                                for s in self.surveys:
                                    try:
                                        s.apply_detection_statistics(self.simulator.syst)
                                    except Exception as e:
                                        print(e)
                                    if len(self.simulator.syst.flare_prob[s.name]):
                                        flare_probs.append(max(self.simulator.syst.flare_prob[s.name]))

                                if len(flare_probs) == 0 or np.all(np.array(flare_probs) == 0):
                                    break  # don't keep scanning declinations after all surveys can't detect anything
                                self.simulator.syst.magnifications = None
                                self.simulator.syst.timestamps = None
                                self.systems.append(self.simulator.syst)  # add this system to the list

                            # count = len(self.systems)
                            count += 1  # number of parameters already covered, not including declination
                            if count > 0 and count % div == 0:
                                current_time = timer() - t0
                                total_time = current_time / count * num
                                print(f'count= {count:10d} / {num} | time= {current_time:.1f} / {total_time:.1f}s')

        print(f'Successfully generated {len(self.systems)} systems in {timer() - t0:.1f}s.')

    def summarize(self):
        """
        Put the results of the simulation into an xarray dataset.
        """

        t0 = timer()
        # first pass of the data:
        star_temps = []
        star_masses = []
        lens_temps = []
        lens_masses = []
        smas = []
        decs = []
        for s in self.systems:
            star_temps.append(s.star_temp)
            star_masses.append(s.star_mass)
            lens_temps.append(s.lens_temp)
            lens_masses.append(s.lens_mass)
            smas.append(s.semimajor_axis)
            decs.append(s.declination)

        star_temps = np.unique(star_temps)
        star_masses = np.unique(star_masses)
        lens_temps = np.unique(lens_temps)
        lens_masses = np.unique(lens_masses)
        smas = np.unique(smas)
        decs = np.unique(decs)

        coords = {
             'lens_mass': lens_masses,
             'lens_temp': lens_temps,
             'star_mass': star_masses,
             'star_temp': star_temps,
             'declination': decs,
             'semimajor_axis': smas,
         }
        survey_names = [s.name for s in g.surveys]
        coord_names = list(coords.keys())
        coord_lengths = tuple(len(v) for v in coords.values())

        coords['survey'] = survey_names  # add another coordinate to keep track of different survey's results

        nan_array = np.empty(coord_lengths)
        nan_array[:] = np.nan
        zeros_array = np.zeros(coord_lengths + (len(self.surveys),))

        data_vars = {
            'system_index': (coord_names, nan_array.copy()),
            'orbital_period': (coord_names, nan_array.copy()),
            'distance': (coord_names + ['survey'], zeros_array.copy()),
            'volume': (coord_names + ['survey'], zeros_array.copy()),
            'total_volume': (coord_names + ['survey'], zeros_array.copy()),
            'flare_duration': (coord_names + ['survey'], zeros_array.copy()),
            'flare_prob': (coord_names + ['survey'], zeros_array.copy()),
            'duty_cycle': (coord_names + ['survey'], zeros_array.copy()),
            'visit_prob': (coord_names + ['survey'], zeros_array.copy()),
            'total_prob': (coord_names + ['survey'], zeros_array.copy()),
            'visit_detections': (coord_names + ['survey'], zeros_array.copy()),
            'total_detections': (coord_names + ['survey'], zeros_array.copy()),
            'effective_volume': (coord_names + ['survey'], zeros_array.copy()),
        }

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

        # second pass on the data:
        for i, s in enumerate(self.systems):
            indexer = {k: getattr(s, k) for k in coord_names}
            ds.system_index.loc[indexer] = i
            ds.orbital_period.loc[indexer] = s.orbital_period
            for sur in survey_names:
                indexer['survey'] = sur
                if len(s.distances[sur]):  # if not, leave zeros for all values
                    ds.distance.loc[indexer] = max(s.distances[sur])
                    ds.volume.loc[indexer] = max(s.volumes[sur])
                    ds.total_volume.loc[indexer] = max(s.total_volumes[sur])
                    ds.flare_duration.loc[indexer] = max(s.flare_durations[sur])
                    ds.flare_prob.loc[indexer] = max(s.flare_prob[sur])
                    ds.duty_cycle.loc[indexer] = max(s.flare_prob[sur]) / (s.orbital_period * 3600)
                    ds.visit_prob.loc[indexer] = max(s.visit_prob[sur])
                    ds.total_prob.loc[indexer] = max(s.total_prob[sur])
                    ds.visit_detections.loc[indexer] = max(s.visit_detections[sur])
                    ds.total_detections.loc[indexer] = max(s.total_detections[sur])
                    ds.effective_volume.loc[indexer] = s.effective_volumes[sur]

        print(f'Time to convert results to dataset: {timer() - t0:.2g}s')

    def get_systems(self, **kwargs):
        """
        Get all systems that match certain criteria.
        E.g., to get all systems with lens_mass=3.0 use:
        >> system_list = grid.get_systems(lens_mass=3.0)
        To get systems fitting a range of parameter values
        use a 2-tuple instead of a single value.

        A list of possible parameters:
        - Source properties: star_mass, star_size, star_temp,
        - Lens properties:
        - Orbital properties:

        Returns a list of all systems matching the requested
        parameters. If no parameters are requested,
        simply returns self.systems.
        """

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


if __name__ == "__main__":
    pass
    g = Grid()
    g.setup_demo_scan()
    g.run_simulation()
    g.summarize()