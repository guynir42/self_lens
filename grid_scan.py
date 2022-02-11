import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

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

        self.setup_small_scan()
        self.setup_default_surveys()

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

        self.star_masses = np.arange(0.2, 1.4, 0.1)
        self.star_temperatures = np.array([5000, 7500, 10000])

        if wd_lens:
            self.lens_masses = np.arange(0.2, 1.4, 0.1)
            self.lens_temperatures = np.array([5000, 7500, 10000])
        else:
            self.lens_masses = np.arange(1.5, 30, 0.5)
            self.lens_temperatures = np.array([5000])

        self.semimajor_axes = np.geomspace(0.0001, 10, 300)
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
        for sl, lens_temp in enumerate(self.lens_temperatures):
            for st, star_temp in enumerate(self.star_temperatures):
                for ms, star_mass in enumerate(self.star_masses):
                    for ml, lens_mass in enumerate(self.lens_masses):
                        for a, sma in enumerate(self.semimajor_axes):
                            for d, dec in enumerate(self.declinations):
                                try:
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
                                    s.apply_detection_statistics(self.simulator.syst)
                                    if len(self.simulator.syst.flare_prob[s.name]):
                                        flare_probs.append(max(self.simulator.syst.flare_prob[s.name]))

                                if len(flare_probs) and np.all(np.array(flare_probs) == 0):
                                    break  # don't keep scanning declinations after all surveys can't detect anything

                                self.systems.append(self.simulator.syst)  # add this system to the list

                            count = len(self.systems)

                            if count > 0 and count % div == 0:
                                current_time = timer() - t0
                                total_time = current_time / count * num
                                print(f'count= {count:10d} / {num} | time= {current_time:.1f} / {total_time:.1f}s')

        print(f'Successfully generated {count} systems in {timer() - t0:1.f}s.')

    def get_systems(self, **kwargs):
        """
        Get all systems that match certain creteria.
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

    g = Grid()
