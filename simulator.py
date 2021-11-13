import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import transfer_matrix

# TODO: add automatic white dwarf radius from https://github.com/mahollands/MR_relation
# TODO: add automatic white dwarf radius from https://ui.adsabs.harvard.edu/abs/2002A%26A...394..489B/abstract eq (5)

# Magnification of each image: https://microlensing-source.org/tutorial/magnification/
# If requested source is smaller than smallest size in transfer matrices, use point approx.
# Must check the occulter size, if it occults one or both images.
# If distance is out of bounds, use point approx, fit to the smaller distances.
# If occulter size is out of bounds, throw an exception (unphysical situation anyway).
# Occulters should be produced to cover the full distances (for small sources)
# Up to a few times the Einstein radius.
# After that, it becomes unphysical to get a giant occulter (it becomes an eclipsing binary)
# If source is too big, should use geometric approximation.


class Simulator:

    def __init__(self):
        self.matrices = None  # list of matrices to use to produce the lightcurves

        self.latest_runtime = 0  # how much time it took to calculate a single lightcurve
        self.use_dilution = False  # automatically dilute the magnification with the companion flux

        self._default_star_type = 'WD'
        self._default_lens_type = 'WD'
        self._default_semimajor_axis = 1  # in AU
        self._default_inclination = 90  # edge on

        # lensing parameters (in units of Einstein radius)
        self.einstein_radius = None  # in Solar units, to translate the input
        self.impact_parameter = None
        self.orbital_period = None  # in units of hours
        self.source_size = None
        self.occulter_size = None

        # physical parameters (in Solar units)
        self.semimajor_axis = None  # in AU
        self.inclination = None  # orbital inclination (degrees)
        self.star_mass = None  # in Solar mass units
        self.star_type = None  # what kind of object: "star", "WD", "NS", "BH"
        self.star_temp = None  # in Kelvin
        self.star_size = None  # in Solar radius units
        self.star_flux = None  # relative brightness in any units you want
        self.lens_mass = None  # in Solar mass units
        self.lens_type = None  # what kind of object: "star", "WD", "NS", "BH"
        self.lens_temp = None  # in Kelvin
        self.lens_size = None  # in Solar radius units
        self.lens_flux = None  # relative brightness in any units you want

        # measurement inputs
        self.timestamps = None  # times measured from closest approach
        self.time_units = "seconds"

        # intermediate results
        self.position_radii = None  # distances from star center, at given timestamps
        self.position_angles = None  # angles of vector connecting source and lens centers, measured at given timestamps

        # outputs
        self.magnifications = None
        self.offset_sizes = None
        self.offset_angles = None

    def load_matrices(self, filenames=None):

        self.matrices = []

        if filenames is None:
            filenames = 'saved/*.npz'

        filenames = glob.glob(filenames)

        for f in filenames:
            self.matrices.append(transfer_matrix.TransferMatrix.from_file(f))

    def translate_time_units(self, units):
        d = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 3600 * 24, 'years': 3600 * 24 * 365.25}
        return d[units.lower()]

    def translate_type(self, type):
        t = type.lower().replace('_', ' ')

        if t == 'ms' or t == 'main sequence' or t == 'star':
            t = 'MS'
        elif t == 'wd' or t == 'white dwarf':
            t = 'WD'
        elif t == 'ns' or t == 'neutron star':
            t = 'NS'
        elif t == 'bh' or t == 'black hole':
            t = 'BH'
        else:
            raise ValueError(f'Unknown object type "{type}". Use "MS" or "WD" or "NS" or "BH"...')

        return t

    def guess_object_properties(self, type, mass, temp=None, size=None, flux=None):
        type = self.translate_type(type)
        if temp is None:
            if type == 'MS':
                temp = 5000
            if type == 'WD':
                temp = 20000
            if type == 'NS':
                temp = 20000
            if type == 'BH':
                temp = 0

        if size is None:
            if type == 'MS':
                size = 1
            if type == 'WD':
                # Equation 5 of https://ui.adsabs.harvard.edu/abs/2002A%26A...394..489B/abstract
                mass_chand = mass / 1.454
                size = 0.001125 * np.sqrt(mass_chand ** (-2/3) - mass_chand ** (2/3))
            if type == 'NS':
                size = 0
            if type == 'BH':
                size = 0

        if flux is None:
            const = 2.744452656619891e+17  # boltzmann const * solar radius ** 2 * ergs
            flux = 4 * np.pi * const * size ** 2 * temp ** 4  # in erg/s

        # print(f'mass= {mass} | type= {type} | size= {size} | temp= {temp} | flux= {flux}')

        return temp, size, flux

    def input_system(self, **kwargs):

        for key, val in kwargs.items():
            if hasattr(self, key) and re.match('(star|lens)_.*', key) or key == 'inclination' or key == 'semimajor_axis' or key == 'time_units':
                setattr(self, key, val)
                # print(f'key= {key}, val= {val}')
            else:
                raise ValueError(f'Unkown input argument "{key}"')

        if self.star_mass is None:
            raise ValueError('Must provide a mass for the source object.')

        if self.lens_mass is None:
            raise ValueError('Must provide a mass for the lensing object.')

        if self.star_type is None:
            self.star_type = self._default_star_type
        if self.lens_type is None:
            self.lens_type = self._default_lens_type

        if self.semimajor_axis is None:
            self.semimajor_axis = self._default_semimajor_axis

        if self.inclination is None:
            self.inclination = self._default_inclination

        (self.star_temp, self.star_size, self.star_flux) = self.guess_object_properties(self.star_type, self.star_mass, self.star_temp, self.star_size, self.star_flux)
        (self.lens_temp, self.lens_size, self.lens_flux) = self.guess_object_properties(self.lens_type, self.lens_mass, self.lens_temp, self.lens_size, self.lens_flux)

        # constants = 6.67408e-11 * 1.989e30 * 1.496e11 / 299792458 ** 2
        constants = 220961382691907.84
        self.einstein_radius = np.sqrt(4 * constants * self.lens_mass * self.semimajor_axis)
        self.einstein_radius /= 6.957e8  # translate from meters to solar radii
        if self.inclination == 90:
            self.impact_parameter = 0
        else:
            self.impact_parameter = self.semimajor_axis * np.cos(self.inclination / 180 * np.pi)
            self.impact_parameter *= 215.032  # convert to solar radii
            self.impact_parameter /= self.einstein_radius  # convert to units of Einstein Radius

        self.orbital_period = self.semimajor_axis ** (3 / 2) / np.sqrt(self.star_mass + self.lens_mass)
        self.orbital_period *= 365.25 * 24  # convert from years to hours

        self.source_size = self.star_size / self.einstein_radius
        self.occulter_size = self.lens_size / self.einstein_radius

    def input_timestamps(self, timestamps=None, units=None):

        if timestamps is not None:
            self.timestamps = timestamps

        if units is not None:
            self.time_units = units

        phases = (self.timestamps * self.translate_time_units(self.time_units)) / (self.orbital_period * 3600)

        projected_radius = self.semimajor_axis * 215.032 / self.einstein_radius  # AU to solar radii to Einstein radius
        x = projected_radius * np.sin(2 * np.pi * phases)
        y = projected_radius * np.cos(2 * np.pi * phases) * np.cos(self.inclination / 180 * np.pi)

        self.position_radii = np.sqrt(x ** 2 + y ** 2)
        self.position_angles = np.arctan2(y, x)

    def choose_matrix(self):
        r = self.source_size
        matrix_list = sorted(self.matrices, key=lambda x: x.max_source, reverse=False)
        this_matrix = None
        for m in matrix_list:
            if m.min_source < r < m.max_source:
                this_matrix = m
                break

        # if this_matrix is None:
            # raise ValueError(f'Cannot find any transfer matrix that includes the source radius ({r})')

        return this_matrix

    def calc_lightcurve(self, **kwargs):

        t0 = timer()
        if 'timestamps' in kwargs:
            timestamps = kwargs.pop('timestamps')
        else:
            timestamps = None

        if 'time_units' in kwargs:
            time_units = kwargs.pop('time_units')
        else:
            time_units = None

        if kwargs:
            self.input_system(**kwargs)

        if timestamps is not None or time_units is not None:
            self.input_timestamps(timestamps, time_units)

        if self.timestamps is None:
            timestamps = np.linspace(-0.01 * self.orbital_period, 0.01 * self.orbital_period, 201, endpoint=True)
            self.input_timestamps(timestamps*3600, 'seconds')

        # first check if the requested source size is lower/higher than any matrix
        max_sizes = np.array([mat.max_source for mat in self.matrices])
        if np.all(max_sizes < self.source_size):
            raise ValueError(f'Requested source size ({self.source_size}) '
                             f'is larger than largest values in all matrices ({np.max(max_sizes)})')

        matrix = self.choose_matrix()

        if matrix is None:  # source too small!
            mag = transfer_matrix.point_source_approximation(self.position_radii)
        else:
            mag = matrix.radial_lightcurve(source=self.source_size,
                                           distances=self.position_radii,
                                           occulter_radius=self.occulter_size,
                                           get_offsets=False  # at some point I'd want to add the offsets too
                                           )

        # dilution of magnification if both objects are luminous
        self.magnifications = mag

        if self.use_dilution:
            total_flux = self.star_flux + self.lens_flux
            self.magnifications = (self.star_flux * self.magnifications + self.lens_flux) / total_flux

        self.latest_runtime = timer() - t0

        return self.magnifications

    def output_system(self):
        sys = System()

        # copy all relevant properties of this object
        for k in sys.__dict__.keys():
            if hasattr(self, k):
                setattr(sys, k, getattr(self, k))

        return sys


class System:
    def __init__(self):
        self.semimajor_axis = None  # in AU
        self.period = None  # in units of hours
        self.inclination = None  # orbital inclination (degrees)
        self.star_mass = None  # in Solar mass units
        self.star_type = None  # what kind of object: "star", "WD", "NS", "BH"
        self.star_temp = None  # in Kelvin
        self.star_size = None  # in Solar radius units
        self.star_flux = None  # bolometric flux in units of erg/s
        self.lens_mass = None  # in Solar mass units
        self.lens_type = None  # what kind of object: "star", "WD", "NS", "BH"
        self.lens_temp = None  # in Kelvin
        self.lens_size = None  # in Solar radius units
        self.lens_flux = None  # bolometric flux in units of erg/s

        self.timestamps = None  # for the lightcurve (units are given below, defaults to seconds)
        self.time_units = None  # for the above-mentioned timestamps
        self.magnifications = None  # the measured lightcurve

        # a few dictionaries to be filled by each survey,
        # each one keyed to the survey's name, and containing
        # the results: detection probability for different distances
        # and also the volume associated with each distance.
        self.distances = {}  # parsec
        self.det_prob = {}  # probability (0 to 1)
        self.volumes = {}  # parsec^3

        # for each parameter in the first block,
        # what are the intervals around the given values,
        # that we can use to estimate how many such systems exist
        # (to fill out the density parameter).
        # E.g., to say there are 10 systems / parsec^3 with lens_mass=0.4
        # you must decide the lens mass range, maybe 0.35 to 0.45,
        # only then can you say how many "such systems" should exist
        self.par_ranges = {}  # for mass/size/temp of lens/star, inclination, semimajor axis
        self.density = None  # how many such system we expect exist per parsec^3

    def plot(self):
        plt.plot(self.timestamps, self.magnifications, '-o')
        plt.xlabel(f'time [{self.time_units}]')
        plt.ylabel('magnification')
        # TODO: add a more info on the plotting tool


if __name__ == "__main__":

    s = Simulator()
    s.load_matrices()
    s.calc_lightcurve(star_mass=0.4, lens_mass=1.5, lens_type='NS', inclination=89.8, semimajor_axis=0.005)
    syst = s.output_system()
    syst.plot()

    # d = s.position_radii[::2]
    # mag1 = transfer_matrix.radial_lightcurve(
    #     source_radius=s.source_size,
    #     occulter_radius=s.occulter_size,
    #     distances=d,
    #     pixels=1e4,
    # )
    # mag2 = transfer_matrix.radial_lightcurve(
    #     source_radius=s.source_size,
    #     occulter_radius=s.occulter_size,
    #     distances=d,
    #     pixels=1e5,
    # )
    # mag3 = transfer_matrix.radial_lightcurve(
    #     source_radius=s.source_size,
    #     occulter_radius=s.occulter_size,
    #     distances=d,
    #     pixels=1e7,
    #     circle_points=1e7,
    # )
    #
    # mag_ps = transfer_matrix.point_source_approximation(d)
    #
    # plt.plot(s.position_radii, s.magnifications, '-*', label='matrix')
    # plt.plot(d, mag1, '-+', label='pixels= 1e4')
    # plt.plot(d, mag2, '-x', label='pixels= 1e6')
    # plt.plot(d, mag3, '-o', label='pixels= 1e8')
    # plt.plot(d, mag_ps, '-', label='point source')
    # plt.xlabel(f'time [{s.time_units}]')
    # plt.ylabel('magnification')
    # plt.legend()
    # plt.show()


