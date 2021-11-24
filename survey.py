"""
defines a survey class that gets a lightcurve and other metadata from the simulator
and from that figures out the probability of detection for those system parameters.
"""

import numpy as np
import scipy.special
from timeit import default_timer as timer

import simulator

MINIMAL_DISTANCE_PC = 10

class Survey:
    def __init__(self, name, **kwargs):
        """
        A survey must have some basic information defined so it can be used
        to estimate the detectability of different self lensing systems.
        Each parameter is defined below, but here's a list of the necessary

        Input keywords necessary to generate a valid survey are:
        location, field_area, exposure_time, cadence, footprint, limmag, precision,

        Another option is to provide one of the following preset survey names:
        ZTF, TESS, ASASSN, W-FAST,
        which will automatically fill the required fields using the default data.
        Any keywords provided in addition to a name that's on the default list,
        will be used to override the default values.
        """

        setup_default_survey(name, kwargs)

        # just some generic info on the survey
        self.name = name
        self.telescope = kwargs.get('telescope')
        self.location = kwargs.get('location')  # can be "north", "south", or "space"
        self.longitude = kwargs.get('longitude')  # degrees
        self.latitude = kwargs.get('latitude')  # degrees
        self.elevation = kwargs.get('elevation')  # meters

        # survey statistics
        self.field_area = kwargs.get('field_area')  # in square degrees
        self.exposure_time = kwargs.get('exposure_time')  # in seconds
        self.dead_time = kwargs.get('dead_time', 0)  # time between exposures in a series
        self.series_length = kwargs.get('series_length', 1)  # number of images taken continuously in one visit
        self.cadence = kwargs.get('cadence')  # how much time (days) passes between starts of visits
        self.footprint = kwargs.get('footprint')  # what fraction of the sky is surveyed (typically 0.5 or 1)
        self.duration = 1  # years (setting default 1 is to compare the number of detections per year)
        self.threshold = 3  # need 3 sigma for a detection by default

        # filters and image depths
        self.limmag = kwargs.get('limmag')  # objects fainter than this would not be seen at all  (faintest magnitude)
        self.precision = kwargs.get('precision')  # photometric precision per image (best possible value)
        self.mag_list = None  # list of magnitudes over which the survey is sensitive
        self.prec_list = None  # list of photometric precision for each mag_list item
        self.distances = kwargs.get('distances', np.geomspace(MINIMAL_DISTANCE_PC, 50000, 50, endpoint=True)[1:])
        self.filter = kwargs.get('filters', 'V')
        self.wavelength = None  # central wavelength (in nm)
        self.bandpass = None  # assume top-hat (in nm)

        (self.wavelength, self.bandpass) = default_filter(self.filter)

        if 'wavelength' in kwargs:
            self.wavelength = kwargs['wavelength']
        if 'bandwidth' in kwargs:
            self.bandpass = kwargs['bandwidth']

        if self.mag_list is not None and self.prec_list is not None:
            self.limmag = max(self.mag_list)
            self.precision = max(self.prec_list)

        list_needed_values = ['location', 'field_area', 'exposure_time', 'cadence', 'footprint', 'limmag', 'precision']
        for k in list_needed_values:
            if getattr(self, k) is None:
                raise ValueError(f'Must have a valid value for "{k}".')

    # def make_lightcurve(self, system):
    #     """
    #     Make a demonstration lightcurve, covering 100 points on either side of the event center
    #
    #     :param system:
    #         A system object.
    #     :return:
    #         A tuple with timestamps, lightcurves (in units of mag) and errors.
    #     """
    #     # start by down sampling the event lightcurve:
    #
    #
    #     # ts = np.arange(-100, 100, 1) * self.cadence * 24 * 3600 * simulator.translate_time_units
    def effective_volume(self, system):
        """
        Find the detection probability for the given system and this survey,
        assuming the system is placed at different distances.
        :param system:
            A list of simulator.System objects.
            Each one should contain the physical properties of the system:
            - mass of the lens and source.
            - total (bolometric) flux of lens and source.
            - temperatures of lens and source.
            - semimajor axis and period
            - inclination
            -
            In addition, each System should also have a magnification lightcurve.
            Finally, each System has a dictionary of results that can be filled by
            each Survey object. This includes one dictionary called distances,
            one dictionary called volumes
            one dictionary called det_prob.
            Each item in the dictionary would be for a different survey.
            Later more dictionaries may be added to contain the expected number of

        """
        t0 = timer()

        # these are the null values for a system that cannot be detected at all (i.e., an early return statement)
        syst.distances[self.name] = np.array([])
        syst.volumes[self.name] = np.array([])
        syst.apparent_mags[self.name] = np.array([])
        syst.total_volumes[self.name] = np.array([])
        syst.flare_durations[self.name] = 0.0
        syst.dilutions[self.name] = 1.0
        syst.visit_prob[self.name] = np.array([])
        syst.total_prob[self.name] = np.array([])
        syst.effective_volumes[self.name] = 0.0

        # first figure out what magnitudes and precisions we can get on this system:
        (bol_correct, dilution) = syst.bolometric_correction(self.wavelength, self.bandpass)
        syst.dilutions[self.name] = dilution

        dist = []
        mag = []
        for i, d in enumerate(self.distances):
            new_mag = system.bolometric_mag(d) + bol_correct
            if new_mag > self.limmag:
                break
            dist.append(d)
            mag.append(new_mag)

        if len(mag) == 0:
            return 0  # cannot observe this target at all, it is always below the limiting magnitude

        dist = np.array(dist)
        mag = np.array(mag)

        if self.prec_list is None:
            prec = np.ones(len(mag)) * self.precision
        else:
            prec = np.array(self.prec_list[:len(mag)])

        prec = np.expand_dims(prec, axis=1)  # make this 2D but make it a column

        # figure out the S/N for this lightcurve, and compare it to the precision we have at each distance
        # shorthand for lightcurve and timestamps
        lc = (system.magnifications - 1) / dilution
        ts = system.timestamps
        mid_idx = np.argmax(lc)

        if system.time_units == 'minutes':
            ts *= 60
        if system.time_units == 'hours':
            ts *= 3600
        if system.time_units == 'days':
            ts *= 3600 * 24
        if system.time_units == 'years':
            ts *= 3600 * 24 * 365.25

        # determine how much time the lensing event is above the survey precision
        det_lc = lc[mid_idx:]  # detection lightcurve
        if not np.any(det_lc > self.precision):
            return 0  # this light curve is below our photometric precision at all times

        if np.all(det_lc > self.precision):
            print('All lightcurve points are above precision level!')
            end_idx = len(lc[mid_idx:])
        else:
            end_idx = np.argmin(det_lc > self.precision)  # first place lc dips below precision
        # print(f'len(lc)= {len(lc)} | mid_idx= {mid_idx} | end_idx= {end_idx}')

        t_flare = 2 * (ts[end_idx + mid_idx - 1] - ts[mid_idx])  # flare duration relative to this survey's precision
        t_exp = self.exposure_time
        t_dead = self.dead_time
        t_series = (t_exp + t_dead) * self.series_length

        # update the results of distances, volumes and magnitudes
        syst.distances[self.name] = dist
        syst.apparent_mags[self.name] = mag
        solid_angle = self.field_area * (np.pi / 180) ** 2
        num_visits_in_survey = (self.duration * 365.25) // self.cadence
        num_fields = (self.cadence * 24 * 3600) // t_series
        syst.volumes[self.name] = np.diff(solid_angle / 3 * dist ** 3, prepend=MINIMAL_DISTANCE_PC)
        syst.total_volumes[self.name] = syst.volumes[self.name] * num_fields
        syst.flare_durations[self.name] = t_flare

        # choose the detection method depending on the flare time and exposure time
        if t_flare * 10 < t_exp:  # flare is much shorter than exposure time
            signal = np.sum(lc[1:] / self.precision * np.diff(ts))
            noise = t_exp
            snr = signal / noise
            snr = np.array([[snr]])  # put this into a 2D array
            coverage = t_exp  # the only times where p>0 is inside the exposure

        elif t_exp * 10 < t_flare < t_series / 10:  # can use matched-filter
            snr = np.sum(snr ** 2) / self.precision  # sum over different exposures hitting places on the LC
            snr = np.array([[snr]])  # put this into a 2D array
            coverage = t_series  # anywhere along the series has the same S/N (ignoring edge effects)
            print(f'snr= {snr} | coverage= {coverage}')
        else:  # all other cases require sliding window

            dt = min(t_flare, t_exp) / 10  # time step
            N_exp = int(np.ceil(t_exp/dt))
            single_exposure_signal = np.convolve(lc, np.ones(N_exp) / N_exp) / self.precision

            if self.series_length > 1:
                N_series = int(np.ceil(self.series_length * t_exp / dt))
                N_repeats = int(np.ceil(t_exp + t_dead) / dt)
                snr_pad = np.pad(single_exposure_signal, (1, N_series))
                snr_square = np.zeros(snr_pad.shape)
                for i in range(self.series_length):  # add each exposure S/N in quadrature
                    snr_square += np.roll(snr_pad ** 2, i * N_repeats)

                snr = np.sqrt(snr_square)  # sqrt of (S/N)**2 summed is the matched filter result

            else:
                snr = single_exposure_signal

            snr = np.expand_dims(snr, axis=0)  # make this a 2D array
            coverage = dt * len(snr)  # all points with non-zero S/N have p>0
            # print(f'snr= {snr} | coverage= {coverage}')

        # check the S/N against precision at each distance and time
        snr = snr * self.precision / prec  # should be 2D, with axes 0: distances, 1: timestamps

        prob = 0.5 * (1 + scipy.special.erf(snr - self.threshold))  # assume Gaussian distribution

        if t_series > system.orbital_period * 3600:  # assume full coverage in a single series
            prob = np.max(prob, axis=1)  # peak brightness is detected?

            # if the flare occurs multiple times in a single visit/series, we must account for that
            num_flares_in_series = (t_series * 3600) // system.orbital_period
            prob = 1 - (1 - prob) ** num_flares_in_series

        else:  # assume survey hits random phases of the orbit every time
            prob = np.mean(prob, axis=1)  # average the probability over multiple time shifts
            # print(f'snr.shape: {snr.shape} | prec.shape= {prec.shape} | prob.shape= {prob.shape}')

            # dilute the det. prop. by the duty cycle (all the time the exposure may have been outside the event)
            prob *= coverage / (system.orbital_period * 3600)

        syst.visit_prob[self.name] = prob
        syst.total_prob[self.name] = 1 - (1 - prob) ** num_visits_in_survey

        effective_volume = np.sum(syst.total_volumes[self.name] * syst.total_prob[self.name])
        syst.effective_volumes[self.name] = effective_volume

        return effective_volume


def setup_default_survey(name, kwargs):
    """
    Modify the dictionary kwargs to have all the details on the
    specific survey, using the default list of surveys we define below.
    """

    defaults = {
        'ZTF': {
            'name': 'ZTF',
            'telescope': 'P48',
            'location': 'north',
            'longitude': None,
            'latitude': None,
            'elevation': None,
            'field_area': 47,
            'footprint': 0.5,
            'cadence': 3,
            'exposure_time': 30,
            'filter': 'r',
            'limmag': 20.5,
            'precision': 0.01,
        }
    }

    name = name.upper().replace('_', '')

    if name not in defaults:
        # raise KeyError(f'Could not find name "{kwargs["name"]}" in defaults. ')
        return  # this survey is not a known default survey on the list

    # replace only arguments that were NOT given by the user
    for k, v in defaults[name].items():
        if k not in kwargs:
            kwargs[k] = v


def default_filter(filter_name):
    """
    Return the central wavelength and bandpass (in nm) for some well known filters.
    :param filter_name:
        Name of the filter. Some common defaults are "V", "r", "i", "F500W".

    :return:
        A tuple of (wavelength, bandwidth), both in nm

    reference: https://en.wikipedia.org/wiki/Photometric_system

    """
    filters = {
        'U': (365, 66),
        'B': (445, 94),
        'G': (464, 128),
        'V': (551, 88),
        'R': (658, 138),
        'I': (806, 149),
        'F500W': (500, 200),
    }
    # TODO: make sure these numbers are correct!

    if filter_name.upper() not in filters:
        # raise KeyError(f'Unknonwn filter name "{filter_name}". Use "V" or "r" etc. ')
        return None, None

    return filters[filter_name.upper()]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    sim = simulator.Simulator()
    sim.load_matrices()
    sim.calc_lightcurve(
        lens_mass=30,
        lens_type='BH',
        lens_temp=5000,
        # lens_size=0.01,
        star_mass=0.5,
        star_temp=4000,
        star_type='MS',
        star_size=1.0,
        inclination=89.99,
        semimajor_axis=1,
        time_units='hours',
    )
    syst = sim.output_system()
    syst.plot()

    ztf = Survey('ztf')
    ztf.series_length = 1
    ztf.exposure_time = 30
    ztf.dead_time = 0
    # ztf.wavelength = None
    # print(f'S/N= {ztf.det_prob(syst)}s')

    ztf.effective_volume(syst)
    if len(syst.total_prob):
        print(f'mean prob= {np.mean(syst.total_prob["ztf"]):.2e} | max dist= {max(syst.distances["ztf"]):.2f} pc | '
              f'volume: {np.sum(syst.total_volumes["ztf"]):.2e} pc^3 | '
              f'effective vol.= {syst.effective_volumes["ztf"]:.2e} pc^3')
    else:
        print('Object is too faint to observe. ')