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
        self.duty_cycle = kwargs.get('duty_cycle')  # fraction on-sky, excluding dead time, daylight, weather
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

        (self.wavelength, self.bandpass) = simulator.default_filter(self.filter)

        if 'wavelength' in kwargs:
            self.wavelength = kwargs['wavelength']
        if 'bandwidth' in kwargs:
            self.bandpass = kwargs['bandwidth']

        if self.mag_list is not None and self.prec_list is not None:
            self.limmag = max(self.mag_list)
            self.precision = max(self.prec_list)

        list_needed_values = ['location', 'field_area', 'exposure_time', 'duty_cycle', 'footprint', 'limmag', 'precision']
        for k in list_needed_values:
            if getattr(self, k) is None:
                raise ValueError(f'Must have a valid value for "{k}".')

    def apply_detection_statistics(self, system):
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
            one dictionary called volumes, one dictionary called det_prob.
            Each item in the dictionary would be for a different survey.

        """
        t0 = timer()

        # these are the null values for a system that cannot be detected at all (i.e., an early return statement)
        system.distances[self.name] = np.array([])  # distances (in pc) at which this system is detectable
        system.volumes[self.name] = np.array([])  # single visit space volume for those distances * field of view (pc^3)
        system.apparent_mags[self.name] = np.array([])  # magnitude of the system at given distance (in survey's filter)
        system.total_volumes[self.name] = np.array([])  # volume for all visits in a year of observations
        system.flare_durations[self.name] = np.array([])  # duration the flare is above threshold, at each distance
        system.dilutions[self.name] = 1.0  # dilution factor of two bright sources, at this filter
        system.visit_prob[self.name] = np.array([])  # detection prob. for a single visit
        system.total_prob[self.name] = np.array([])  # multiply prob. by number of total visits per year in all sky
        system.effective_volumes[self.name] = 0.0  # volume covered by all visits to all sky (weighed by probability)

        # first figure out what magnitudes and precisions we can get on this system:
        (bol_correct, dilution) = system.bolometric_correction(self.wavelength, self.bandpass)
        system.dilutions[self.name] = dilution

        dist = []
        mag = []
        for i, d in enumerate(self.distances):
            new_mag = system.bolometric_mag(d) - bol_correct
            if new_mag > self.limmag:
                break
            dist.append(d)
            mag.append(new_mag)

        if len(mag) == 0:
            return  # cannot observe this target at all, it is always below the limiting magnitude

        dist = np.array(dist)
        mag = np.array(mag)

        if self.prec_list is None:
            prec = np.ones(len(mag)) * self.precision
        else:
            prec = np.array(self.prec_list[:len(mag)])

        prec = np.expand_dims(prec, axis=1)  # this is still a vector but 2D to make it a column

        # figure out the S/N for this lightcurve, and compare it to the precision we have at each distance
        # shorthand for lightcurve and timestamps
        lc = (system.magnifications - 1) / dilution
        ts = system.timestamps
        mid_idx = len(lc) // 2

        # convert timestamps to seconds
        ts *= simulator.translate_time_units(system.time_units)

        # determine how much time the lensing event is above the survey precision
        det_lc = lc[mid_idx:]  # detection lightcurve

        if not np.any(det_lc > self.precision):
            return  # this light curve is below our photometric precision at all times

        flare_time_per_distance = []
        for p in prec:
            if np.all(det_lc > p):
                # this should not happen in general, if there are enough points outside the flare
                print('All lightcurve points are above precision level!')
                end_idx = len(lc[mid_idx:])
            else:
                end_idx = np.argmin(det_lc > p)  # first index where lc dips below precision

            # flare_time_per_distance.append(2 * (ts[end_idx + mid_idx - 1] - ts[mid_idx]))
            t1 = ts[mid_idx + end_idx - 1]
            t2 = ts[mid_idx + end_idx]
            l1 = det_lc[end_idx - 1]
            l2 = det_lc[end_idx]
            flare_time_per_distance = t1 + (p - l1) / (l2 - l1) * (t2 - t1)  # linear interpolation
            flare_time_per_distance *= 2  # symmetric lightcurve

        t_flare = max(flare_time_per_distance)  # flare duration relative to survey's best precision
        t_exp = self.exposure_time
        t_dead = self.dead_time
        t_series = (t_exp + t_dead) * self.series_length

        # update the results of distances, volumes and magnitudes
        system.distances[self.name] = dist
        system.apparent_mags[self.name] = mag
        solid_angle = self.field_area * (np.pi / 180) ** 2
        # num_visits_in_survey = (self.duration * 365.25) // self.cadence
        # num_fields = (self.cadence * 24 * 3600) // t_series
        num_visits_per_year = 365.25 * 24 * 3600 * self.duty_cycle / t_series  # in all sky locations combined
        system.volumes[self.name] = np.diff(solid_angle / 3 * dist ** 3, prepend=solid_angle / 3 * MINIMAL_DISTANCE_PC ** 3)  # single visit
        system.total_volumes[self.name] = system.volumes[self.name] * num_visits_per_year  # for all visits, per year
        system.flare_durations[self.name] = flare_time_per_distance

        # find the S/N for the best precision, then scale it for each distance
        (snr, coverage) = calc_snr_and_coverage(lc, ts, self.precision, t_flare, t_exp, t_dead, self.series_length)

        period = system.orbital_period * 3600  # shorthand
        if period < t_exp:  # this only applies to simple S/N case of diluted signal
            pass  # TODO: figure out this case later
        # cases where the series is longer than the period are considered in "multi visit prob." below

        # adjust the S/N with precision at each distance and time
        snr = np.expand_dims(snr, axis=0)  # make this a 2D array
        snr = snr * self.precision / prec  # axes 0: distances, axis 1: timestamps

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

        system.visit_prob[self.name] = prob
        # TODO: add multi-dectecion probability for very long series

        # the probability to find at least one flare (cannot exceed 1)
        system.total_prob[self.name] = 1 - (1 - prob) ** num_visits_per_year

        # if single visit prob. is large, can accumulate more effective volume than volume
        system.effective_volumes[self.name] = np.sum(system.total_volumes[self.name] * system.visit_prob[self.name])

    def visit_prob_all_declinations(self, sim, num_points=1e4):
        num_points = int(num_points)

        dec = np.linspace(0, 90, num_points)
        prob = np.zeros(dec.shape)

        for i, d in enumerate(dec):
            sim.calculate(declination=d)
            self.apply_detection_statistics(sim.syst)
            if sim.syst.visit_prob['ZTF']:
                prob[i] = sim.syst.visit_prob['ZTF']
            else:
                break

        # marginalize over all angles
        total_prob = np.sum(np.cos(np.deg2rad(dec)) * prob) / np.sum(np.cos(np.deg2rad(dec)))
        peak_prob = np.max(prob)

        return peak_prob, total_prob


def calc_snr_and_coverage(lc, ts, precision, t_flare, t_exp, t_dead=0, series_length=1):
    """
    For a given lightcurve calculate the S/N and time coverage.
    The S/N is the expected signal, either for a general exposure
    (single or series) or, if using a sliding window, it will output
    the S/N for each offset between the mid-exposure and mid-flare time.

    :param lc: float array
        lightcurve array of the flare.
    :param ts: float array
        timestamps array (must be the same size as "lc", must be uniformly sampled).
    :param precision: float scalar
        the base photometric precision of the survey, in fractional RMS units.
    :param t_flare: float scalar
        duration of the flare (in seconds).
    :param t_exp: float scalar
        duration of the exposure (in seconds).
    :param t_dead:
        additional time (in seconds) between consecutive exposures in a series (e.g., readout time).
    :param series_length: int scalar
        number of exposures in a single visit / series.

    :return: 2-tuple
        - an array of S/N values for different time offsets.
          In cases where all offsets are the same (ignoring edges)
          then a single value is returned, in a 1D array.
        - the coverage value (in seconds) describing what segment
          of the orbit has been used in the S/N calculation.
          For getting the average S/N, the remaining orbit time
          that is further from the flare is assumed to have S/N=0.

    """

    t_series = series_length * (t_exp + t_dead)

    # choose the detection method depending on the flare time and exposure time
    if t_flare * 10 < t_exp:  # flare is much shorter than exposure time
        signal = np.sum(lc[1:] * np.diff(ts))  # total signal in flare (skip edge of LC to multiply with diff)
        noise = t_exp * precision  # total noise in exposure
        snr = signal / noise
        snr = np.array([snr])  # put this into a 1D array
        coverage = t_exp  # the only times where prob>0 is inside the exposure

    elif t_exp * 10 < t_flare < t_series / 10:  # can use matched-filter
        snr = np.sqrt(np.sum(lc ** 2)) / precision  # sum over different exposures hitting places on the LC
        snr = np.array([snr])  # put this into a 1D array
        coverage = t_series  # anywhere along the series has the same S/N (ignoring edge effects)
        # print(f'snr= {snr} | coverage= {coverage}')
    else:  # all other cases require sliding window

        # dt = min(t_flare, t_exp) / 10  # time step
        dt = ts[1] - ts[0]  # assume timestamps are uniformly sampled in the simulated LC
        N_exp = int(np.ceil(t_exp / dt))  # number of time steps in single exposure
        single_exposure_snr = np.convolve(lc, np.ones(N_exp)) / N_exp / precision

        if series_length > 1:
            N_btw_repeats = int(np.ceil((t_exp + t_dead) / dt))  # number of steps between repeat exposures
            N_series = N_btw_repeats * series_length  # number of steps in entire series
            snr_pad = np.pad(single_exposure_snr, (1, N_series))
            snr_square = np.zeros(snr_pad.shape)
            for i in range(series_length):  # add each exposure S/N in quadrature
                snr_square += np.roll(snr_pad ** 2, i * N_btw_repeats)

            snr = np.sqrt(snr_square)  # sqrt of (S/N)**2 summed is the matched filter result

        else:
            snr = single_exposure_snr

        coverage = dt * len(snr)  # all points with non-zero S/N have prob>0

    return snr, coverage


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
            'duty_cycle': 0.2,
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



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    sim = simulator.Simulator()
    sim.load_matrices()
    sim.calculate(
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

    sim.syst.plot()

    ztf = Survey('ztf')
    ztf.series_length = 1
    ztf.exposure_time = 30
    ztf.dead_time = 0
    # ztf.wavelength = None
    # print(f'S/N= {ztf.det_prob(syst)}s')

    ztf.effective_volume(sim.syst)
    if len(sim.syst.total_prob):
        print(f'mean prob= {np.mean(sim.syst.total_prob["ztf"]):.2e} | '
              f'max dist= {max(sim.syst.distances["ztf"]):.2f} pc | '
              f'volume: {np.sum(sim.syst.total_volumes["ztf"]):.2e} pc^3 | '
              f'effective vol.= {sim.syst.effective_volumes["ztf"]:.2e} pc^3')
    else:
        print('Object is too faint to observe.')
