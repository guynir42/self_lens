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
        self.name = name.upper()
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

        system.flare_prob[self.name] = np.array([])  # detection prob. for a flare, at best timing
        system.visit_prob[self.name] = np.array([])  # detection prob. for a single visit
        system.total_prob[self.name] = np.array([])  # multiply prob. by number of total visits per year in all sky
        system.effective_volumes[self.name] = 0.0  # volume covered by all visits to all sky (weighed by probability)
        system.num_detections[self.name] = np.array([])  # how many detections (on average)

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
        lc = (system.magnifications - 1) * dilution
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
                # raise ValueError('All lightcurve points are above precision level!')
                print('All lightcurve points are above precision level!')
                end_idx = len(det_lc) - 1
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

        best_precision_idx = np.argmax(self.precision)
        if hasattr(self.precision, '__len__'):
            best_precision = self.precision[best_precision_idx]
            best_flare_time = flare_time_per_distance[best_precision_idx]
        else:  # assume scalar precision (same for all magnitudes)
            best_precision = self.precision
            best_flare_time = flare_time_per_distance

        period = system.orbital_period * 3600  # shorthand
        (peak_prob, mean_prob, num_detections) = self.calc_prob(lc, ts, best_precision, best_flare_time, period)

        if t_exp > period:  # this only applies to simple S/N case of diluted signal
            pass  # TODO: figure out this case later

        system.flare_prob[self.name] = peak_prob
        system.visit_prob[self.name] = mean_prob
        system.num_detections[self.name] = num_detections

        # the probability to find at least one flare (cannot exceed 1)
        system.total_prob[self.name] = 1 - (1 - mean_prob) ** num_visits_per_year

        # if single visit prob. is large, can accumulate more effective volume than volume
        system.effective_volumes[self.name] = np.sum(system.total_volumes[self.name] * system.visit_prob[self.name])

    def calc_prob(self, lc, ts, precision, t_flare, period):
        """
        For a given lightcurve calculate the S/N and time coverage.
        In some cases the S/N is just one number, while in other cases
        it is an array with values for each time offset between the
        exposure and the flare time.
        The S/N is converted to probability, either the peak_prob,
        which is the probability to find a peak assuming perfect timing
        of exposure on top of the flare, and the mean_prob which is
        the average probability to find a flare assuming a uniform
        distribution of timing offsets over the entire orbit.

        :param lc: float array
            lightcurve array of the flare.
        :param ts: float array
            timestamps array (must be the same size as "lc", must be uniformly sampled).
        :param precision: float scalar
            the best precision of the survey, to find the S/N and rescale to all other values of precision.
        :param t_flare: float scalar
            the maximal flare duration, measured at the best precision, to set the scale for S/N calculations.
        :param period: float scalar
            the orbital period of the flares, in seconds

        :return: 3-tuple
            peak_prob: the probability to find the flare if the timing is ideal
                       (i.e., the exposure is right on the peak).
           mean_prob: the probability to find a flare, averaged over all possible
                       timing offsets in the orbital period (uniform phase).
                       If the series length is longer than one orbit, then the
                       probability is for getting at least one detection.
           num_detections: average number of detections per visit.
                           For a single exposure, this is just equal to mean_prob.
                           For multiple exposures, if the series is larger than
                           the orbital period, the number of detections is set
                           by the average number of flares in a series,
                           multiplied by the peak probability
                           (so it assumes the peak probability is the same
                           for each flare in the series).

        """

        t_exp = self.exposure_time
        t_dead = self.dead_time
        t_series = (t_exp + t_dead) * self.series_length

        # find the S/N for the best precision, then scale it for each distance
        # choose the detection method depending on the flare time and exposure time
        if t_flare * 10 < t_exp:  # flare is much shorter than exposure time
            signal = np.sum(lc[1:] * np.diff(ts))  # total signal in flare (skip 1st bin of LC to multiply with diff)
            noise = t_exp * precision  # total noise in exposure
            snr = signal / noise
            snr = np.array([snr])  # put this into a 1D array
            coverage = t_exp  # the only times where prob>0 is inside the exposure

        elif t_exp * 10 < t_flare < t_series / 10:  # exposures are short and numerous: can use matched-filter
            snr = np.sqrt(np.sum(lc ** 2)) / precision  # sum over different exposures hitting places on the LC
            snr = np.array([snr])  # put this into a 1D array
            coverage = t_series  # anywhere along the series has the same S/N (ignoring edge effects)
            # print(f'snr= {snr} | coverage= {coverage}')

        else:  # all other cases require sliding window
            dt = ts[1] - ts[0]  # assume timestamps are uniformly sampled in the simulated LC
            N_exp = int(np.ceil(t_exp / dt))  # number of time steps in single exposure
            single_exposure_snr = np.convolve(lc, np.ones(N_exp), mode='same') / N_exp / precision
            coverage = dt * len(single_exposure_snr)  # all points with non-zero S/N have prob>0

            if self.series_length == 1:
                snr = single_exposure_snr
            else:
                # Need to calculate the matched-filter result for several exposures.
                # For a given exposure start time relative to the flare,
                # we need to sum the (S/N)**2 curve in jumps of N_btw_repeats.
                # Essentially this means folding the (S/N)**2 curve over N_btw_repeats,
                # and then saving the total (matched-filter) S/N for each offset inside
                # one exposure.
                N_btw_repeats = int(np.ceil((t_exp + t_dead) / dt))  # number of steps between repeat exposures
                snr_square = single_exposure_snr ** 2
                number_bins = (len(snr_square) // N_btw_repeats + 1) * N_btw_repeats
                snr_square = np.pad(snr_square, (0, number_bins - len(snr_square)))
                snr_square_reshaped = np.reshape(snr_square, (-1, N_btw_repeats))

                if self.series_length >= snr_square_reshaped.shape[0]:  # series is longer than flare time
                    snr = np.sqrt(np.sum(snr_square_reshaped, axis=0))  # S/N for each offset of the exposure
                else:  # only part of the flare is covered by series, need multiple shifts

                    before = self.series_length // 2
                    after = (self.series_length + 1) // 2
                    # add a few additional time bins before/after the flare, for multiple exposures
                    snr_square_reshaped = np.pad(snr_square_reshaped, ((before, after), (0, 0)))

                    snr = np.zeros(snr_square_reshaped.shape)
                    for i in range(snr.shape[0]):
                        idx_low = max(i - before, 0)
                        idx_high = min(i + after, snr.shape[0])
                        snr[i, :] = np.sqrt(np.sum(snr_square_reshaped[idx_low:idx_high + 1, :], axis=0))
                    snr = np.reshape(snr, (1, -1))[0]  # matched-filter result per time shift
                    coverage = dt * snr.shape[0]  # adjust the coverage to the additional time bins

        # snr must be a 1D array of S/N for each time offset (possibly with one element)
        # now add a dimension for each precision value
        snr = np.expand_dims(snr, axis=0)  # make this a 2D array, axes 0: distances, axis 1: timestamps
        snr = snr * self.precision / precision  # broadcast to various values of precision

        # snr is 2D, with axis 0 for precision values, axis 1 for time offsets (each may be 1 element)
        # now calculate the probabilities

        prob = 0.5 * (1 + scipy.special.erf(snr - self.threshold))  # assume Gaussian distribution
        peak_prob = np.max(prob, axis=1)  # peak brightness of the flare assuming perfect timing

        # find the average probability and number of detections

        if t_series < period:  # assume survey hits random phases of the orbit every time
            mean_prob = np.mean(prob, axis=1)  # average the probability over multiple time shifts

            # dilute the det. prop. by the duty cycle (all the time the exposure may have been outside the event)
            mean_prob *= coverage / period

            # there's no way to see more than one flare in this series,
            # so the average number of detections is just the probability to see one
            num_detections = mean_prob

        else:  # assume full coverage in a single series
            # if the flare occurs multiple times in a single visit/series, we must account for that
            # the number of flares in the series is between N and N+1 (depending on fraction of period)
            num_flares_in_series = t_series // period
            fraction = (t_series - num_flares_in_series * period) / period  # fraction of period after num_flares were seen

            # a simplifying assumption that for multiple flares in a series,
            # the peak_prob is always seen for each flare.
            prob1 = 1 - (1 - peak_prob) ** num_flares_in_series  # at least one detection
            prob2 = 1 - (1 - peak_prob) ** (num_flares_in_series + 1)  # at least one detection
            mean_prob = prob1 * (1 - fraction) + prob2 * fraction  # weighted average of the two options

            # to count the average number of detections assume each detection has the same (peak) probability
            num_detections = peak_prob * (num_flares_in_series + fraction)  # average of N and N+1 with weight

        return peak_prob, mean_prob, num_detections

    def visit_prob_all_declinations(self, sim, num_points=1e4):
        """
        Calculate the visit probability for different declinations,
        with a loop over many declination values.
        The best_prob is usually the visit prob for declination=0.
        The total_prob is the weighted average probability given
        the uniform distribution of sin(dec)=cos(i).

        :param sim:
            A working Simulator object from simulator module.
        :param num_points:
            The number of declination points to sample
            between 0 and 90 degrees. In general, the loop
            never samples all points, because at low declinations
            of a few degrees (at most) the probability drops to
            zero and the loop is cut short.

        :return:
            a 2-tuple with best_prob and total_prob.
        """

        num_points = int(num_points)

        dec = np.linspace(0, 90, num_points)
        prob = np.zeros(dec.shape)

        for i, d in enumerate(dec):
            sim.calculate(declination=d)
            self.apply_detection_statistics(sim.syst)
            if sim.syst.visit_prob[self.name]:
                prob[i] = sim.syst.visit_prob[self.name]
            else:
                break

        # marginalize over all angles
        total_prob = np.sum(np.cos(np.deg2rad(dec)) * prob) / np.sum(np.cos(np.deg2rad(dec)))
        best_prob = np.max(prob)

        return best_prob, total_prob


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
        },
        'LSST': {
            'name': 'LSST',
            'telescope': 'Vera Rubin 8.4m',
            'location': 'south',
            'longitude': None,
            'latitude': None,
            'elevation': None,
            'field_area': 10,
            'footprint': 0.5,
            'cadence': 10,
            'duty_cycle': 0.2,
            'exposure_time': 15,
            'series_length': 2,
            'filter': 'r',
            'limmag': 24.5,
            'precision': 0.01,
        },
        'CURIOS': {
            'name': 'CuRIOS',
            'location': 'space',
            'longitude': None,
            'latitude': None,
            'elevation': None,
            'field_area': 5 ** 2 * np.pi,  # 10 deg f.o.v diameter
            'footprint': 1.0,
            'cadence': 12,
            'duty_cycle': 1.0,
            'exposure_time': 10,
            'filter': 'i',
            'limmag': 18.0,
            'precision': 0.01,
            'series_length': 10000,
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
    sim.calculate(
        lens_mass=2.0,
        star_mass=0.5,
        star_temp=7000,
        declination=0.0,
        semimajor_axis=0.001,
    )

    sim.syst.plot()

    ztf = Survey('ztf')
    ztf.series_length = 1
    ztf.exposure_time = 30
    ztf.dead_time = 0

    ztf.apply_detection_statistics(sim.syst)
    name = "ZTF"
    if len(sim.syst.total_prob):
        print(f'flare prob= {np.max(sim.syst.flare_prob[name]):.2e} | '
              f'visit prob= {np.max(sim.syst.visit_prob[name]):.2e} | '
              f'num det= {np.max(sim.syst.num_detections[name])} | '
              f'max dist= {max(sim.syst.distances[name]):.2f} pc | '
              f'volume: {np.sum(sim.syst.total_volumes[name]):.2e} pc^3 | '
              f'effective vol.= {sim.syst.effective_volumes[name]:.2e} pc^3')
    else:
        print('Object is too faint to observe.')

    cu = Survey('curios')

    t0 = timer()
    cu.apply_detection_statistics(sim.syst)
    print(f'Time to apply CuRIOS detection: {timer() - t0}s')
    name = 'CURIOS'
    if len(sim.syst.total_prob):
        print(f'flare prob= {np.max(sim.syst.flare_prob[name]):.2e} | '
              f'visit prob= {np.max(sim.syst.visit_prob[name]):.2e} | '
              f'num det= {np.max(sim.syst.num_detections[name])} | '
              f'max dist= {max(sim.syst.distances[name]):.2f} pc | '
              f'volume: {np.sum(sim.syst.total_volumes[name]):.2e} pc^3 | '
              f'effective vol.= {sim.syst.effective_volumes[name]:.2e} pc^3')
    else:
        print('Object is too faint to observe.')

