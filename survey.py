"""
defines a survey class that gets a lightcurve and other metadata from the simulator
and from that figures out the probability of detection for those system parameters.
"""

import numpy as np
import scipy.special
from timeit import default_timer as timer

import simulator

MIN_DIST_PC = 10


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
        self.num_visits = kwargs.get('num_visits')  # how many times each field is visited throughout survey life
        self.num_fields = kwargs.get('num_fields')  # how many different fields are observed by the survey
        self.cadence = kwargs.get('cadence')  # how much time (days) passes between starts of visits (do we need this?)
        self.slew_time = kwargs.get('slew_time', 0)  # time (in seconds) it takes to slew to new field at end of visit
        self.duty_cycle = kwargs.get('duty_cycle')  # fraction on-sky, excluding dead time, daylight, weather
        self.footprint = kwargs.get('footprint')  # what fraction of the sky is surveyed (typically 0.5 or 1)
        self.duration = 1  # years (setting default 1 is to compare the number of detections per year)

        # filters and image depths
        self.limmag = kwargs.get('limmag')  # objects fainter than this would not be seen at all  (faintest magnitude)
        self.precision = kwargs.get('precision')  # photometric precision per image (best possible value)
        self.threshold = 3  # need 3 sigma for a detection by default
        self.mag_list = kwargs.get('mag_list')  # list of magnitudes over which the survey is sensitive
        self.prec_list = kwargs.get('prec_list')  # list of photometric precision for each mag_list item
        self.distances = kwargs.get('distances', np.geomspace(MIN_DIST_PC, 50000, 50, endpoint=True)[1:])
        self.filter = kwargs.get('filter', 'V')
        self.wavelength = None  # central wavelength (in nm)
        self.bandpass = None  # assume top-hat (in nm)

        (self.wavelength, self.bandpass) = simulator.default_filter(self.filter)

        if 'wavelength' in kwargs:
            self.wavelength = kwargs['wavelength']
        if 'bandwidth' in kwargs:
            self.bandpass = kwargs['bandwidth']

        if self.mag_list is not None and self.prec_list is not None:
            self.limmag = max(self.mag_list)
            self.precision = min(self.prec_list)

        self.find_fields_and_visits()

        list_needed_values = ['location', 'field_area', 'exposure_time', 'num_visits', 'num_fields', 'limmag', 'precision']
        for k in list_needed_values:
            if getattr(self, k) is None:
                raise ValueError(f'Must have a valid value for "{k}".')

    def find_fields_and_visits(self):
        """
        Use the cadence, series time, survey durations, etc,
        to figure out the number of fields and the number of visits
        to each field over the duration of the survey.
        """

        if self.num_visits is None and self.num_fields is None and self.series_length is not None:
            # validate that we have all the needed information
            for arg in ['cadence', 'duration', 'duty_cycle', 'series_length', 'exposure_time',
                        'slew_time', 'dead_time', 'field_area']:
                if getattr(self, arg) is None:
                    raise ValueError(f'Cannot find number of fields/visits without {arg}')

            visit_time = self.series_length * (self.exposure_time + self.dead_time) + self.slew_time
            self.num_fields = int(np.round(self.cadence * 24 * 3600 * self.duty_cycle / visit_time))
            self.num_visits = int(np.round(self.duration * 365.25 / self.cadence))

        # in case we know the cadence and want to figure out the series length
        elif self.num_visits is None and self.num_fields is None and self.series_length is None:
            # validate that we have all the needed information
            for arg in ['footprint', 'duration', 'duty_cycle', 'cadence',
                        'exposure_time', 'slew_time', 'dead_time', 'field_area']:
                if getattr(self, arg) is None:
                    raise ValueError(f'Cannot find number of fields/visits without {arg}')

            self.num_visits = int(np.round(self.duration * 365.25 / self.cadence))
            self.num_fields = int(np.round(self.footprint * 4 * 180 ** 2 / np.pi / self.field_area))
            visit_time = self.cadence * 24 * 3600 * self.duty_cycle / self.num_fields
            self.series_length = int(np.round((visit_time - self.slew_time) / (self.exposure_time + self.dead_time)))
        else:
            raise ValueError('Illegal combination of values for this survey!')

        self.footprint = self.num_fields * self.field_area / (4 * 180 ** 2 / np.pi)

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

        # these are the null values for a system that cannot be detected at all (i.e., an early return statement)
        system.distances[self.name] = np.array([])  # distances (in pc) at which this system is detectable
        system.volumes[self.name] = np.array([])  # single visit space volume for those distances * field of view (pc^3)
        system.apparent_mags[self.name] = np.array([])  # magnitude of the system at given distance (in survey's filter)
        system.precisions[self.name] = np.array([])  # precision measured for each apparent magnitude
        system.total_volumes[self.name] = np.array([])  # volume for all fields
        system.dilutions[self.name] = 1.0  # dilution factor of two bright sources, at this filter

        system.flare_durations[self.name] = np.array([])  # duration the flare is above threshold, at each distance
        system.flare_prob[self.name] = np.array([])  # detection prob. for a flare, at best timing
        system.visit_prob[self.name] = np.array([])  # detection prob. for a single visit
        system.total_prob[self.name] = np.array([])  # multiply prob. by number of total visits for a single field
        system.visit_detections[self.name] = np.array([])  # how many detections (on average) per visit
        system.total_detections[self.name] = np.array([])  # how many detections (on average) over all visits
        system.effective_volumes[self.name] = 0.0  # volume covered by all visits to all sky (weighed by probability)

        # first figure out what magnitudes and precisions we can get on this system:
        (abs_mag, dilution) = system.ab_mag(self.wavelength, self.bandpass, get_dilution=True)
        system.dilutions[self.name] = dilution

        dist = []
        mag = []
        for i, d in enumerate(self.distances):
            new_mag = system.apply_distance(abs_mag, d)  # TODO: add extinction
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
            prec = np.interp(mag, self.mag_list, self.prec_list)  # interpolate the precision at given magnitudes

        if len(prec) == 0:
            return  # cannot observe this target at all, we have no precision for this magnitude

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
                raise ValueError('All lightcurve points are above precision level!')
                # print('All lightcurve points are above precision level!')
                # end_idx = len(det_lc) - 1
            elif np.all(det_lc < p):
                flare_time_per_distance.append(0)
            else:
                end_idx = np.argmin(det_lc > p)  # first index where lc dips below precision

                t1 = ts[mid_idx + end_idx - 1]
                t2 = ts[mid_idx + end_idx]
                l1 = det_lc[end_idx - 1]
                l2 = det_lc[end_idx]
                t_flare = t1 + (p - l1) / (l2 - l1) * (t2 - t1)  # linear interpolation
                t_flare *= 2  # symmetric lightcurve
                flare_time_per_distance.append(t_flare)

        # get rid of distances, precisions and flare durations where the flare is totally undetectable
        flare_time_per_distance = np.array(flare_time_per_distance)
        idx = flare_time_per_distance > 0
        flare_time_per_distance = flare_time_per_distance[idx]
        prec = prec[idx]
        mag = mag[idx]
        dist = dist[idx]

        # update the results of distances, volumes and magnitudes
        system.distances[self.name] = dist
        system.apparent_mags[self.name] = mag
        solid_angle = self.field_area * (np.pi / 180) ** 2
        system.volumes[self.name] = np.diff(solid_angle / 3 * dist ** 3, prepend=solid_angle / 3 * MIN_DIST_PC ** 3)
        system.total_volumes[self.name] = system.volumes[self.name] * self.num_fields  # for all fields
        system.flare_durations[self.name] = flare_time_per_distance
        system.precisions[self.name] = prec

        best_precision_idx = np.argmax(prec) if len(prec) else []  # index of best precision
        best_precision = prec[best_precision_idx]
        best_flare_time = flare_time_per_distance[best_precision_idx]

        period = system.orbital_period * 3600  # shorthand
        if len(prec):  # only calculate probabilities for distances that have any chance of detection
            (peak_prob, mean_prob, num_detections) = self.calc_prob(lc, ts, prec, best_precision, best_flare_time, period)
            if self.exposure_time > period:  # this only applies to simple S/N case of diluted signal
                pass  # TODO: figure out this case later

            system.flare_prob[self.name] = peak_prob
            system.visit_prob[self.name] = mean_prob
            system.visit_detections[self.name] = num_detections

            # the probability to find at least one flare per system (cannot exceed 1)
            system.total_prob[self.name] = 1 - (1 - mean_prob) ** self.num_visits

            # average number of detections over many visits can exceed 1
            system.total_detections[self.name] = num_detections * self.num_visits

            # if total_detections is large, can accumulate more effective volume than volume
            # total_detections assumes the volume contains one system
            # to get the actual number of detections you need to multiply effective volume with spatial density
            system.effective_volumes[self.name] = np.sum(system.total_volumes[self.name] * system.total_detections[self.name])

        # consider what happens if the survey does exist in the list, should it be updated??
        if self.name not in (s.name for s in system.surveys):
            system.surveys.append(self)

    def calc_prob(self, lc, ts, precision, best_precision, best_t_flare, period):
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
        :param precision: float array
            precision array, for each distance the system can be.
        :param best_precision: float scalar
            the best precision of the survey, to find the S/N and rescale to all other values of precision.
        :param best_t_flare: float scalar
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
        if best_t_flare * 10 < t_exp:  # flare is much shorter than exposure time
            signal = np.sum(lc[1:] * np.diff(ts))  # total signal in flare (skip 1st bin of LC to multiply with diff)
            noise = t_exp * best_precision  # total noise in exposure
            snr = signal / noise
            snr = np.array([snr])  # put this into a 1D array
            coverage = t_exp  # the only times where prob>0 is inside the exposure

        elif t_exp * 10 < best_t_flare < t_series / 10:  # exposures are short and numerous: can use matched-filter
            snr = np.sqrt(np.sum(lc ** 2)) / best_precision  # sum over different exposures hitting places on the LC
            snr = np.array([snr])  # put this into a 1D array
            coverage = t_series  # anywhere along the series has the same S/N (ignoring edge effects)
            # print(f'snr= {snr} | coverage= {coverage}')

        else:  # all other cases require sliding window
            dt = ts[1] - ts[0]  # assume timestamps are uniformly sampled in the simulated LC
            N_exp = int(np.ceil(t_exp / dt))  # number of time steps in single exposure
            single_exposure_snr = np.convolve(lc, np.ones(N_exp), mode='same') / N_exp / best_precision
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
        snr = snr * best_precision / np.expand_dims(precision, axis=1)   # broadcast to various values of precision

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

    def binomial_prob(self, num_hits, num_tries, base_prob):
        binomials = scipy.special.comb(num_tries, num_hits, exact=True)
        probs = base_prob ** num_hits * (1 - base_prob) ** (num_tries - num_hits)
        return binomials * probs

    def binomial_cumulative(self, min_num_hits, num_tries, base_prob):
        # find the total prob. of finding all the num_hits lower than min_num_hits
        prob_sum = 0
        for i in range(min_num_hits):
            prob_sum += self.binomial_prob(i, num_tries, base_prob)

        # the reciprocal of that is the prob to get min_num_hits and all higher numbers
        return 1 - prob_sum

    def print(self):
        print(f'Survey name: {self.name}')
        print(f'filter: {self.filter} | limiting mag: {self.limmag:.2g} | precision: {self.precision:.3g}')
        print(f'duration: {self.duration:.2g} year | cadence: {self.cadence:.2g} day | '
              f'duty cycle: {self.duty_cycle:.2g} | area: {self.field_area:.2g} deg^2')
        print(f'exp: {self.exposure_time:.2g} s | dead time: {self.dead_time:.2g} s '
              f'| slew time: {self.slew_time:.2g} s | series length: {self.series_length}')
        print(f'num fields: {self.num_fields} | num visits: {self.num_visits} | '
              f'footprint: {self.footprint:.2g} sky')


def setup_default_survey(name, kwargs):
    """
    Modify the dictionary kwargs to have all the details on the
    specific survey, using the default list of surveys we define below.
    """

    # get these from package "ztf_wd", module "collect_summaries.py" with function "ztf_rms"
    ztf_mag = np.arange(12.5, 21.5, 0.5)
    ztf_rms = np.array([0.01399772, 0.01397103, 0.01356743, 0.01314759, 0.01332573,
               0.01342575, 0.01358046, 0.01439889, 0.01580364, 0.01857277,
               0.02347019, 0.03147624, 0.0440884,  0.06288794, 0.09019933,
               0.12347744, 0.15113318, 0.18553443]) * np.log(10) / 2.5

    defaults = {
        'ZTF': {
            'name': 'ZTF',
            'telescope': 'P48',
            'field_area': 47,
            # 'num_visits': 500,
            'exposure_time': 30,
            'dead_time': 10,
            'slew_time': 0,
            'filter': 'r',
            'limmag': 20.5,
            'prec_list': ztf_rms,
            'mag_list': ztf_mag,
            # 'footprint': 0.5,
            'cadence': 1.5,
            'duty_cycle': 0.2,
            'location': 'north',
            'longitude': None,
            'latitude': None,
            'elevation': None,
        },
        'LSST': {
            'name': 'LSST',
            'telescope': 'Vera Rubin 8.4m',
            'field_area': 10,
            'num_visits': 1000,
            'exposure_time': 15,
            'series_length': 2,
            'filter': 'r',
            'limmag': 24.5,
            'precision': 0.01,

            'footprint': 0.5,
            'cadence': 10,
            'duty_cycle': 0.2,
            'location': 'south',
            'longitude': None,
            'latitude': None,
            'elevation': None,
        },
        'CURIOS': {
            'name': 'CuRIOS',
            'field_area': 5 ** 2 * np.pi,  # 10 deg f.o.v diameter
            # 'num_visits': 1,
            'exposure_time': 10,
            'filter': 'i',
            'limmag': 18.0,
            'precision': 0.01,
            'series_length': None,  # find this value automatically
            'slew_time': 3600,
            'footprint': 1.0 / 525,
            'cadence': 1,
            'duration': 1,
            'duty_cycle': 1.0,
            'location': 'space',
            'longitude': None,
            'latitude': None,
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
        lens_mass=1.5,
        star_mass=0.6,
        star_temp=7000,
        declination=0.001,
        semimajor_axis=0.0005,
    )

    # sim.syst.plot()

    ztf = Survey('ztf')
    ztf.print()
    print()
    ztf.apply_detection_statistics(sim.syst)

    # name = "ZTF"
    # if len(sim.syst.flare_prob[name]):
    #     print(f'flare prob= {np.max(sim.syst.flare_prob[name]):.2e} | '
    #           f'visit prob= {np.max(sim.syst.visit_prob[name]):.2e} | '
    #           f'visit det= {np.max(sim.syst.visit_detections[name]):.2e} | '
    #           f'total det= {np.max(sim.syst.total_detections[name]):.2e} | '
    #           f'max dist= {max(sim.syst.distances[name]):.2f} pc | '
    #           f'volume: {np.sum(sim.syst.total_volumes[name]):.2e} pc^3 | '
    #           f'effective vol.= {sim.syst.effective_volumes[name]:.2e} pc^3')
    # else:
    #     print('Object is too faint to observe.')

    cu = Survey('curios')

    cu.print()
    print()

    cu.apply_detection_statistics(sim.syst)
    sim.syst.print(surveys=['CURIOS'])
    # name = 'CURIOS'
    # if len(sim.syst.flare_prob[name]):
    #     print(f'flare prob= {np.max(sim.syst.flare_prob[name]):.2e} | '
    #           f'visit prob= {np.max(sim.syst.visit_prob[name]):.2e} | '
    #           f'visit det= {np.max(sim.syst.visit_detections[name]):.2e} | '
    #           f'total det= {np.max(sim.syst.total_detections[name]):.2e} | '
    #           f'max dist= {max(sim.syst.distances[name]):.2f} pc | '
    #           f'volume: {np.sum(sim.syst.total_volumes[name]):.2e} pc^3 | '
    #           f'effective vol.= {sim.syst.effective_volumes[name]:.2e} pc^3')
    # else:
    #     print('Object is too faint to observe.')

