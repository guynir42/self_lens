"""
defines a survey class that gets a lightcurve and other metadata from the simulator
and from that figures out the probability of detection for those system parameters.
"""

import numpy as np
import scipy.special
from timeit import default_timer as timer
import matplotlib.pyplot as plt

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
        self.duration = kwargs.get('duration', 1)  # years (setting default 1 to get detections per year)

        # filters and image depths
        self.limmag = kwargs.get('limmag')  # objects fainter than this would not be seen at all  (faintest magnitude)
        self.precision = kwargs.get('precision')  # photometric precision per image (best possible value)
        self.threshold = kwargs.get('threshold', 5)  # need 5 sigma for a detection by default
        self.mag_list = kwargs.get('mag_list')  # list of magnitudes over which the survey is sensitive
        self.prec_list = kwargs.get('prec_list')  # list of photometric precision for each mag_list item
        self.distances = kwargs.get('distances', np.geomspace(MIN_DIST_PC, 50000, 200, endpoint=True)[1:])
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

        # in case we know the cadence and want to figure out the series length (e.g., for CURIOS)
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

        # # we know what part of the sky we want to cover but want the cadence to be set automatically
        # elif self.cadence is None and self.num_fields is None and self.num_visits is None:
        #     # validate that we have all the needed information
        #     for arg in ['footprint', 'duration', 'duty_cycle', 'series_length',
        #                 'exposure_time', 'slew_time', 'dead_time', 'field_area']:
        #         if getattr(self, arg) is None:
        #             raise ValueError(f'Cannot find number of fields/visits without {arg}')
        #
        #         visit_time = self.series_length * (self.exposure_time + self.dead_time) + self.slew_time
        #         self.num_fields = int(np.ceil(self.footprint * (4 * 180 ** 2 / np.pi) / self.field_area))
        #         self.num_visits = int(np.round(self.duration * 365.25 / self.cadence))

        else:
            raise ValueError('Illegal combination of values for this survey!')

        self.footprint = self.num_fields * self.field_area / (4 * 180 ** 2 / np.pi)

    def apply_detection_statistics(self, system, plotting=False):
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
            # p *= self.threshold
            if np.all(det_lc > p):
                # this should not happen in general, if there are enough points outside the flare
                # raise ValueError('All lightcurve points are above precision level!')
                # print('All lightcurve points are above precision level!')
                # end_idx = len(det_lc) - 1
                flare_time_per_distance.append(ts[-1]-ts[0])
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
                # if plotting:
                #     print(f't1= {t1}, t2= {t2}, l1= {l1}, l2= {l2}, t_flare= {t_flare}')

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

        best_precision_idx = np.argmin(prec) if len(prec) else []  # index of best precision
        best_precision = prec[best_precision_idx]
        best_flare_time = flare_time_per_distance[best_precision_idx]

        period = system.orbital_period * 3600  # convert to seconds
        if len(prec):  # only calculate probabilities for distances that have any chance of detection

            (peak_prob, mean_prob, num_detections) = self.calc_prob(
                lc,
                ts,
                prec,
                best_precision,
                best_flare_time,
                period,
                plotting
            )

            if self.exposure_time > period:  # this only applies to simple S/N case of diluted signal
                pass  # TODO: figure out this case later

            system.flare_prob[self.name] = peak_prob
            system.visit_prob[self.name] = mean_prob
            system.visit_detections[self.name] = num_detections

            # the probability to find at least one flare per system (cannot exceed 1)
            system.total_prob[self.name] = 1 - (1 - mean_prob) ** self.num_visits

            # average number of detections over many visits (can exceed 1)
            system.total_detections[self.name] = num_detections * self.num_visits

            # if total_detections is large, can accumulate more effective volume than volume
            # total_detections assumes the volume contains one system
            # to get the actual number of detections you need to multiply effective volume with spatial density
            system.effective_volumes[self.name] = np.sum(system.total_volumes[self.name] * system.total_detections[self.name])

        # consider what happens if the survey does exist in the list, should it be updated??
        if self.name not in (s.name for s in system.surveys):
            system.surveys.append(self)

    def calc_prob(self, lc, ts, precision, best_precision, best_t_flare, period, plotting=False):
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
        # the snr represents the probability to detect a single flare (ignoring periodicity),
        # assuming the flare occured inside the time span of the series

        multiplicative_factor = 1  # the S/N values are the only valid probability points
        if best_t_flare * 10 < t_exp:  # flare is much shorter than exposure time
            signal = np.sum(lc[1:] * np.diff(ts))  # total signal in flare (skip 1st bin of LC to multiply with diff)
            noise = t_exp * best_precision  # total noise in exposure
            snr = signal / noise
            snr = np.array([snr])  # put this into a 1D array

            # coverage = t_exp  # the only times where prob>0 is inside the exposure
            dt = t_series * t_exp / (t_exp + t_dead)  # some parts of t_series are dead times
        elif t_exp * 10 < best_t_flare < t_series / 10:  # exposures are short and numerous: can use matched-filter
            # TODO: add dead time
            snr = np.sqrt(np.sum(lc ** 2)) / best_precision  # sum over different exposures hitting places on the LC
            snr = np.array([snr])  # put this into a 1D array
            dt = t_series  # anywhere along the series has the same S/N and prob (ignoring edge effects)

        else:  # all other cases require sliding window
            dt = ts[1] - ts[0]  # assume timestamps are uniformly sampled in the simulated LC
            N_exp = max(1, int(np.round(t_exp / dt)))  # number of time steps in single exposure
            single_exposure_snr = np.convolve(lc, np.ones(N_exp), mode='same') / N_exp / best_precision
            # coverage = dt * len(single_exposure_snr)  # all points with non-zero S/N have prob>0

            if self.series_length == 1:
                snr = single_exposure_snr
            else:
                # Need to calculate the matched-filter result for several exposures.
                # go over the single-image response and convolve that
                # with the addition of multiple values from separate exposures
                N_btw = int(np.round((t_exp + t_dead) / dt))  # number of steps between repeat exposures
                N_btw = max(N_btw, 1)
                N_flare = int(np.round(best_t_flare / dt))
                N_flare = max(N_flare, 1)
                N_filter = min(N_flare * 5, int(N_btw * self.series_length))  # number of steps in filter
                N_filter = max(N_filter, 1)
                # if we only sample probabilities from a subset of the series
                # then we must increase the total prob*dt later
                multiplicative_factor = int(N_btw * self.series_length) / N_filter

                # the filter samples the mid exposure with the correct dead time
                matched_filter = np.zeros(N_filter)
                matched_filter[N_exp // 2::N_btw] = 1
                snr = np.sqrt(np.convolve(single_exposure_snr ** 2, matched_filter, mode='same'))

                # coverage = t_series / (dt * N_filter)

            # coverage = min(coverage, period)  # coverage cannot exceed period (when series is longer, see below)

        if plotting:
            print(f'Flare duration= {best_t_flare:.2f}s | dt= {dt:.2f}s')
            plt.clf()
            plt.plot(ts, lc, '-x', label='Raw magnification')
            if snr.shape == (1,):
                plt.plot(ts, np.ones(ts.shape) * snr[0], '--', label='S/N value')
            else:
                plt.plot(ts, snr, '--o', label='S/N curve')
            plt.xlabel('Time [seconds]')
            plt.legend()
            plt.show()

        # snr must be a 1D array of S/N for each time offset (possibly with one element)
        # now add a dimension for each precision value
        snr = np.expand_dims(snr, axis=0)  # make this a 2D array, axes 0: distances, axis 1: timestamps
        snr = snr * best_precision / np.expand_dims(precision, axis=1)   # broadcast to various values of precision

        # snr is 2D, with axis 0 for precision values, axis 1 for time offsets (each may be 1 element)
        # now calculate the probabilities
        prob = 0.5 * (1 + scipy.special.erf(snr - self.threshold))  # assume Gaussian distribution
        peak_prob = np.max(prob, axis=1)  # peak brightness of the flare assuming perfect timing

        # find the average probability to make a detection, and the average number of detections
        # now we must add the probability of a flare to be inside the span of the series
        # and account for multiple flares inside the series

        # the probability for each time step dt
        # (or entire series, if uniform probability)
        # is summed to give the total probability
        # divide by the period to get the real prob
        # including the condition that flare is inside
        # each of the dt ranges
        mean_prob = np.sum(prob, axis=1) * dt / period * multiplicative_factor
        mean_prob = np.minimum(mean_prob, 1.0)  # cannot exceed 1, although this should never happen
        # there's no way to see more than one flare in this series,
        # so the average number of detections is just the probability to see one
        num_detections = mean_prob

        if t_series > period:  # entire orbit is covered by the series
            # if the flare occurs multiple times in a single visit/series, we must account for the fact
            # that the number of flares in the series is between N and N+1 (depending on fraction of period)
            num_flares_in_series = t_series // period
            fraction = t_series % period / period  # fraction of period after num_flares were seen

            num_detections = mean_prob * (num_flares_in_series + fraction)  # average of N and N+1 with weight

            # a simplifying assumption that for multiple flares in a series,
            # the peak_prob is the same for each flare.
            prob1 = 1 - (1 - mean_prob) ** num_flares_in_series  # at least one detection
            prob2 = 1 - (1 - mean_prob) ** (num_flares_in_series + 1)  # at least one detection
            mean_prob = prob1 * (1 - fraction) + prob2 * fraction  # weighted average of the two options

            # to count the average number of detections assume each detection has the same (peak) probability

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

    # this rough estimate using package "ztf_wd", module "make_plots.py" with function "choose_summaries"
    # we assume these precision values go together with a 7.5 sigma cut
    # and even then we get tens of thousands of candidates at faint magnitudes
    ztf_rms = np.array([0.11, 0.11, 0.11, 0.11, 0.11, 0.15, 0.15, 0.15, 0.17,
                        0.17, 0.23, 0.3, 0.47, 0.75, 1.1, 1.7, 2.5, 3.7]) / 7.5

    # this comes from the TESS references, but it seems really too optimistic
    tess_mag_rough = np.array(list(range(6, 11)) + [15])
    tess_rms_rough = np.array([26, 30, 35, 50, 100, 450]) * 1e-6 * 3.6
    tess_mag = np.arange(6, 15, 0.5)
    tess_rms = np.interp(tess_mag, tess_mag_rough, tess_rms_rough)

    # from Hanna's email:
    # The table below contains values of the expected SNR for different source magnitudes and integration
    # times for the CuRIOS monolithic, 15 cm optic and Sony IMS455 design. All values were taken for r-band
    # observations.
    curios_exp_times = [1, 15, 60, 300, 900]
    curios_mag_snr = [
        (14.0, 9.676, 37.789, 75.613, 169.1, 292.9),
        (14.5, 7.49, 29.4, 58.8, 131.55, 227.85),
        (15.0, 5.73, 22.6, 22.61, 45.28, 101.265, 175.4),
        (15.5, 4.3104, 17.146, 34.342, 76.822, 133.069),
        (16.0, 3.173, 12.750, 25.553, 57.170, 99.031),
        (16.5, 2.279, 9.261, 18.574, 41.563, 71.998),
        (17.0, 2.279, 9.261, 18.574, 41.563, 71.998),
        (17.5, 1.594, 6.555, 13.155, 29.444, 51.006),
        (18.0, 0.726, 3.045, 6.120, 13.702, 23.738),
        (18.5, 0.476, 2.012, 4.046, 9.060, 15.696),
        (19.0, 0.308, 1.310, 2.635, 5.902, 10.225),
        (19.5, 0.198, 0.844, 1.698, 3.804, 6.591),
        (20.0, 0.126, 0.540, 1.086, 2.434, 4.216),
        (20.5, 0.080, 0.343, 0.691, 1.549, 2.684),
        (21.0, 0.050, 0.218, 0.439, 0.983, 1.703)
    ]
    curios_chosen_exp_time = 15
    idx = curios_exp_times.index(curios_chosen_exp_time) + 1  # plus one to account for the magnitude first element
    curios_precision = [1 / m[idx] for m in curios_mag_snr if m[idx] > 3.0]
    curios_mags = [m[0] for m in curios_mag_snr if m[idx] > 3.0]

    last_mag_rms = [
        (10.0, 0.005),
        (10.5, 0.005),
        (11.0, 0.006),
        (11.5, 0.007),
        (12.0, 0.008),
        (12.5, 0.009),
        (13.0, 0.010),
        (13.5, 0.014),
        (14.0, 0.020),
        (14.5, 0.028),
        (15.0, 0.040),
        (15.5, 0.055),
        (16.0, 0.077),
        (16.5, 0.110),
        (17.0, 0.152),
        (17.5, 0.214),
        (18.0, 0.300)
    ]

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
            'threshold': 5,
            # 'footprint': 0.5,
            'cadence': 1.5,
            'duty_cycle': 0.25,
            'location': 'north',
            'longitude': None,
            'latitude': None,
            'elevation': None,
            'duration': 3,
            'distances': np.geomspace(MIN_DIST_PC, 5000, 100, endpoint=True)[1:]
        },
        'LSST': {
            'name': 'LSST',
            'telescope': 'Vera Rubin 8.4m',
            'field_area': 10,
            # 'num_visits': 1000,
            'exposure_time': 15,
            'series_length': 2,
            'dead_time': 2.5,
            'slew_time': 5,
            'filter': 'g',
            'limmag': 24.5,
            'precision': 0.01,
            'cadence': 3,
            'duty_cycle': 0.25,
            'location': 'south',
            'longitude': None,
            'latitude': None,
            'elevation': None,
            'duration': 5,
            'distances': np.geomspace(MIN_DIST_PC, 50000, 200, endpoint=True)[1:]
        },
        'TESS': {
            'name': 'TESS',
            'telescope': 'TESS',
            'field_area': 2300,
            'exposure_time': 30 * 60,
            'cadence': 2 * 365.25,  # visit per two year
            'series_length': 27 * 24 * 60 / 30,  # 27 days, in 30min exposures
            'dead_time': 0,
            'filter': 'TESS',
            'duty_cycle': 1.0,
            'location': 'space',
            'precision': 0.01,
            'limmag': 15,
            'prec_list': tess_rms,
            'mag_list': tess_mag,
            'footprint': 1,  # entire sky
            'duration': 5,
            'distances': np.geomspace(MIN_DIST_PC, 300, 100, endpoint=True)[1:]
        },
        'CURIOS': {
            'name': 'CuRIOS',
            # 'field_area': 5 ** 2 * np.pi,  # 10 deg f.o.v diameter
            'field_area': 49,  # 7 degree square
            'exposure_time': curios_chosen_exp_time,
            'filter': 'r',
            # 'series_length': None,  # find this value automatically
            'series_length': 60,  # 15 minute visits, with 15 second exposures
            'mag_list': curios_mags,
            'prec_list': curios_precision,
            'slew_time': 5,
            # 'footprint': 1.0 / 525,
            'cadence': 1.5 / 24,  # orbit of 1.5 hours, in units of visits per day
            'duty_cycle': 0.95,
            'location': 'space',
            'longitude': None,
            'latitude': None,
            'duration': 5,
            'distances': np.geomspace(MIN_DIST_PC, 1000, 100, endpoint=True)[1:]
        },
        'CURIOS ARRAY': {
            'name': 'CuRIOS array',
            'field_area': 4 * 180 ** 2 / np.pi,  # entire sky
            'exposure_time': curios_chosen_exp_time,
            'filter': 'r',
            'mag_list': curios_mags,
            'prec_list': curios_precision,
            # 'series_length': None,  # find this value automatically
            'series_length': 60,  # 15 minute visits, with 15 second exposures
            # 'footprint': 1.0,
            'cadence': 15 / 24 / 60,  # start a new series every 15 minutes
            'slew_time': 5,
            'duty_cycle': 0.95,
            'location': 'space',
            'longitude': None,
            'latitude': None,
            'duration': 5,
            'distances': np.geomspace(MIN_DIST_PC, 1000, 100, endpoint=True)[1:]
        },
        'LAST': {
            'name': 'LAST',
            'telescope': '11" RASA',
            'field_area': 355,
            'exposure_time': 20,
            'series_length': 20,
            'cadence': 1.0,
            'dead_time': 0,
            'slew_time': 0,
            'filter': 'white',
            'limmag': 18.0,
            'prec_list': [p[1] for p in last_mag_rms],
            'mag_list': [p[0] for p in last_mag_rms],
            'duty_cycle': 0.25,
            'location': 'north',
            'longitude': None,
            'latitude': None,
            'elevation': None,
            'duration': 5,
            'distances': np.geomspace(MIN_DIST_PC, 1000, 100, endpoint=True)[1:]
        }
    }

    name = name.upper().replace('_', ' ')

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
        lens_mass=0.6,
        star_mass=0.6,
        star_temp=10000,
        lens_temp=10000,
        declination=0.000,
        semimajor_axis=0.18,
    )

    # sim.syst.plot()

    tess = Survey('TESS')
    tess.print()
    tess.apply_detection_statistics(sim.syst)
    # sim.syst.print(surveys=['TESS'])
    # print(f'effective volume (P={sim.syst.period_string()})= {sim.syst.effective_volumes["TESS"]:.2e}')

    # sim.calculate(semimajor_axis=0.2)
    # tess.apply_detection_statistics(sim.syst)
    # print(f'effective volume (P={sim.syst.period_string()})= {sim.syst.effective_volumes["TESS"]:.2e}')

    print()
    ztf = Survey('ztf')
    ztf.print()
    ztf.apply_detection_statistics(sim.syst)
    # sim.syst.print(surveys=['ZTF'])

    print()
    lsst = Survey('LSST')
    lsst.print()
    lsst.apply_detection_statistics(sim.syst)

    print()
    cu = Survey('curios')
    cu.print()
    cu.apply_detection_statistics(sim.syst)
    # sim.syst.print(surveys=['CURIOS'])

    print()
    array = Survey('curios array')
    array.print()
    cu.apply_detection_statistics(sim.syst)

    # print()
    # last = Survey('last')
    # last.print()
    # last.apply_detection_statistics(sim.syst)

    # print()
    # print('System overview:')
    # sim.syst.print()
