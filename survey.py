"""
defines a survey class that gets a lightcurve and other metadata from the simulator
and from that figures out the probability of detection for those system parameters.
"""

import numpy as np


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
        self.series_length = kwargs.get('series_length', 1)  # number of images taken continuously in one visit
        # self.strategy = kwargs.get('strategy', None)  # can be "sparse", or "continuous"
        self.cadence = kwargs.get('cadence')  # how much time (days) passes between starts of visits
        self.footprint = kwargs.get('footprint')  # what fraction of the sky is surveyed (typically 0.5 or 1)
        self.duration = 1  # years (setting default 1 is to compare the number of detections per year)

        # filters and image depths
        self.limmag = kwargs.get('limmag')  # objects fainter than this would not be seen at all
        self.precision = kwargs.get('precision')  # photometric precision per image
        self.distances = kwargs.get('distances', np.geomspace(10, 50000, 50, endpoint=True))
        self.filter = kwargs.get('filters', 'V')
        self.wavelength = None  # central wavelength (in nm)
        self.bandpass = None  # assume top-hat (in nm)

        (self.wavelength, self.bandpass) = default_filter(self.filter)

        if 'wavelength' in kwargs:
            self.wavelength = kwargs['wavelength']
        if 'bandwidth' in kwargs:
            self.bandpass = kwargs['bandwidth']

        list_needed_values = ['location', 'field_area', 'exposure_time', 'cadence', 'footprint', 'limmag', 'precision']
        for k in list_needed_values:
            if getattr(self, k) is None:
                raise ValueError(f'Must have a valid value for "{k}".')

    def det_prob(self, system):
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

        # determine how much time the lensing event is above the survey precision
        

        # choose the detection method depending on the flare time and exposure time
        # what is the det. prob. for a series mid time randomly shifted relative to the event mid time?

        # dilute the det. prop. by the duty cycle (all the time the exposure may have been outside the event)

        # multiply by the number of times the survey returns to this spot during a period



        pass


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
            'field_area': 37,
            'footprint': 0.5,
            'cadence': 3,
            'exposure_time': 30,
            'filter': 'r',
            'limmag': 20.5,
            'precision': 0.01,
        }
    }

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
    """
    filters = {
        'V': (550, 100),
        'r': (700, 100),
        'i': (800, 100),
        'F500W': (500, 200),
    }
    # TODO: make sure these numbers are correct!

    if filter_name not in filters:
        # raise KeyError(f'Unknonwn filter name "{filter_name}". Use "V" or "r" etc. ')
        return None, None

    return filters[filter_name]
