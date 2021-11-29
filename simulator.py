import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, TextBox, Slider

from timeit import default_timer as timer
import scipy.integrate as integ

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
        self.gui = None  # fill this using make_gui()
        self.matrices = None  # list of matrices to use to produce the lightcurves
        self.syst = None  # the latest system output

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
        self._time_units = "seconds"

        # intermediate results
        self.position_radii = None  # distances from star center, at given timestamps
        self.position_angles = None  # angles of vector connecting source and lens centers, measured at given timestamps
        self.fwhm = None

        # outputs
        self.magnifications = None
        self.offset_sizes = None
        self.offset_angles = None

        self.load_matrices()

        self.calculate(
            star_mass=0.5,
            star_size=0.5,
            lens_mass=30,
            lens_type='BH',
            inclination=89.8,
            semimajor_axis=0.1
        )


    @property
    def time_units(self):
        return self._time_units

    @time_units.setter
    def time_units(self, new_units):
        if self.timestamps is not None:
            self.timestamps *= translate_time_units(self.time_units) / translate_time_units(new_units)
        self._time_units = new_units

    def load_matrices(self, filenames=None):

        self.matrices = []

        if filenames is None:
            if os.path.isdir('saved'):
                filenames = 'saved/*.npz'
            else:
                filenames = 'matrix.npz'

        filenames = glob.glob(filenames)

        for f in filenames:
            self.matrices.append(transfer_matrix.TransferMatrix.from_file(f))

        self.matrices = sorted(self.matrices, key=lambda mat: mat.max_source)

    def translate_type(self, type):
        t = type.lower().replace('_', ' ').strip()

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
                temp = 10000
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
                size = 0.01125 * np.sqrt(mass_chand ** (-2/3) - mass_chand ** (2/3))
            if type == 'NS':
                size = 0
            if type == 'BH':
                size = 0

        if flux is None:
            const = 2.744452656619891e+17  # boltzmann const * solar radius ** 2 * ergs
            flux = 4 * np.pi * const * size ** 2 * temp ** 4  # in erg/s

        # print(f'mass= {mass} | type= {type} | size= {size} | temp= {temp} | flux= {flux}')

        return temp, size, flux

    def input(self, **kwargs):

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

    def ingest_timestamps(self, timestamps=None, units=None):

        if timestamps is not None:
            self.timestamps = timestamps

        if units is not None:
            self.time_units = units

        if self.timestamps is None:
            return  # have to put in the default timestamps and call this function again

        phases = (self.timestamps * translate_time_units(self.time_units)) / (self.orbital_period * 3600)

        projected_radius = self.semimajor_axis * 215.032 / self.einstein_radius  # AU to solar radii to Einstein radius
        x = projected_radius * np.sin(2 * np.pi * phases)
        y = projected_radius * np.cos(2 * np.pi * phases) * np.cos(self.inclination / 180 * np.pi)

        self.position_radii = np.sqrt(x ** 2 + y ** 2)
        self.position_angles = np.arctan2(y, x)

    def choose_matrix(self, source_size=None):

        if source_size is None:
            source_size = self.source_size
        matrix_list = sorted(self.matrices, key=lambda x: x.max_source, reverse=False)
        this_matrix = None
        for m in matrix_list:
            if m.min_source < source_size < m.max_source:
                this_matrix = m
                break

        # if this_matrix is None:
            # raise ValueError(f'Cannot find any transfer matrix that includes the source radius ({r})')

        return this_matrix

    def crossing_time(self):
        """
        A rough estimate of the crossing time, considering
        the impact parameter and the resulting source+lens chords,
        and the orbital velocity.
        """

        width = 0
        if self.impact_parameter < 1:
            width += 2 * np.sqrt(1 - self.impact_parameter ** 2)

        if self.impact_parameter < self.source_size:
            width += 2 * np.sqrt(self.source_size ** 2 - self.impact_parameter ** 2)

        width = max(1, width)
        width *= self.einstein_radius / 215.032 # convert from Einstein radius to Solar radius to AU

        velocity = 2 * np.pi * self.semimajor_axis / (self.orbital_period * 3600)

        return width / velocity

    def calculate(self, **kwargs):

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
            self.input(**kwargs)
        else:
            self.input()

        if timestamps is not None or time_units is not None:
            self.ingest_timestamps(timestamps, time_units)

        if self.timestamps is None:
            # time_range = 0.01 * self.orbital_period * 3600
            time_range = 8 * self.crossing_time()
            timestamps = np.linspace(-time_range, time_range, 201, endpoint=True)
            self.ingest_timestamps(timestamps / translate_time_units(self.time_units), self.time_units)

        # first check if the requested source size is lower/higher than any matrix
        max_sizes = np.array([mat.max_source for mat in self.matrices])
        if np.all(max_sizes < self.source_size):
            mag = transfer_matrix.large_source_approximation(
                distances=self.position_radii,
                source_radius=self.source_size,
                occulter_radius=self.occulter_size,
            )
            # raise ValueError(f'Requested source size ({self.source_size}) '
            #                  f'is larger than largest values in all matrices ({np.max(max_sizes)})')
        else:
            matrix = self.choose_matrix()

            if matrix is None:  # source too small!
                mag = transfer_matrix.point_source_approximation(self.position_radii)
            else:
                # print(f'occulter_size= {self.occulter_size} | matrix.max_occulter= {matrix.max_occulter}')
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

        self.fwhm = self.get_fwhm()

        self.latest_runtime = timer() - t0

        self.syst = System()

        # copy all relevant properties of this object
        for k in self.syst.__dict__.keys():
            if hasattr(self, k):
                setattr(self.syst, k, getattr(self, k))

        return self.magnifications

    def get_fwhm(self):

        lc = self.magnifications - 1
        ts = self.timestamps

        peak_idx = len(lc) // 2
        peak = np.max(lc)
        half_idx = np.argmax(lc > 0.5 * peak)
        # print(f'peak= {peak} | half_idx= {half_idx}')

        return 2 * (ts[peak_idx] - ts[half_idx])

    def make_gui(self):
        self.gui = self.GUI(self)

    class GraphicButton:
        def __init__(self, gui, axes, uitype, parameter, label=None, dx=None, altdata=None):
            self.gui = gui  # link back to the GUI object that owns this object
            self.par_of_gui = False  # is the parameter associated with this GUI (=True) or with its owner (=False)
            self.button = None  # the ui-object
            self.uitype = uitype  # can be "info", "toggle", "text", "number", "slider", "push", etc
            self.axes = axes  # the axes the ui-object is drawn on
            self.parameter = parameter
            self.label = label if label is not None else parameter.replace('_', ' ') + '='
            self.use_log = False  # plot the sliders using a logarithmic scale

            if uitype in ('toggle', 'text', 'number', 'slider'):
                bbox = axes.get_position()
                dx = dx if dx is not None else 0.5
                bbox.x0 += dx * bbox.width
                axes.set_position(bbox)
            if uitype in ('slider'):
                bbox = axes.get_position()
                bbox.x1 = 0.75
                h = bbox.height
                bbox.y0 += h * 0.4
                bbox.y1 -= h * 0.4
                axes.set_position(bbox)

            value = self.gui.pars.get(parameter)  # default to None if missing key

            if self.uitype == 'toggle':
                self.button = Button(axes, str(value))
                self.axes.text(0, 0.5, self.label, ha='right', va='center', transform=self.axes.transAxes)

                def callback_toggle(event):
                    self.gui.toggle_activation(self.parameter, self.button)

                self.button.on_clicked(callback_toggle)

            elif uitype in ('text', 'number'):
                self.button = TextBox(self.axes, self.label, initial=str(value))
                if uitype == 'text':
                    def callback_input(text):
                        new_value = text
                        self.gui.input_activation(self.parameter, self.button, new_value)
                if uitype == 'number':
                    def callback_input(text):
                        new_value = float(text)
                        self.gui.input_activation(self.parameter, self.button, new_value)

                self.button.on_submit(callback_input)
            elif uitype == 'slider':
                lower = altdata[0]
                upper = altdata[1]
                if len(altdata) > 2:
                    self.use_log = altdata[2]

                if self.use_log:
                    self.button = Slider(ax=self.axes, label=self.label, valmin=np.log10(lower), valmax=np.log10(upper), valinit=np.log10(value))
                else:
                    self.button = Slider(ax=self.axes, label=self.label, valmin=lower, valmax=upper, valinit=value)

                def callback_input(new_value):
                    if self.use_log:
                        new_value = 10 ** new_value
                    self.gui.input_activation(self.parameter, self.button, new_value)

                self.button.on_changed(callback_input)

            elif uitype == 'push':
                b = Button(axes, label)
                # TODO: need to finish this

            else:
                raise KeyError(f'Unknown uitype ("{uitype}"). Try "push" or "toggle" or "input"')

        def update(self):
            value = self.gui.pars[self.parameter]
            if self.uitype == 'toggle':
                self.button.label.set_text(str(value))
            elif self.uitype == 'text':
                self.button.text_disp.set_text(value)
            elif self.uitype == 'number':
                self.button.text_disp.set_text(str(round(value, 4)))
            elif self.uitype == 'slider':
                if self.use_log:
                    value = np.log10(value)
                    self.button.set_val(value)
                    self.button.valtext.set_text(str(round(10 ** value, 2)))
                else:
                    self.button.set_val(value)


    class GUI:
        def __init__(self, owner):
            self.owner = owner
            self.pars = {}  # parameters to be passed to owner
            self.buttons = []  # a list of GraphicButton objects for all input/info/action buttons
            # self.buttons = {}  # dictionary of name: GUI object (e.g., buttons, sliders, text boxes)
            # self.button_axes = {}  # dictionary of name: object axes (the axes containing buttons, etc)
            # self.axes_button = {}  # reverse dictionary of above: keys are axes and values are parameter names
            self.fig = None  # figure object
            self.gs = None  # gridspec object
            self.subfigs = None  # separate panels help build this up in a modular way
            self.plot_fig = None  # a specific subfigure for display of plots
            self.left_side_fig = None  # a shortcut to the left-side subfigure
            self.auto_update = True  # if true, will also update the owner on every change
            self.update_pars()
            # these are specific for this type of GUI

            # start building the subfigures and buttons
            self.fig = plt.figure(num='Self lensing GUI', figsize=(15, 10), clear=True)
            self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
            self.gs = self.fig.add_gridspec(ncols=3, nrows=3, width_ratios=[2, 8, 2], height_ratios=[3, 10, 1])
            self.subfigs = []

            # the plotting area
            self.subfigs.append(self.fig.add_subfigure(self.gs[1, 1], facecolor='0.75'))
            self.plot_fig = self.subfigs[-1]

            # add buttons
            # left side:
            self.subfigs.append(self.fig.add_subfigure(self.gs[:, 0], facecolor='0.75'))
            self.left_side_fig = self.subfigs[-1]

            axes_left = self.subfigs[-1].subplots(12, 1)
            ind = 0
            self.add_button(axes_left[ind], 'toggle', 'auto_update'); ind += 1
            self.add_button(axes_left[ind], 'number', 'inclination'); ind += 1
            self.add_button(axes_left[ind], 'slider', 'inclination', '', 0, (89.0, 90.0)); ind += 1
            self.add_button(axes_left[ind], 'number', 'semimajor_axis'); ind += 1
            self.add_button(axes_left[ind], 'slider', 'semimajor_axis', '', 0, (0.01, 10, True)); ind += 1

            # make sure all buttons and display are up to date
            self.update_pars()
            self.update_buttons()
            self.update_display()

        def add_button(self, axes, uitype, parameter, label=None, dx=None, altdata=None):
            self.buttons.append(Simulator.GraphicButton(self, axes, uitype, parameter, label, dx, altdata))

        def toggle_activation(self, parameter, button):
            self.pars[parameter] = not self.pars[parameter]
            button.label.set_text(str(self.pars[parameter]))
            self.update()

        def input_activation(self, parameter, button, value):
            if self.pars[parameter] != value:
                self.pars[parameter] = value
                self.update()

        def update(self):
            self.left_side_fig.suptitle('Working...', fontsize='x-large')
            self.left_side_fig.canvas.draw()
            self.auto_update = self.pars['auto_update']
            if self.auto_update:
                # t0 = timer()
                self.update_owner()
                # print(f'time to update owner= {timer() - t0:.1f}s')
                # t0 = timer()
                self.update_pars()
                # print(f'time to update pars= {timer() - t0:.1f}s')
                # t0 = timer()
                self.update_display()
                # print(f'time to update display= {timer() - t0:.1f}s')
            # t0 = timer()
            self.update_buttons()
            # print(f'time to update buttons= {timer() - t0:.1f}s')

            self.left_side_fig.suptitle('', fontsize='x-large')

        def update_owner(self):  # apply the values in pars to the owner and run code to reflect that
            for k, v in self.pars.items():
                if hasattr(self.owner, k):
                    setattr(self.owner, k, v)
            self.owner.calculate()

        def update_pars(self):  # get updated pars from the owner, after its code was run
            self.pars = vars(self.owner)

            # add GUI parameters here on top of owner parameters
            self.pars['auto_update'] = self.auto_update

        def update_display(self):  # update the display from the owner's plotting tools
            axes = self.plot_fig.axes
            [self.plot_fig.delaxes(ax) for ax in axes]
            self.owner.syst.plot(fig=self.plot_fig)

        def update_buttons(self):  # make sure the buttons all show what is saved in pars
            for b in self.buttons:
                b.update()


class System:
    def __init__(self):
        self.semimajor_axis = None  # in AU
        self.orbital_period = None  # in units of hours
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
        self.einstein_radius = None  # in Solar radius units

        self.timestamps = None  # for the lightcurve (units are given below, defaults to seconds)
        self._time_units = 'seconds'  # for the above-mentioned timestamps
        self.magnifications = None  # the measured lightcurve

        # a few dictionaries to be filled by each survey,
        # each one keyed to the survey's name, and containing
        # the results: detection probability for different distances
        # and also the volume associated with each distance.
        self.distances = {}  # parsec
        self.volumes = {}  # parsec^3
        self.apparent_mags = {}  # in the specific band
        self.visit_prob = {}  # probability (0 to 1) of detection in one visit
        self.total_prob = {}  # probability after multiple visit in the duration of the survey
        self.total_volumes = {}  # volumes observed over the duration of the survey
        self.footprints = {}  # the angular area (in square degrees) over the duration of the survey
        self.flare_durations = {}  # duration above detection limit (sec)
        self.dilutions = {}  # how much is the lens light diluting the flare in each survey
        self.effective_volumes = {}  # how much space is covered, including the detection probability, for each survey

        # for each parameter in the first block,
        # what are the intervals around the given values,
        # that we can use to estimate how many such systems exist
        # (to fill out the density parameter).
        # E.g., to say there are 10 systems / parsec^3 with lens_mass=0.4
        # you must decide the lens mass range, maybe 0.35 to 0.45,
        # only then can you say how many "such systems" should exist
        self.par_ranges = {}  # for mass/size/temp of lens/star, inclination, semimajor axis
        self.density = None  # how many such system we expect exist per parsec^3

    @property
    def time_units(self):
        return self._time_units

    @time_units.setter
    def time_units(self, new_units):
        self.timestamps *= translate_time_units(self.time_units) / translate_time_units(new_units)
        self._time_units = new_units

    def bolometric_correction(self, wavelength=None, bandwidth=None):
        """
        Get a correction term to translate the bolometric magnitude
        to the one measured by the filter.

        :param wavelength:
            Central wavelength of filter (if None, will assume broadband)
        :param bandwidth:
            Spectral width of filter (if None, will assume broadband)
        :return 2-tuple:
            A two-element tuple containing:
            - A bolometric correction that needs to be added to any magnitude
              to account for the limited bandwidth of the filter
            - A dilution factor showing how much is the source flux mixed
              with the lens flux. If the lens is dark (NS or BH) the dilution is 1.
              If the fluxes are equal (inside the band!) the dilution is 0.5.
        """
        if wavelength is None or bandwidth is None:
            return 0

        def bb(la, temp):
            const = 0.014387773538277204  # hc/k_b = 6.62607004e-34 * 299 792 458 / 1.38064852e-23
            la *= 1e-9  # convert wavelength from nm to m
            return 1 / (la ** 5 * (np.exp(const/(la * temp)) - 1))

        la1 = wavelength - bandwidth/2
        la2 = wavelength + bandwidth/2

        min_la = 100
        max_la = 1e5

        bolometric1 = integ.quad(bb, min_la, max_la, self.star_temp)[0]
        in_band1 = integ.quad(bb, la1, la2, self.star_temp)[0]
        fraction1 = in_band1 / bolometric1
        # print(f'bolometric1= {bolometric1} | in_band1= {in_band1}')

        if self.lens_flux > 0 and self.lens_temp > 0:
            bolometric2 = integ.quad(bb, min_la, max_la, self.lens_temp)[0]
            in_band2 = integ.quad(bb, la1, la2, self.lens_temp)[0]
            fraction2 = in_band2 / bolometric2
        else:
            fraction2 = 0

        flux_band = self.star_flux * fraction1 + self.lens_flux * fraction2
        flux_bolo = self.star_flux + self.lens_flux

        ratio = flux_band / flux_bolo
        dilution = (self.star_flux * fraction1) / (self.star_flux * fraction1 + self.lens_flux * fraction2)

        return -2.5 * np.log10(ratio), dilution

    def bolometric_mag(self, distance_pc):
        """
        Get the apparent magnitude of the system, not including the magnification,
        given the distance (in pc). The magnitude is bolometric (i.e., it needs
        a correction factor if using anything but a broadband filter).

        :param distance_pc:
            Distance to system, in parsec
        :return:
            Magnitude measured on Earth for this system at that distance
        """

        flux = self.star_flux + self.lens_flux

        # ref: https://www.iau.org/static/resolutions/IAU2015_English.pdf
        abs_mag = -2.5 * np.log10(flux) + 71.197425 + 17.5  # the 17.5 is to convert W->erg/s

        return abs_mag + 5 * np.log10(distance_pc / 10)

    def plot(self, detection_limit=0.01, distance_pc=10, fig=None, font_size=16):

        if fig is None:
            fig, ax = plt.subplots(1, 2)
        else:
            ax = fig.subplots(2, 1)

        ax[0].plot(self.timestamps, self.magnifications, '-ob',
                   label=f'magnification (max= {np.max(self.magnifications):.4f})')
        ax[0].set(
            xlabel=f'time [{self.time_units}]',
            ylabel='magnification',
        )

        # show the detection limit
        ax[0].plot(self.timestamps, np.ones(self.magnifications.shape)*(1 + detection_limit),
                   '--g', label=f'detection limit ({int(detection_limit * 100) : d}%)')

        # show the region above the limit
        idx = self.magnifications - 1 > detection_limit
        if np.any(idx):
            width = max(self.timestamps[idx]) - min(self.timestamps[idx])
            ax[0].plot(self.timestamps[idx], self.magnifications[idx], '-or',
                     label=f'event time ({width:.1f} {self.time_units})')

        legend_handles, legend_labels = ax[0].get_legend_handles_labels()

        # add the period to compare to the flare time
        legend_handles.append(Line2D([0], [0], lw=0))
        period = self.orbital_period * 3600 / translate_time_units(self.time_units)
        legend_labels.append(f'Period= {period:.1f} {self.time_units}')

        # add the magnitudes at distance_pc
        R = default_filter('R')
        V = default_filter('V')
        B = default_filter('B')

        base_mag = self.bolometric_mag(distance_pc)
        mag_R = base_mag + self.bolometric_correction(*R)[0]
        mag_V = base_mag + self.bolometric_correction(*V)[0]
        mag_B = base_mag + self.bolometric_correction(*B)[0]
        legend_handles.append(Line2D([0], [0], lw=0))

        legend_labels.append(f'Mag (at {int(distance_pc):d}pc):\n R~{mag_R:4.2f}\n V~{mag_V:4.2f}\n B~{mag_B:4.2f} ')

        ax[0].legend(legend_handles, legend_labels, bbox_to_anchor=(1.00, 0.95), loc="upper right")

        # add a cartoon of the system in a subplot
        ax[0].set(position=[0.1, 0.1, 0.85, 0.85])
        ax[1].set(position=[0.15, 0.45, 0.3, 0.25])
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)

        # set the scale of the cartoon (for the object sizes at least)
        scale = max(self.star_size, self.einstein_radius)

        # add the star / source
        solar_radii = '$R_\u2609$'
        star_label = f'source size= {self.star_size:4.3f}{solar_radii}'
        star = plt.Circle((-2, 0), self.star_size / scale, color=get_star_plot_color(self.star_temp), label=star_label)
        ax[1].set_xlim((-3.2, 3.2))
        ax[1].set_ylim((-1.5, 1.5))
        ax[1].set_aspect("equal")
        ax[1].add_patch(star)

        # add the lensing object
        lens_label = f'occulter size= {self.lens_size:4.3f}{solar_radii}'
        lt = self.lens_type.strip().upper()

        if lt == 'WD':
            lens = plt.Circle((2, 0), self.lens_size / scale, color=get_star_plot_color(self.lens_temp), label=lens_label)
        elif lt == 'NS':
            lens = plt.Circle((2, 0), 0.1, color=get_star_plot_color(self.lens_temp), label=lens_label)
        elif lt == 'BH':
            lens = plt.Circle((2, 0), 0.05, color='black', label=lens_label)
        else:
            raise KeyError(f'Unknown lens type ("{self.lens_type}"). Try using "WD" or "BH"... ')

        ax[1].add_patch(lens)

        # add the Einstein ring
        ring = plt.Circle((2, 0), self.einstein_radius / scale, color='r', linestyle=':', fill=False,
                          label=f'Einstein radius= {self.einstein_radius:4.3f}{solar_radii}')
        ax[1].add_patch(ring)

        phi = np.linspace(0, 2*np.pi, 1000)
        x = 2*np.cos(phi)
        y = 2*np.sin(phi) * np.sin(np.pi / 180 * (90 - self.inclination) * 50)
        ax[1].plot(x, y, '--k', label=f'a= {self.semimajor_axis}AU, i={self.inclination}$^\circ$')

        legend_handles, legend_labels = ax[1].get_legend_handles_labels()
        # replace the legend marker for the WD star
        legend_handles[1] = Line2D([0], [0], lw=0, marker='o', color=get_star_plot_color(self.lens_temp))
        legend_handles[2] = Line2D([0], [0], marker='o', lw=0, color=get_star_plot_color(self.star_temp))
        legend_handles[3] = Line2D([0], [0], linestyle=':', color='r')

        # order = [2, 0, 3, 1]
        # legend_handles = [legend_handles[i] for i in order]
        # legend_labels = [legend_labels[i] for i in order]

        ax[1].legend(legend_handles, legend_labels,
                   bbox_to_anchor=(0.5, 1.04), loc="lower center")

        return ax


def translate_time_units(units):
    d = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 3600 * 24, 'years': 3600 * 24 * 365.25}
    if units not in d:
        raise KeyError(f'Unknown units specified ("{units}"). Try "seconds" or "hours". ')
    return d[units.lower()]


def get_star_plot_color(temp):

    lim1 = 2000
    lim2 = 10000
    push_low = 0.05

    cmp = plt.get_cmap('turbo')

    if temp < lim1:
        return cmp(1)

    if temp > lim2:
        return cmp(push_low)

    return cmp((lim2 - temp) / (lim2 - lim1) + push_low)


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

    s = Simulator()
    s.make_gui()
    # s.calculate(star_mass=0.5, star_size=0.5, lens_mass=30, lens_type='BH ', inclination=89.8, semimajor_axis=0.1)
    # syst = s.output_system()
    # syst.plot()

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


