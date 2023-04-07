import os
import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, TextBox, Slider, RadioButtons

from timeit import default_timer as timer
import scipy.integrate as integ

from src.transfer_matrix import (
    TransferMatrix,
    large_source_approximation,
    point_source_approximation,
    distance_for_precision,
)

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

        # lensing parameters (in units of Einstein radius)
        self.einstein_radius = None  # in Solar units, to translate the input
        self.impact_parameter = None
        self.orbital_period = None  # in units of hours
        self.roche_lobe = None  # in units of solar radii
        self.source_size = None
        self.occulter_size = None

        # physical parameters (in Solar units)
        self.semimajor_axis = 1  # in AU
        # self.inclination = 89.95  # orbital inclination (degrees)
        self.declination = 0.01  # 90-inclination (degrees)
        self.compact_source = True  # is the source a compact object or a main sequence star?
        self.star_mass = 1  # in Solar mass units
        self.star_temp = 5000  # in Kelvin
        # use the following hidden properties to override the calculated type, size and flux
        self._star_type = None  # what kind of object: "star", "WD", "NS", "BH"
        self._star_size = None  # in Solar radius units
        self._star_flux = None  # luminosity in ergs/s
        self.lens_mass = 1  # in Solar mass units
        self.lens_temp = 5000  # in Kelvin
        # use the following hidden properties to override the calculated type, size and flux
        self._lens_type = None  # what kind of object: "star", "WD", "NS", "BH"
        self._lens_size = None  # in Solar radius units
        self._lens_flux = None  # luminosity in ergs/s

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

        self.calculate()

    @property
    def inclination(self):
        return 90 - self.declination

    @inclination.setter
    def inclination(self, new_value):
        self.declination = 90 - new_value

    @property
    def orbital_period(self):
        if self.semimajor_axis is None:
            return None

        period = self.semimajor_axis ** (3 / 2) / np.sqrt(self.star_mass + self.lens_mass)
        period *= 365.25 * 24  # convert from years to hours
        return period

    @orbital_period.setter
    def orbital_period(self, period):
        if period is None:
            self.semimajor_axis = None
            return
        period /= 365.25 * 24  # convert from hours to years
        self.semimajor_axis = (period * np.sqrt(self.star_mass + self.lens_mass)) ** (2 / 3)

    @property
    def star_type(self):
        if self._star_type is not None:
            return self._star_type
        elif not self.compact_source:
            return "MS"
        else:
            return guess_compact_type(self.star_mass)

    @star_type.setter
    def star_type(self, new_type):
        self._star_type = self.translate_type(new_type)

    @property
    def star_size(self):
        if self._star_size is not None:
            return self._star_size
        elif not self.compact_source:
            return main_sequence_size(self.star_mass)
        else:
            return compact_object_size(self.star_mass)

    @star_size.setter
    def star_size(self, new_size):
        self._star_size = new_size

    @property
    def star_flux(self):
        if self._star_flux is not None:
            return self._star_flux
        elif self.star_size == 0:
            return 0
        else:
            const = 2.744452656619891e17  # boltzmann const * solar radius ** 2 * ergs
            return 4 * np.pi * const * self.star_size**2 * self.star_temp**4  # in erg/s

    @star_flux.setter
    def star_flux(self, new_flux):
        self._star_flux = new_flux

    @property
    def lens_type(self):
        if self._lens_type is not None:
            return self._lens_type
        else:
            return guess_compact_type(self.lens_mass)

    @lens_type.setter
    def lens_type(self, new_type):
        self._lens_type = self.translate_type(new_type)

    @property
    def lens_size(self):
        if self._lens_size is not None:
            return self._lens_size
        else:
            return compact_object_size(self.lens_mass)

    @lens_size.setter
    def lens_size(self, new_size):
        self._lens_size = new_size

    @property
    def lens_flux(self):
        if self._lens_flux is not None:
            return self._lens_flux
        elif self.lens_size == 0:
            return 0
        else:
            const = 2.744452656619891e17  # boltzmann const * solar radius ** 2 * ergs
            return 4 * np.pi * const * self.lens_size**2 * self.lens_temp**4  # in erg/s

    @lens_flux.setter
    def lens_flux(self, new_flux):
        self._lens_flux = new_flux

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
            if os.path.isdir("saved"):
                filenames = "saved/*.npz"
            else:
                filenames = "matrix.npz"

        filenames = glob.glob(filenames)

        for f in filenames:
            self.matrices.append(TransferMatrix.from_file(f))

        self.matrices = sorted(self.matrices, key=lambda mat: mat.max_source)

    def translate_type(self, type):

        if type is None:
            return None

        t = type.lower().replace("_", " ").strip()

        if t == "ms" or t == "main sequence" or t == "star":
            t = "MS"
        elif t == "wd" or t == "white dwarf":
            t = "WD"
        elif t == "ns" or t == "neutron star":
            t = "NS"
        elif t == "bh" or t == "black hole":
            t = "BH"
        else:
            raise ValueError(f'Unknown object type "{type}". Use "MS" or "WD" or "NS" or "BH"...')

        return t

    def input(self, **kwargs):

        for key, val in kwargs.items():
            if (
                hasattr(self, key)
                and re.match("(star|lens)_.*", key)
                or key
                in (
                    "declination",
                    "inclination",
                    "semimajor_axis",
                    "time_units",
                    "compact_source",
                )
            ):
                setattr(self, key, val)
            elif key == "orbital_period":
                pass
            else:
                raise ValueError(f'Unknown input argument "{key}"')

        # after setting these, we can also check if orbital_period is given:
        if "orbital_period" in kwargs:
            self.orbital_period = kwargs["orbital_period"]

        if self.star_mass is None:
            raise ValueError("Must provide a mass for the source object.")

        if self.lens_mass is None:
            raise ValueError("Must provide a mass for the lensing object.")

        constants = 220961382691907.84  # = G*M_sun*AU/c^2 = 6.67408e-11 * 1.989e30 * 1.496e11 / 299792458 ** 2
        self.einstein_radius = np.sqrt(4 * constants * self.lens_mass * self.semimajor_axis)
        self.einstein_radius /= 6.957e8  # translate from meters to solar radii
        if self.declination == 0:
            self.impact_parameter = 0
        else:
            self.impact_parameter = self.semimajor_axis * np.sin(self.declination / 180 * np.pi)
            self.impact_parameter *= 215.032  # convert to solar radii
            self.impact_parameter /= self.einstein_radius  # convert to units of Einstein Radius

        # self.orbital_period = self.semimajor_axis ** (3 / 2) / np.sqrt(self.star_mass + self.lens_mass)
        # self.orbital_period *= 365.25 * 24  # convert from years to hours

        # ref: https://ui.adsabs.harvard.edu/abs/1983ApJ...268..368E/abstract
        q = self.star_mass / self.lens_mass
        self.roche_lobe = (0.49 * q ** (2 / 3)) / (0.6 * q ** (2 / 3) + np.log(1 + q ** (1 / 3)))
        self.roche_lobe *= self.semimajor_axis * 215.032  # convert AU to Solar radii

        self.source_size = self.star_size / self.einstein_radius
        self.occulter_size = self.lens_size / self.einstein_radius

        # make sure to ingest the timestamps and units in the right order
        if "time_units" in kwargs:
            self.time_units = kwargs["time_units"]

        # now that units are established, we accept timestamps AS GIVEN (assuming they are already in those units!)
        if "timestamps" in kwargs:
            self.timestamps = kwargs["timestamps"]
        else:
            time_range = 16 * self.crossing_time()  # in seconds
            time_range = min(time_range, self.orbital_period * 3600 / 4)
            self.timestamps = np.linspace(-time_range, time_range, 2001, endpoint=True)
            self.timestamps /= translate_time_units(self.time_units)  # convert to correct units

        phases = (self.timestamps * translate_time_units(self.time_units)) / (self.orbital_period * 3600)

        projected_radius = self.semimajor_axis * 215.032 / self.einstein_radius  # AU to solar radii to Einstein radii
        x = projected_radius * np.sin(2 * np.pi * phases)
        y = projected_radius * np.cos(2 * np.pi * phases) * np.sin(self.declination / 180 * np.pi)
        self.position_radii = np.sqrt(x**2 + y**2)
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
        and the orbital velocity. The output is in seconds.
        """

        width = 0
        if self.impact_parameter < 1:
            width += 2 * np.sqrt(1 - self.impact_parameter**2)

        if self.impact_parameter < self.source_size:
            width += 2 * np.sqrt(self.source_size**2 - self.impact_parameter**2)

        width = max(1, width)
        width *= self.einstein_radius / 215.032  # convert from Einstein radius to Solar radius to AU

        velocity = 2 * np.pi * self.semimajor_axis / (self.orbital_period * 3600)

        return width / velocity

    def calculate(self, **kwargs):

        t0 = timer()
        if "timestamps" in kwargs:
            timestamps = kwargs.pop("timestamps")
        else:
            timestamps = self.timestamps

        if "time_units" in kwargs:
            time_units = kwargs.pop("time_units")
        else:
            time_units = self.time_units

        self.input(**kwargs)

        # first check if the requested source size is lower/higher than any matrix
        max_sizes = np.array([mat.max_source for mat in self.matrices])
        if np.all(max_sizes < self.source_size):
            mag = large_source_approximation(
                distances=self.position_radii,
                source_radius=self.source_size,
                occulter_radius=self.occulter_size,
            )
            # raise ValueError(f'Requested source size ({self.source_size}) '
            #                  f'is larger than largest values in all matrices ({np.max(max_sizes)})')
        else:
            matrix = self.choose_matrix()

            if matrix is None:  # source too small!
                mag = point_source_approximation(self.position_radii)
            else:
                # print(f'occulter_size= {self.occulter_size} | matrix.max_occulter= {matrix.max_occulter}')
                mag = matrix.radial_lightcurve(
                    source=self.source_size,
                    distances=self.position_radii,
                    occulter_radius=self.occulter_size,
                    get_offsets=False,  # at some point I'd want to add the offsets too
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

        center_idx = len(lc) // 2
        peak_idx = np.argmax(lc[center_idx:]) + center_idx  # first peak from center
        peak = lc[peak_idx]
        half_idx = peak_idx
        for i in range(peak_idx, len(lc)):
            if lc[i] < 0.5 * peak:
                half_idx = i
                break

        # half_idx must be bigger than peak_idx
        t1 = ts[half_idx - 1]
        t2 = ts[half_idx]
        l1 = lc[half_idx - 1]
        l2 = lc[half_idx]
        t_flare = t1 + (0.5 * peak - l1) / (l2 - l1) * (t2 - t1)  # linear interpolation
        t_flare *= 2  # symmetric lightcurve

        return t_flare
        # return 2 * (ts[half_idx] - ts[center_idx])

    def length_scale_estimate(self, precision=0.01, threshold=3):
        """
        Find the length scale (in km) away from perfect alignment where
        a lens can be relative to the source and still have magnification
        brighter than precision times threshold.
        """
        einstein_km = self.einstein_radius * 700000  # convert solar radii to km

        scale_km = einstein_km

        # maximum length scale for lensing is lens+source size (already in einstein units)
        multiplier_star = 1 + self.source_size
        multiplier_prec = distance_for_precision(precision * threshold)
        # print(f'star= {multiplier_star:.3f} | prec= {multiplier_prec:.3f}')
        scale_km *= max(multiplier_star, multiplier_prec)

        return scale_km

    def best_prob_all_declinations_estimate(self, precision=0.01, threshold=3):
        """
        The probability of this system to be detectable, assuming best
        possible timing (or continuous coverage).
        Does not consider the effects of lens occultation.
        """

        scale_km = self.length_scale_estimate(precision, threshold)
        quarter_circle_km = self.semimajor_axis * 150e6 / 2 * np.pi  # convert AU to km

        return np.sin(scale_km / quarter_circle_km)  # sin weighting of low declination more than high

    def visit_prob_all_declinations_estimate(self, precision=0.01, threshold=3):
        """
        Get a back-of-the-envelope estimate for the single visit probability,
        when marginalizing over all declinations / inclinations.

        """
        scale_km = self.length_scale_estimate(precision, threshold)

        sma_km = self.semimajor_axis * 150e6  # convert AU to km

        # what fraction of the sphere is covered by the einstein ring
        angle_rad = scale_km / sma_km

        total_prob_estimate = angle_rad**2 / 4

        return total_prob_estimate

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
            self.func = None  # assign the function to call when using a push button
            self.label = label if label is not None else parameter.replace("_", " ") + "="
            self.use_log = False  # plot the sliders using a logarithmic scale

            if uitype in ("toggle", "text", "number", "slider", "radio"):
                bbox = axes.get_position()
                dx = dx if dx is not None else 0.5
                bbox.x0 += dx * bbox.width
                axes.set_position(bbox)
            if uitype in ("slider"):
                bbox = axes.get_position()
                bbox.x1 = 0.75
                h = bbox.height
                bbox.y0 += h * 0.4
                bbox.y1 -= h * 0.4
                axes.set_position(bbox)

            value = self.gui.pars.get(parameter)  # default to None if missing key

            if self.uitype == "toggle":
                self.button = Button(axes, str(value))
                self.axes.text(
                    0,
                    0.5,
                    self.label,
                    ha="right",
                    va="center",
                    transform=self.axes.transAxes,
                )

                def callback_toggle(event):
                    self.gui.toggle_activation(self.parameter, self.button)

                self.button.on_clicked(callback_toggle)
            elif uitype in ("text", "number"):
                self.button = TextBox(self.axes, self.label, initial=str(value))
                if uitype == "text":

                    def callback_input(text):
                        new_value = text
                        self.gui.input_activation(self.parameter, self.button, new_value)

                if uitype == "number":

                    def callback_input(text):
                        new_value = float(text)
                        self.gui.input_activation(self.parameter, self.button, new_value)

                self.button.on_submit(callback_input)
            elif uitype == "slider":
                lower = altdata[0]
                upper = altdata[1]
                if len(altdata) > 2:
                    self.use_log = altdata[2]

                if self.use_log:
                    self.button = Slider(
                        ax=self.axes,
                        label=self.label,
                        valmin=np.log10(lower),
                        valmax=np.log10(upper),
                        valinit=np.log10(value),
                    )
                else:
                    self.button = Slider(
                        ax=self.axes,
                        label=self.label,
                        valmin=lower,
                        valmax=upper,
                        valinit=value,
                    )

                def callback_input(new_value):
                    if self.use_log:
                        new_value = 10**new_value
                    self.gui.input_activation(self.parameter, self.button, new_value)

                self.button.on_changed(callback_input)
            elif uitype == "radio":
                self.button = RadioButtons(axes, altdata)
                self.axes.text(
                    0,
                    0.5,
                    self.label,
                    ha="right",
                    va="center",
                    transform=self.axes.transAxes,
                )

                def callback_radio(label):
                    self.gui.input_activation(self.parameter, self.button, label)

                self.button.on_clicked(callback_radio)
            elif uitype == "push":
                self.button = Button(axes, label)
                # try to find this function in the GUI owner, if not, on the GUI itself
                if hasattr(self.gui.owner, parameter) and callable(getattr(self.gui.owner, parameter)):
                    self.func = getattr(self.gui.owner, parameter)
                elif hasattr(self.gui, parameter) and callable(getattr(self.gui, parameter)):
                    self.func = getattr(self.gui, parameter)
                else:
                    raise KeyError(f'Cannot find a function named "{parameter}" on the GUI or owner.')

                def callback_push(event):
                    self.func()

                self.button.on_clicked(callback_push)
            elif uitype == "custom":
                self.button = Button(axes, label)
            else:
                raise KeyError(f'Unknown uitype ("{uitype}"). Try "push" or "toggle" or "input"')

        def update(self):
            value = self.gui.pars.get(self.parameter)
            if self.uitype == "toggle":
                self.button.label.set_text(str(value))
            elif self.uitype == "text":
                self.button.text_disp.set_text(value)
            elif self.uitype == "number":
                self.button.text_disp.set_text(str(round(value, 4)))
            elif self.uitype == "slider":
                if self.use_log:
                    if value <= 0:
                        value = 0
                    else:
                        value = np.log10(value)
                    self.button.set_val(value)
                    self.button.valtext.set_text(str(round(10**value, 2)))
                else:
                    self.button.set_val(value)
            elif self.uitype == "push":
                self.button.label.set_text(self.parameter + "()")

    class GUI:
        def __init__(self, owner):
            self.owner = owner
            self.pars = {}  # parameters to be passed to owner
            self.buttons = []  # a list of GraphicButton objects for all input/info/action buttons
            self.fig = None  # figure object
            self.gs = None  # gridspec object
            self.subfigs = None  # separate panels help build this up in a modular way
            self.plot_fig = None  # a specific subfigure for display of plots
            self.left_side_fig = None  # a shortcut to the left-side subfigure
            self.auto_update = True  # if true, will also update the owner on every change
            self.distance_pc = 10  # at what distance to show the magnitudes
            self.detection_limit = 0.01  # equivalent to photometric precision
            self.filter_list = [
                "R",
                "V",
                "B",
            ]  # which filters should we use to show the apparent magnitude

            self.update_pars()

            # start building the subfigures and buttons
            self.fig = plt.figure(num="Self lensing GUI", figsize=(15, 10), clear=True)
            self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
            self.gs = self.fig.add_gridspec(ncols=3, nrows=3, width_ratios=[2, 8, 2], height_ratios=[2, 10, 1])
            self.subfigs = []

            # the plotting area
            self.subfigs.append(self.fig.add_subfigure(self.gs[1, 1], facecolor="0.75"))
            self.plot_fig = self.subfigs[-1]

            # add buttons
            # left side:
            self.subfigs.append(self.fig.add_subfigure(self.gs[:, 0], facecolor="0.75"))
            self.left_side_fig = self.subfigs[-1]

            axes_left = self.subfigs[-1].subplots(13, 1)
            ind = 0
            self.add_button(axes_left[ind], "number", "declination")
            ind += 1
            self.add_button(axes_left[ind], "slider", "declination", "", 0, (1e-4, 10, True))
            ind += 1
            self.add_button(axes_left[ind], "number", "semimajor_axis")
            ind += 1
            self.add_button(axes_left[ind], "slider", "semimajor_axis", "", 0, (1e-4, 10, True))
            ind += 1
            self.add_button(axes_left[ind], "toggle", "compact_source")
            ind += 1

            self.add_button(axes_left[ind], "number", "star_mass")
            ind += 1
            self.add_button(axes_left[ind], "slider", "star_mass", "", 0, (0.1, 100, True))
            ind += 1
            self.add_button(axes_left[ind], "number", "star_temp")
            ind += 1
            self.add_button(axes_left[ind], "slider", "star_temp", "", 0, (2500, 15000))
            ind += 1

            self.add_button(axes_left[ind], "number", "lens_mass")
            ind += 1
            self.add_button(axes_left[ind], "slider", "lens_mass", "", 0, (0.1, 100, True))
            ind += 1
            self.add_button(axes_left[ind], "number", "lens_temp")
            ind += 1
            self.add_button(axes_left[ind], "slider", "lens_temp", "", 0, (2500, 15000))
            ind += 1

            # right side:
            self.subfigs.append(self.fig.add_subfigure(self.gs[:, 2], facecolor="0.75"))
            self.right_side_fig = self.subfigs[-1]

            axes_right = self.subfigs[-1].subplots(6, 1, gridspec_kw={"hspace": 1.2})
            ind = 0
            self.add_button(axes_right[ind], "number", "distance_pc")
            ind += 1
            self.add_button(axes_right[ind], "slider", "distance_pc", "", 0, (5, 10000, True))
            ind += 1
            self.add_button(axes_right[ind], "number", "detection_limit")
            ind += 1
            self.add_button(axes_right[ind], "slider", "detection_limit", "", 0, (1e-4, 10, True))
            ind += 1
            self.add_button(axes_right[ind], "text", "filter_list")
            ind += 1
            self.add_button(
                axes_right[ind],
                "radio",
                "time_units",
                None,
                None,
                ("seconds", "minutes", "hours"),
            )
            ind += 1

            # top panel:
            self.subfigs.append(self.fig.add_subfigure(self.gs[0, 1], facecolor="0.75"))
            self.top_fig = self.subfigs[-1]
            # axes_top = self.subfigs[-1].subplots(2, 3)

            # bottom panel:
            self.subfigs.append(self.fig.add_subfigure(self.gs[2, 1], facecolor="0.75"))
            self.bottom_fig = self.subfigs[-1]

            axes_bottom = self.subfigs[-1].subplots(1, 2, gridspec_kw={"width_ratios": (1, 4)})
            self.add_button(axes_bottom[0], "toggle", "auto_update")
            self.add_button(axes_bottom[1], "custom", "calculate", "calculate()")

            def callback_calculate(event):
                self.update(full_update=True)

            self.buttons[-1].button.on_clicked(callback_calculate)

            # make sure all buttons and display are up to date
            self.update_pars()
            self.update_buttons()
            self.update_display()

        def add_button(self, axes, uitype, parameter, label=None, dx=None, altdata=None):
            self.buttons.append(Simulator.GraphicButton(self, axes, uitype, parameter, label, dx, altdata))
            if uitype not in ("push", "custom"):
                if hasattr(self.owner, parameter):
                    self.pars[parameter] = getattr(self.owner, parameter)
                elif hasattr(self, parameter):
                    self.pars[parameter] = getattr(self, parameter)
                else:
                    raise KeyError(f'Parameter "{parameter}" does not exist in GUI or owner.')

        def toggle_activation(self, parameter, button):
            self.pars[parameter] = not self.pars[parameter]
            button.label.set_text(str(self.pars[parameter]))
            self.update()

        def input_activation(self, parameter, button, value):
            if self.pars[parameter] != value:
                self.pars[parameter] = value
                self.update()

        def update(self, full_update=None):
            if full_update is None:
                full_update = self.pars["auto_update"]

            self.left_side_fig.suptitle("Working...", fontsize="x-large")
            self.left_side_fig.canvas.draw()
            self.auto_update = self.pars["auto_update"]

            if full_update:
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

            self.left_side_fig.suptitle(f"runtime: {self.owner.latest_runtime:.4f}s", fontsize="x-large")

        def update_owner(
            self,
        ):  # apply the values in pars to the owner and run code to reflect that
            for k, v in self.pars.items():
                if hasattr(self.owner, k):
                    setattr(self.owner, k, v)

            self.owner.calculate()

        def update_pars(
            self,
        ):  # get updated pars from the owner, after its code was run

            for k, v in vars(self.owner).items():
                self.pars[k] = v

            # add GUI parameters here on top of owner parameters
            self.pars["auto_update"] = self.auto_update

        def update_display(self):  # update the display from the owner's plotting tools
            axes = self.plot_fig.axes
            [self.plot_fig.delaxes(ax) for ax in axes]

            par_list = ["distance_pc", "detection_limit", "filter_list", "auto_update"]
            for p in par_list:
                if p in self.pars:
                    setattr(self, p, self.pars[p])

            self.owner.syst.plot(
                distance_pc=self.distance_pc,
                detection_limit=self.detection_limit,
                filter_list=self.filter_list,
                fig=self.plot_fig,
            )

        def update_buttons(
            self,
        ):  # make sure the buttons all show what is saved in pars
            for b in self.buttons:
                b.update()

            # make the temperature sliders color match the star temperature
            for b in self.buttons:
                if isinstance(b.button, matplotlib.widgets.Slider) and "_temp" in b.parameter:
                    b.button.poly.set_color(get_star_plot_color(b.button.val))


class System:
    def __init__(self):
        self.semimajor_axis = None  # in AU
        self.orbital_period = None  # in units of hours
        self.roche_lobe = None  # in units of solar radii
        self.declination = None  # 90 - i, the orbital inclination (degrees)
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
        self.source_size = None  # in units of Einstein radii
        self.occulter_size = None  # in units of Einstein radii
        self.impact_parameter = None  # in units of Einstein radii
        self.fwhm = None  # the width of the peak (not specific to any survey)
        self.timestamps = None  # for the lightcurve (units are given below, defaults to seconds)
        self._time_units = "seconds"  # for the above-mentioned timestamps
        self.magnifications = None  # the measured lightcurve

        self.surveys = []  # keep a list of all the surveys applied to this system
        # a few dictionaries to be filled by each survey,
        # each one keyed to the survey's name, and containing
        # the results: detection probability for different distances
        # and also the volume associated with each distance.
        self.distances = {}  # parsec
        self.volumes = {}  # parsec^3 for each distance, for a single field
        self.apparent_mags = {}  # in the specific band
        self.precisions = {}  # for each given distance
        self.dilutions = {}  # how much is the lens light diluting the flare
        self.flare_durations = {}  # duration above detection limit (sec)
        self.flare_prob = {}  # probability to detect flare at peak (i.e., best timing)
        # probability of detection in one visit (including duty cycle)
        self.visit_prob = {}
        # probability after multiple visits over the duration of the survey per field
        self.total_prob = {}
        self.total_volumes = {}  # volumes observed over the duration of the survey
        # how many detections per visit per field (on average)
        self.visit_detections = {}
        self.total_detections = {}  # how many detections per field over all visits

        # how much space is covered, weighed by visit prob.
        # multiply this by spatial density to get number of detections
        self.effective_volumes = {}

        # for each parameter in the first block,
        # what are the intervals around the given values,
        # that we can use to estimate how many such systems exist
        # (to fill out the density parameter).
        # E.g., to say there are 10 systems / parsec^3 with lens_mass=0.4
        # you must decide the lens mass range, maybe 0.35 to 0.45,
        # only then can you say how many "such systems" should exist
        # for mass/size/temp of lens/star, inclination, semimajor axis:
        self.par_ranges = {}

        self.spatial_density = None  # how many such system we expect exist per parsec^3

    @property
    def time_units(self):
        return self._time_units

    @time_units.setter
    def time_units(self, new_units):
        self.timestamps *= translate_time_units(self.time_units) / translate_time_units(new_units)
        self._time_units = new_units

    @property
    def inclination(self):
        return 90 - self.declination

    @inclination.setter
    def inclination(self, new_value):
        self.declination = 90 - new_value

    def bolometric_mag(self):
        """
        Get the absolute magnitude of the system, not including the magnification.
        The magnitude is bolometric (includes all wavelengths).

        :return:
            Magnitude measured on Earth for this system at 10pc.
        """

        flux = self.star_flux + self.lens_flux

        # ref: https://www.iau.org/static/resolutions/IAU2015_English.pdf
        return -2.5 * np.log10(flux) + 71.197425 + 17.5  # the 17.5 is to convert W->erg/s

    def ab_mag(self, wavelength, bandwidth, get_dilution=False):
        """
        Find the absolute (10 pc) magnitude of the system in AB magnitudes
        for the filter specified by wavelength and bandwidth.

        :param wavelength: scalar float
            central wavelength of the filter (assume tophat) in nm.
        :param bandwidth: scalar float
            width of the filter in nm.
        :param get_dilution: boolean
            if True will return a 2-tuple with the magnitude and
            the dilution factor of the system.
        :return: 2-tuple or scalar
            if get_dilution is True:
            - the magnitude in AB system based on the provided filter,
                for a system at 10pc.
            - the dilution factor for the lightcurve magnification
                (1 is only the source, 0.5 for equal source/lens, etc).
            if not, return only the magnitude
        """
        # a few important constants
        c_nm = 2.99792458e17  # speed of light in nm/s
        sun_radius = 6.957e10  # in cm
        pc = 3.086e18  # parsec in cm
        one_over_h = 1 / 6.62607004e-27  # 1/h in units of erg * Hz

        filt_bounds = (wavelength - bandwidth / 2, wavelength + bandwidth / 2)
        filter_low_f = c_nm / max(filt_bounds)
        filter_high_f = c_nm / min(filt_bounds)
        flux1 = integ.quad(self.black_body, filter_low_f, filter_high_f, args=(self.star_temp, True))[0]
        # integrate over surface of star, divide by sphere at 10pc
        flux1 *= (self.star_size * sun_radius) ** 2 / (10 * pc) ** 2  # erg/s/cm^2

        if self.lens_flux > 0 and self.lens_temp > 0:
            flux2 = integ.quad(
                self.black_body,
                filter_low_f,
                filter_high_f,
                args=(self.lens_temp, True),
            )[0]
            flux2 *= (self.lens_size * sun_radius) ** 2 / (10 * pc) ** 2  # erg/s/cm^2
        else:
            flux2 = 0

        # this flux is equivalent to flux from flat spectrum source with how many janskies?
        # flat_source = (filter_high_f - filter_low_f)  # this is true if we don't count photons
        flat_source = one_over_h * np.log(filter_high_f / filter_low_f)  # integrate over 1/(hf)
        equiv_flux1 = flux1 / flat_source  # in erg/s/cm^2
        equiv_flux2 = flux2 / flat_source

        mag = -2.5 * np.log10(equiv_flux1 + equiv_flux2) - 48.60  # -2.5 * np.log10(equiv_flux / 3631e-23)

        if get_dilution:
            dilution = flux1 / (flux1 + flux2)
            return mag, dilution
        else:
            return mag

    def apply_distance(self, abs_mag, distance, wavelength=None, bandwidth=None):
        """
        Apply a distance modulus (and possibly extinction) to
        :param abs_mag: float array or scalar
            absolute magnitude (at 10pc) of the system.
            can be a scalar or array, but the two first
            inputs must be braodcastable.
        :param distance: float array or scalar
            distance to the system in pc.
        :param wavelength: tbd
        :param bandwidth: tbd
        :return:
            the magnitude after applying the
            decrease in the amount of light
            due to the distance (1/r^2 law)
            and extinction (to be added).
        """
        mag = abs_mag + 5 * np.log10(distance / 10)

        if wavelength is not None:
            mag += distance / 1000  # the simplest prescription of 1mag / kpc
            # TODO: calculate the extinction using a better model!
            # TODO: check if bandwidth is not None and use that, too
            pass

        return mag

    @staticmethod
    def scale_with_temperature(temp1, temp2, wavelength, bandwidth):
        """
        Calculate the expected scaling up of the total effective volume
        that happens when the temperature is changed from temp1 to temp2.
        Takes into account the increased flux inside the filter, and scales
        each distance by an amount that reflects the change in brightness.
        The scaling of distance is then translated to scaling of volume.

        :param temp1: scalar float
            current temperature of the dominant body in the system.
        :param temp2: scalar float
            new temperature of the dominant body (assume the second body
            either stays sub-dominant or otherwise has similar temperature
            to the dominant one, before and after the transformation).
        :param wavelength: scalar float
            central wavelength of filter passband (in nm).
        :param bandwidth: scalar float
            width of the filter passband (in nm).
        :return: scalar float
            scaling factor of the effective volume.
        """
        c_nm = 2.99792458e17  # speed of light in nm/s
        filt_bounds = (wavelength - bandwidth / 2, wavelength + bandwidth / 2)
        filter_low_f = c_nm / max(filt_bounds)
        filter_high_f = c_nm / min(filt_bounds)
        flux1 = integ.quad(System.black_body, filter_low_f, filter_high_f, args=(temp1, True))[0]
        flux2 = integ.quad(System.black_body, filter_low_f, filter_high_f, args=(temp2, True))[0]

        distance_ratio = np.sqrt(flux2 / flux1)

        volume_ratio = distance_ratio**3

        return volume_ratio

    @staticmethod
    def black_body(f, temp, photons=False):
        """
        Black body radiation per frequency.

        :param f: float array or scalar
            Array or scalar of frequencies (in Hz).
        :param temp: float array or scalar
            Array or scalar of frequencies (in Kelvin).
            The two first inputs must be broadcastable.
        :param photons: bool
            When true, will return the number of photons
            in this frequency range, instead of the
            flux spectral density.
        :return:
            The flux spectral density in units of
            ergs per second per cm^2 per Hz per steradian.
            If photons=True will instead give
            photons per second per cm^2 per Hz per steradian.
            In either case this number must be multiplied
            by the surface area of the star,
            and divided by the 4pi*D^2 to get the
            flux at the distance D.
        """

        if photons:  # get the number of photons instead of the total flux
            const1 = 6.990986484228638e-21  # 2 * pi / c ** 2 = 2 / 2.99792458e10 ** 2
        else:
            const1 = 4.632276581355286e-47  # 2 * pi * h / c ** 2 = 2 * 6.626070e-27 / 2.99792458e10 ** 2

        const2 = 4.7992429647216633e-11  # h/k = 6.626070e-27 / 1.380649e-16

        return const1 * f ** (3 - photons) / (np.exp(const2 * f / temp) - 1)

    # to be deprecated:
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
            return 0  # TODO: add the dilution in this case

        filt_la1 = wavelength - bandwidth / 2
        filt_la2 = wavelength + bandwidth / 2

        bolometric1 = integ.quad(black_body, *find_lambda_range(self.star_temp), args=(self.star_temp, True))[0]
        in_band1 = integ.quad(black_body, filt_la1, filt_la2, args=(self.star_temp, True))[0]
        fraction1 = in_band1 / bolometric1
        # print(f'bolometric1= {bolometric1} | in_band1= {in_band1}')

        if self.lens_flux > 0 and self.lens_temp > 0:
            bolometric2 = integ.quad(
                black_body,
                *find_lambda_range(self.lens_temp),
                args=(self.lens_temp, True),
            )[0]
            in_band2 = integ.quad(black_body, filt_la1, filt_la2, args=(self.lens_temp, True))[0]
            fraction2 = in_band2 / bolometric2
        else:
            fraction2 = 0

        flux_band = self.star_flux * fraction1 + self.lens_flux * fraction2
        flux_bolo = self.star_flux + self.lens_flux

        ratio = flux_band / flux_bolo
        dilution = (self.star_flux * fraction1) / (self.star_flux * fraction1 + self.lens_flux * fraction2)

        return 2.5 * np.log10(ratio), dilution

    def merger_time(self):
        """
        Calculate the merger time (in Gyr) for this system.
        ref: https://ui.adsabs.harvard.edu/abs/2016ApJ...824...46B/abstract
        """
        return (
            47.925
            * (self.star_mass + self.lens_mass) ** (1 / 3)
            / (self.star_mass * self.lens_mass)
            * (self.orbital_period / 24) ** (8 / 3)
        )

    def is_detached(self):
        return self.roche_lobe > self.star_size

    def plot(
        self,
        detection_limit=0.01,
        distance_pc=10,
        filter_list=["R", "V", "B"],
        fig=None,
        font_size=14,
    ):

        if fig is None:
            fig, ax = plt.subplots(1, 2)
        else:
            ax = fig.subplots(2, 1)

        ax[0].plot(
            self.timestamps,
            self.magnifications,
            "-b",
            lw=3.0,
            label=f"magnification (max= {np.max(self.magnifications):.4f})",
        )
        ax[0].set_xlabel(f"time [{self.time_units}]", fontsize=font_size)
        ax[0].set_ylabel("magnification", fontsize=font_size)

        # show the detection limit
        ax[0].plot(
            self.timestamps,
            np.ones(self.magnifications.shape) * (1 + detection_limit),
            "--g",
            label=f"detection limit ({int(detection_limit * 100) : d}%)",
        )

        # show the region above the limit
        idx = self.magnifications - 1 > detection_limit
        if np.any(idx):
            width = max(self.timestamps[idx]) - min(self.timestamps[idx])
            ax[0].plot(
                self.timestamps[idx],
                self.magnifications[idx],
                "-r",
                label=f"event time ({width:.1f} {self.time_units})",
                lw=3.0,
            )

        legend_handles, legend_labels = ax[0].get_legend_handles_labels()

        # add the period to compare to the flare time
        legend_handles.append(Line2D([0], [0], lw=0))
        period = self.orbital_period * 3600 / translate_time_units(self.time_units)
        # legend_labels.append(f'Period= {period:.1f} {self.time_units}')
        legend_labels.append(f"Period= {self.period_string()}")

        # explicitly show the duty cycle
        if np.any(idx):
            legend_handles.append(Line2D([0], [0], lw=0))
            legend_labels.append(f"Duty cycle= 1/{period / width:.1f}")

        # add the magnitudes at distance_pc
        if isinstance(filter_list, str):
            filter_list = filter_list.replace("[", "").replace("]", "").replace("'", "").strip().split(",")

        mag_str = f"Mag (at {int(distance_pc):d}pc):"
        for i, filt in enumerate(filter_list):
            filter_pars = default_filter(filt.strip())
            magnitude = self.apply_distance(self.ab_mag(*filter_pars), distance_pc, *filter_pars)
            if i % 3 == 0:
                mag_str += "\n"
            mag_str += f" {filt}~{magnitude:.2f},"

        # add results to legend
        legend_labels.append(mag_str)
        legend_handles.append(Line2D([0], [0], lw=0))

        ax[0].legend(
            legend_handles,
            legend_labels,
            bbox_to_anchor=(1.00, 0.95),
            loc="upper right",
            fontsize=font_size - 4,
        )

        # add a cartoon of the system in a subplot
        ax[0].set(position=[0.1, 0.1, 0.85, 0.85])
        ax[1].set(position=[0.15, 0.35, 0.3, 0.25])
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)

        # set the scale of the cartoon (for the object sizes at least)
        scale = max(self.star_size, self.einstein_radius)

        # add the star / source
        solar_radii = "$R_\u2609$"
        solar_mass = "$M_\u2609$"
        star_label = f"{self.star_type} source: {self.star_mass:.2f}{solar_mass}, " f"{self.source_size:.2f}$R_E$"
        # f'{self.star_size:.3f}{solar_radii}, {self.source_size:.2f}$R_E$'
        star = plt.Circle(
            (-2, 0),
            self.star_size / scale,
            color=get_star_plot_color(self.star_temp),
            label=star_label,
        )
        ax[1].set_xlim((-3.2, 3.2))
        ax[1].set_ylim((-1.5, 1.5))
        ax[1].set_aspect("equal")
        ax[1].add_patch(star)

        # add the lensing object
        lens_label = f"{self.lens_type} occulter: {self.lens_mass:.2f}{solar_mass}, " f"{self.occulter_size:.2f}$R_E$"
        # f'{self.lens_size:.3f}{solar_radii}, {self.occulter_size:.2f}$R_E$'

        lt = self.lens_type.strip().upper()

        if lt == "WD":
            lens = plt.Circle(
                (2, 0),
                self.lens_size / scale,
                color=get_star_plot_color(self.lens_temp),
                label=lens_label,
            )
        elif lt == "NS":
            lens = plt.Circle((2, 0), 0.1, color=get_star_plot_color(self.lens_temp), label=lens_label)
        elif lt == "BH":
            lens = plt.Circle((2, 0), 0.05, color="black", label=lens_label)
        else:
            raise KeyError(f'Unknown lens type ("{self.lens_type}"). Try using "WD" or "BH"... ')

        ax[1].add_patch(lens)

        # add the Einstein ring
        ring = plt.Circle(
            (2, 0),
            self.einstein_radius / scale,
            color="r",
            linestyle=":",
            fill=False,
            label=f"Einstein radius= {self.einstein_radius:4.3f}{solar_radii}",
        )
        ax[1].add_patch(ring)

        phi = np.linspace(0, 2 * np.pi, 1000)
        x = 2 * np.cos(phi)
        y = 2 * np.sin(phi) * np.sin(np.pi / 180 * (90 - self.inclination) * 50)
        if self.semimajor_axis < 1:
            fmt = f".{int(np.ceil(-np.log10(self.semimajor_axis)))}f"
        else:
            fmt = ".2f"
        ax[1].plot(
            x,
            y,
            "--k",
            label=f"a= {self.semimajor_axis:{fmt}}AU, "
            rf"i={self.inclination:.3f}$^\circ$, "
            f"b={self.impact_parameter:.2f}$R_E$",
        )

        legend_handles, legend_labels = ax[1].get_legend_handles_labels()

        # replace the legend markers
        legend_handles[0] = Line2D([0], [0], lw=0, marker="o", color=get_star_plot_color(self.star_temp))  # star

        if lt == "BH":
            legend_handles[1] = Line2D([0], [0], lw=0, marker="o", color="black")  # lens
        else:
            legend_handles[1] = Line2D([0], [0], lw=0, marker="o", color=get_star_plot_color(self.lens_temp))  # lens

        legend_handles[2] = Line2D([0], [0], linestyle=":", color="r")  # einstein ring

        # order = [2, 0, 3, 1]
        # legend_handles = [legend_handles[i] for i in order]
        # legend_labels = [legend_labels[i] for i in order]

        ax[1].legend(
            legend_handles,
            legend_labels,
            bbox_to_anchor=(0.5, 1.04),
            loc="lower center",
            fontsize=font_size - 4,
        )

        [l.set_fontsize(font_size) for l in ax[0].get_xticklabels()]
        [l.set_fontsize(font_size) for l in ax[0].get_yticklabels()]

        return ax

    def period_string(self):
        P = self.orbital_period
        if P < 0.01:
            return f"{P * 3600:.2g} s"
        if P < 0.1:
            return f"{P * 60:.2g} min"
        if P > 24:
            return f"{P / 24:.2g} days"

        return f"{P:.2g} h"

    def print(self, pars=None, surveys=None):

        par_dict = {
            "inclination": ("i", ""),
            "einstein_radius": ("R_E", "Rsun"),
            "lens_mass": ("M_l", "Msun"),
            "lens_temp": ("T_l", "K"),
            "lens_size": ("R_l", "Rsun"),
            "lens_type": ("lens", ""),
            "star_mass": ("M_s", "Msun"),
            "star_temp": ("T_s", "K"),
            "star_size": ("R_s", "Rsun"),
            "star_type": ("source", ""),
            "semimajor_axis": ("a", "AU"),
            "orbital_period": ("P", ""),
            "roche_lobe": ("Roche", "Rsun"),
            "source_size": ("R_source", "R_E"),
            "occulter_size": ("R_occulter", "R_E"),
            "impact_parameter": ("b", "R_E"),
        }

        if pars is None:
            pars = [
                ["lens_type", "lens_mass", "lens_size", "lens_temp"],
                ["star_type", "star_mass", "star_size", "star_temp"],
                ["semimajor_axis", "orbital_period", "inclination", "impact_parameter"],
                ["einstein_radius", "source_size", "occulter_size"],
            ]

        # a scalar string should be turned into a list with one member
        if isinstance(pars, str):
            pars = [pars]
        if isinstance(pars, list):  # this fails if pars is False
            for p_list in pars:
                if not isinstance(p_list, list):
                    p_list = [p_list]
                new_str = []
                for p in p_list:
                    if p == "orbital_period":
                        value = self.period_string()
                    elif p == "inclination":
                        if self.inclination < 90:
                            value = f"90 - {90 - self.inclination:.2g} deg"
                        else:
                            value = "90 deg"
                    else:
                        value = getattr(self, p)
                        if isinstance(value, float) and value != 0:
                            value = f"{value:.2g}"

                    if p in par_dict:
                        if value and par_dict[p][1]:
                            new_str.append(f"{par_dict[p][0]}: {value} {par_dict[p][1]}")
                        else:
                            new_str.append(f"{par_dict[p][0]}: {value}")
                    else:
                        new_str.append(f"{p}: {value}")
                print(" | ".join(new_str))

        print(f"flare peak: {max(self.magnifications)-1:.2g} | flare FWHM: {self.fwhm:.2g} s")

        if surveys is None:
            surveys = list(self.apparent_mags.keys())

        # a scalar string should be turned into a list with one member
        if isinstance(surveys, str):
            surveys = [surveys]

        if isinstance(surveys, list):  # this fails if surveys is False
            for s in surveys:

                survey = next((sur for sur in self.surveys if sur.name == s), None)

                if survey is None or len(self.flare_durations[s]) == 0:
                    continue

                print()
                print(f"Survey results for {s}:")

                series_time = survey.series_length * (survey.exposure_time + survey.dead_time)

                new_str = []
                t_flare = max(self.flare_durations[s])
                new_str.append(f"flare duration: {t_flare:.2g} s")
                new_str.append(f"period: {self.period_string()}")
                new_str.append(f"duty cycle: {t_flare / self.orbital_period / 3600:.2g}")
                new_str.append(f"exp time: {survey.exposure_time}")
                new_str.append(f"series time: {series_time:.2g} s")
                print(" | ".join(new_str))

                new_str = []
                new_str.append(f"flare prob: {max(self.flare_prob[s]):.2g}")
                visit_prob = max(self.visit_prob[s])
                new_str.append(f"visit prob: {visit_prob:.2g}")
                # if visit_prob != max(self.visit_detections[s]):
                new_str.append(f"visit det: {max(self.visit_detections[s]):.2g}")
                new_str.append(f"total det: {max(self.total_detections[s]):.2g}")
                print(" | ".join(new_str))

                new_str = []
                new_str.append(f"max dist: {max(self.distances[s]):.2g} pc")
                new_str.append(f"volume: {np.sum(self.volumes[s]):.2g} pc^3")
                new_str.append(f"total vol: {np.sum(self.total_volumes[s]):.2g} pc^3")
                new_str.append(f"eff. vol: {self.effective_volumes[s]:.2g} pc^3")
                print(" | ".join(new_str))


# to be deprecated:
def find_lambda_range(temp):
    """
    Find the range of temperatures that are relevant for
    a black body of this temperature.
    It uses Wein's displacement law to find the peak radiation,
    and gives *10 and /10 of that value.
    :param temp:
        temperature of the black body.
    :return: 2-tuple
        the minimal and maximal wavelength range (in nm).
    """
    wein_b = 2.897e6  # in units of nm/K
    best_la = wein_b / temp
    la1 = best_la / 10
    la2 = best_la * 10
    return la1, la2


# to be deprecated:
def black_body(la, temp, photons=False):  # black body radiation
    """
    get the amount of radiation expected from a black body
    of the given temperature "temp", at the given wavelengths "la".
    :param la: float array or scalar
        wavelength(s) where the radiation should be calculated.
    :param temp: float scalar
        temperature of the black body.
    :param photons: boolean
        use this to get the BB number of photons,
        instead of total energy.
    :return:
        return the radiation (in units of Watts per steradian per m^2 per nm)
        if photons=True, returns photons per second per steradian per m^2 per nm.
    """
    const = 0.014387773538277204  # h*c/k_b = 6.62607004e-34 * 299792458 / 1.38064852e-23
    amp = 1.1910429526245744e-25  # 2*h*c**2 * (nm / m) = 2*6.62607004e-34 * 299792458**2 / 1e9 the last term is units
    la = la * 1e-9  # convert wavelength from nm to m

    return amp / (la ** (5 - photons) * (np.exp(const / (la * temp)) - 1))


def translate_time_units(units):
    d = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 3600 * 24,
        "years": 3600 * 24 * 365.25,
    }
    if units not in d:
        raise KeyError(f'Unknown units specified ("{units}"). Try "seconds" or "hours". ')
    return d[units.lower()]


def get_star_plot_color(temp):

    lim1 = 2000
    lim2 = 10000
    push_low = 0.05

    cmp = plt.get_cmap("turbo")

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
        "U": (365, 66),
        "B": (445, 94),
        "G": (464, 128),
        "V": (551, 88),
        "R": (658, 138),
        "I": (806, 149),
        "F500W": (500, 200),
        "WHITE": (550, 300),
        "TESS": (825, 450),
    }
    # TODO: make sure these numbers are correct!

    if filter_name.upper() not in filters:
        # raise KeyError(f'Unknonwn filter name "{filter_name}". Use "V" or "r" etc. ')
        return None, None

    return filters[filter_name.upper()]


def guess_compact_type(mass):
    if mass is None:
        return None
    elif mass <= 1.2:  # was 1.454
        return "WD"
    elif mass <= 2.5:  # was 2.8
        return "NS"
    else:
        return "BH"


def main_sequence_size(mass):
    """
    reference: https://ui.adsabs.harvard.edu/abs/1991Ap%26SS.181..313D/abstract (page 8, empirical data)
    """
    if mass is None:
        return None
    elif mass <= 1.66:
        return 1.06 * mass**0.945
    else:
        return 1.33 * mass**0.555


def compact_object_size(mass):
    """
    reference: https://ui.adsabs.harvard.edu/abs/2002A%26A...394..489B/abstract (Equation 5)
    """
    if mass is None:
        return None
    elif mass < 1.454:
        mass_chand = mass / 1.454
        return 0.01125 * np.sqrt(mass_chand ** (-2 / 3) - mass_chand ** (2 / 3))
    else:
        return 0  # we can use Schwartschild radius for BH but what about NS?


if __name__ == "__main__":

    s = Simulator()
    # s.make_gui()
    s.calculate(
        star_mass=1.0,
        star_size=None,
        star_type=None,
        star_temp=5778,
        lens_mass=3.0,
        lens_type=None,
        inclination=89.0,
        semimajor_axis=0.1,
    )
