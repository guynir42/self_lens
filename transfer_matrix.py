import numpy as np
import copy
import matplotlib.pyplot as plt
from datetime import datetime
from timeit import default_timer as timer

"""
This module generates the core (radial) lightcurves and offsets
from micro-lensing events where the size of the source (and occulter)
are all similar to the lens size (the Einstein radius)

The TransferMatrix is used to pre-calculate many lightcurves
for various micro-lensing geometries, and can interpolate between
those results to give any values in range.
The matrix transfers the light from each annulus on the source
separately, and gives the magnification of that annulus, as well
as the 1st moment for that annulus (that must be normalized!)
so that the total magnification and astrometric offset can be
calculated given a source size and brightness profile.

The StarProfile makes it easy to define a star with either
uniform brightness or with limb darkening, and then apply it
to a TransferMatrix to get the total magnification and offset.

Also provided are some standalone functions that are used
to calculate the shape of the lensed image.
We start by calculating the images of the edges
of a circular source to get a large and small image contours,
using the draw_contours() function.
Then we turn the contours into a boolean map showing either
two separate images or an image with a hole in it, using
the make_surface() function.
Finally, if a non-zero sized occulter (physical size of
the lensing object) is used, the shape of the occulter
is removed from the surface map using remove_occulter().
The function single_geometry() combines these three steps
to calculate the magnification and offset for a combination
of source radius, occulter radius, and distance between
lens and source centers.
The function radial_lightcurve() calculates multiple
such geometries along an array of distances,
generating a "radial magnification curve" (or offset curve).
These curves can later be sampled into real event lightcurves,
provided an impact parameter and velocity of the lens.

All units are normalized to the Einstein radius.
"""

class TransferMatrix:
    """
    The TransferMatrix is used to pre-calculate the magnification
    and centroid offsets for all combinations of source and occulter size,
    for all distances, given the ranges (min/max values) given.

    Then the existing matrix is saved/loaded from disk, and
    used to make quick lightcurve estimates.
    Any values between the given source/occulter radii or distances
    are interpolated linearly, which is reasonable if the step sizes
    are not too big and the functions don't change too rapidly.

    The name TransferMatrix refers to the transformation of the
    light from each annulus on the source into the light seen
    after it was lensed (i.e., magnified).

    The data saved in the output arrays of the TransferMatrix
    is arranged into four arrays: "flux", "input_flux", "magnification"
    and "moment". These are 3D arrays, with axis 0 for occulter radii,
    axis 1 for source radii, and axis 2 for distances
    (between source and lens centers).
    The "input_flux" specifies the surface area of each annulus
    (between r_i and r_i+1) in units of Einsten radius squared.
    The "flux" array specifies the total surface area of each
    annulus in the source, after it was imaged by the lens
    (i.e., magnified). The different distances and occulter
    radii change the amount of light that is transferred to
    the final image.
    The "magnification" is just the division of the two
    above-mentioned arrays, and can be used to estimate
    the magnification of each annulus.
    The "moment" array gives the center of light of each
    annulus, but without the normalization. This is because
    the normalization must come after integrating over all
    the different annuli in the source surface,
    with the correct weights given by the star profile.
    The moments can be summed over the different radii,
    and the "flux" array can be summed as well.
    Then their ratio gives the normalized offset
    of the center of light from the center of the source.

    USAGE: choose min/max/step for distances, source radii,
           and occulter radii. Call make_matrix() and wait.
           Once done, use save() to keep the matrix
           for later use. Retrieve the matrix from file
           using the load() function.
           Once ready, use radial_lightcurve() to produce
           a set of magnifications and offsets for a list
           of distances, interpolated from the existing data.

    All units are normalized to the Einstein radius.
    """

    def __init__(self):
        self.num_points = 1e5
        self.num_pixels = 1e7
        self.min_dist = 0
        self.max_dist = 20
        self.step_dist = 0.1

        self.min_source = 0.1
        self.max_source = 10
        self.step_source = 0.1

        self.max_occulter = 3
        self.step_occulter = 0.1

        self.distances = np.array([])
        self.source_radii = np.array([])
        self.occulter_radii = np.array([])

        # the following are 3D matrices, corresponding
        # to the amount of flux and 1st moments,
        # that are produced by one uniform annulus of
        # the source, with surface brightness = 1
        # the axes of each data matrix correspond to
        # 0: lens radius, 1: source size, 2: distance
        self.flux = np.array([])
        self.input_flux = np.array([])
        self.magnification = np.array([])
        self.moment = np.array([])

        # some info about the calculations
        self.calc_time = 0
        self.complete = False
        self.timestamp = ''
        self.notes = ''

    def update_axes(self):
        """make the arrays for distances, source and occulter radii"""
        num_dist = int(np.round((self.max_dist - self.min_dist) / self.step_dist) + 1)
        self.distances = np.round(
            np.linspace(self.min_dist, self.max_dist, num_dist, endpoint=True), 5,
        )

        num_radii = int(np.round((self.max_source - self.min_source) / self.step_source) + 1)
        self.source_radii = np.round(
            np.linspace(self.min_source, self.max_source, num_radii, endpoint=True), 5
        )

        num_occulters = int(np.round(self.max_occulter / self.step_occulter) + 1)
        self.occulter_radii = np.round(
            np.linspace(0, self.max_occulter, num_occulters, endpoint=True), 5,
        )

    def allocate_arrays(self):
        """make the arrays used to store the flux, input_flux, magnification and moments. """
        self.flux = np.zeros(
            (self.occulter_radii.size, self.source_radii.size, self.distances.size),
            dtype=np.single,
        )
        self.input_flux = np.zeros(
            (self.occulter_radii.size, self.source_radii.size, self.distances.size),
            dtype=np.single
        )
        self.moment = np.zeros(
            (self.occulter_radii.size, self.source_radii.size, self.distances.size),
            dtype=np.single,
        )

    def update_notes(self):
        """make a string with the most important parameters for this matrix. """
        self.notes = f'distances= {self.min_dist}-{self.max_dist} ({self.step_dist} step) | ' \
                     f'source radii= {self.min_source}-{self.max_source} ({self.step_source} step) | ' \
                     f'occulter radii= 0-{self.max_occulter} ({self.step_occulter} step) | ' \
                     f'num pixels= {self.num_pixels:.0f} | num points= {self.num_points:.0f} | ' \
                     f'calculation time= {self.calc_time:.1f}s'

    def data_size(self):
        """get the number of data points in each of the data arrays. """
        return self.distances.size * self.source_radii.size * self.occulter_radii.size

    def make_matrix(self, plotting=False):
        """
        Populate the matrix with the magnification and moment results for each value combination
        in the distances, source_radii, and occulter_radii arrays.
        Will calculate the result for each annulus on the source, up to the maximum source radii.
        This function takes a long time to run (possibly hours).

        :param plotting: if true, will produce a surface plot of the lensed geometry for each iteration
        :return: None
        """

        print("calculating matrix! ")
        self.update_axes()
        self.update_notes()
        self.allocate_arrays()

        if plotting:
            plt.ion()

        t1 = timer()

        for i, d in enumerate(self.distances):

            for j, source_radius in enumerate(self.source_radii):

                (z1x, z1y, z2x, z2y) = draw_contours(d, source_radius, points=self.num_points)

                (im, x_grid, y_grid, resolution) = make_surface(
                    d,
                    source_radius,
                    z1x,
                    z1y,
                    z2x,
                    z2y,
                    pixels=self.num_pixels,
                )

                r_squared_grid = x_grid ** 2 + y_grid ** 2

                for k, occulter_radius in enumerate(self.occulter_radii):
                    # t0 = timer()
                    (im2, changed_flag) = remove_occulter(im, occulter_radius, r_squared_grid, return_changed_flag=True)
                    # print(f'timer to remove occulter: {timer()-t0:.3f}s'); t0 = timer()

                    self.input_flux[k, j, i] = np.pi * source_radius ** 2

                    if k == 0 or changed_flag:  # must calculate sums

                        self.flux[k, j, i] = np.sum(im2) * 2 / resolution ** 2
                        self.moment[k, j, i] = np.sum(im2 * (x_grid - d)) * 2 / resolution ** 2

                    else:
                        self.flux[k, j, i] = self.flux[k - 1, j, i]
                        self.moment[k, j, i] = self.moment[k - 1, j, i]

                    # print(f'timer to calculate sums: {timer() - t0:.3f}s')

                    if plotting:
                        offset = self.moment[k, j, i] / self.flux[k, j, i]
                        mag = self.flux[k, j, i] / self.input_flux[k, j, i]
                        plot_geometry(im2, x_grid, y_grid, source_radius, d, occulter_radius, mag, offset)

            self.calc_time = timer() - t1

            print(
                f"{i+1:3d}/{self.distances.size} | Distance= {d} | runtime = {self.calc_time:.1f}s"
            )

        self.flux = np.diff(self.flux, axis=1, prepend=0)
        self.input_flux = np.diff(self.input_flux, axis=1, prepend=0)
        self.magnification = self.flux / self.input_flux

        self.moment = np.diff(self.moment, axis=1, prepend=0)

        self.complete = True
        self.timestamp = str(datetime.utcnow())

    def radial_lightcurve(
        self, source=3, distances=None, occulter_radius=0, pixels=1e6, get_offsets=False
    ):
        """
        :param source:
            Give the size of the source, or a brightness profile
            (array or StarProfile object) that specifies the surface brightness
            of each annulus. Such an array must match in size the
            matrix's source_radii array, and each value of surface
            brightness in the profile must correspond to that radius in source_radii.
        :param distances:
            A list of distances (between lens and source centers)
            where the results should be calculated.
            If None (default) will return the values at the matrix's distances array.
            Otherwise, the values are linearly interpolated from the distances
            that were calculated by the matrix. If any of the distances surpass
            the matrix's maximum distance, we replace the results with the
            point-source approximation (the reasoning is that far away from the lens,
            each point on the source will have a similar magnification, which is
            given by the point-source approximation).
        :param occulter_radius:
            Provide the radius of the physical extent of the lensing object.
            For very tiny objects (e.g., black holes) this can be effectively zero (default).
            Otherwise, the results are interpolated from the results calculated
            using the matrix's occulter_radii array.
            The value requested cannot exceed the maximum of the matrix
            (will throw a ValueError) -- if your occulter is very big you'll
            want to use eclipsing binary / transiting planet codes.
        :param get_offsets:
            If True, returns a tuple with the magnification and the astrometric offset.
            If False (default) returns just the magnifications.
        :return:
            The magnifications are the total magnification of the source,
            given the occulter radius, at each of the distances requested.
            An additional output, the astrometric offset of the source center
            (in Einstein units) is given if setting get_offsets=True.
        """
        if distances is None:
            d = self.distances
        else:
            d = distances

        d = np.array(d)

        # figure out the stellar light profile
        N = self.flux.shape[1]  # number of source radii
        frac = 1  # the relative weight between first and second profiles (if given!)
        if isinstance(source, StarProfile):
            star_profile = source.get_matrix()
        elif type(source) is np.ndarray:
            if source.ndim == 1 and source.shape[0] == N:
                star_profile = np.expand_dims(source, axis=[0, 2])
            elif source.ndim == 2 and source.shape[1] == N:
                star_profile = np.expand_dims(source, axis=2)
            elif source.ndim == 3 and source.shape[1] == N:
                star_profile = source
        elif hasattr(source, "__len__") and len(source) == N:
            star_profile = np.expand_dims(np.array(source), axis=[0, 2])
        elif np.isscalar(source):
            if source < np.min(self.source_radii) or source > np.max(self.source_radii):
                raise ValueError(
                    f'Requested source size ({source}) '
                    f'is out of range for this matrix ({np.min(self.source_radii)}-{np.max(self.source_radii)})'
                )
            # check if requested source radius is an exact match,
            # if it isn't, need some interpolation
            if np.any(source == self.source_radii):
                star_profile = np.zeros((1, N, 1))
                star_profile[:, source >= self.source_radii, :] = 1
            else:
                # index above the requested point
                # (must interpolate btw idx and idx-1)
                star_profile = np.zeros((2, N, 1))
                star_profile[:, source >= self.source_radii, :] = 1

                idx = np.argmax(self.source_radii > source)
                star_profile[1, idx, :] = 1
                frac = (source - self.source_radii[idx-1]) / (self.source_radii[idx] - self.source_radii[idx-1])

        else:
            raise TypeError(
                'Must provide "source" input as a numeric scalar, '
                f"a list, or an array (of size {N}). "
            )

        # star_profile /= np.sum(star_profile, axis=1, keepdims=True)  # normalize to unity

        if occulter_radius > np.max(self.occulter_radii):
            raise ValueError(
                f"requested occulter radius {occulter_radius} is out of range "
                f"for this matrix ({np.min(self.occulter_radii)}-{np.max(self.occulter_radii)})"
            )

        occulter_idx = occulter_radius == self.occulter_radii

        if np.any(occulter_idx):  # no need to interpolate on occulter size
            idx = [np.argmax(occulter_idx)]  # single index
        else: # need to interpolate btw two occulter sizes
            # index of occulter radius below requested value
            idx = np.argmax(self.occulter_radii < occulter_radius)
            idx = [idx, idx + 1]  # two indices
        flux = self.interp_on_profile(self.flux[idx, :, :], star_profile, frac)
        in_flux = self.interp_on_profile(self.input_flux[idx, :, :], star_profile, frac)
        mag = flux / in_flux

        if distances is not None:
            mag = np.interp(d, self.distances, mag, right=1)
            mag[d > self.max_dist] = point_source_approximation(d[d > self.max_dist])

        if not get_offsets:
            return mag
        else:
            moments = self.interp_on_profile(self.moment[idx, :, :], star_profile, frac)
            offsets = moments / flux

            if distances is not None:
                offsets = np.interp(d, self.distances, offsets, right=0)
            return mag, offsets

    @staticmethod
    def interp_on_profile(data, profile, frac):
        """
        Internal method used for interpolation btw two
        star profiles for e.g., similar star radii.
        It will multiply the data (flux, moment, etc)
        with the star profile, that has a 0th dimension
        with either length one or two.
        If it is length one, that means no interpolation.
        If it is length two, we weigh the different results
        with "frac" and then add them together.
        If the data's 0th dim is non-scalar,
        it just loops over each one.
        In any case, the shape of the returned value
        will have dim 1 removed (summed over).
        """
        for i in range(data.shape[0]):
            d = np.sum(data[i] * profile, axis=1)  # apply the source profile
            if profile.shape[0] > 1:
                d[0, :] *= 1 - frac
                d[1, :] *= frac
                d = np.sum(d, axis=0)
            else:
                d = d[0, :]
            data[i] = d

        return data

    def save(self, filename="matrix"):
        """
        Save the matrix data into an .npz file on disk.
        Also save an accompanying .txt file with the same
        name with a summary of the matrix statistics
        (see the update_notes() function).


        :param filename: string
            The filename used to save the matrix.
            The default is "matrix" but we encourage
            to save the matrix with some parameters in the name.
            E.g., "matrix_SR1.000-2.000_D0.000-5.000" would
            be a good name for a matrix with source radius (SR)
            between 1 and 2, and distances (D) between 0 and 5.
            The filename extension .npz is optional
            (will be added automatically if missing).
        """
        if filename.endswith('.npz'):
            filename = filename[:-4]
        np.savez_compressed(filename, **vars(self))
        with open(f'{filename}.txt', 'w') as text_file:
            text_file.write(self.notes)

    def load(self, filename):
        """
        Load a the matrix data from an .npz file, into an existing matrix.

        :param filename:
            The filename to load the data from.
            In this case the filename must include the .npz extension.
        """
        A = np.load(filename, allow_pickle=True)
        for k in vars(self).keys():
            val = A[k]
            if val.ndim == 0:
                val = val.item()
            setattr(self, k, val)

    @classmethod
    def from_file(cls, filename):
        """
        Create a new matrix based on data in an .npz file.

        :param filename:
            Name of file to load from (see the load() function).
        :return:
            A newly created TransferMatrix object,
            with the data loaded from file.
        """
        T = TransferMatrix()
        T.load(filename)
        return T

    def __add__(self, other):
        """
        Generate a new TransferMatrix object based on the combination
        of two existing matrices. The distances covered by the two matrices
        should span different ranges, while the other parameters
        such as source and occulter radii must be the same.
        If one of the matrices doesn't have a complete dataset,
        the new matrix is just a copy of the matrix that does
        have existing data. If both are empty it raises a ValueError.

        Ideally the two matrices should be covering adjacent ranges
        in distances. If there is a single value overlap, that value
        is dropped from the larger-distance matrix. E.g., if we add
        a matrix with distances [0 0.1 0.2] and [0.2 0.3 0.4] then
        the values associated with 0.2 in the second matrix are dropped.
        The step sizes do not need to be the same between matrices.

        :param other: TransferMatrix
            A matrix to be added to this matrix.
        :return:
            A new TransferMatrix object with the combined values.
        """

        if not isinstance(other, TransferMatrix):
            raise TypeError(f'Must give a TransferMatrix to __add__(). Was given a {type(other)} instead...')

        if not self.complete and not other.complete:
            raise ValueError("Both operands to addition of matrices are incomplete!")
        elif not self.complete:
            return copy.deepcopy(other)
        elif not other.complete:
            return copy.deepcopy(self)

        # assume both matrices are filled with data
        if not np.array_equal(self.source_radii, other.source_radii):
            raise ValueError('Values for "source_radii" are inconsistent!')
        if not np.array_equal(self.occulter_radii, other.occulter_radii):
            raise ValueError('Values for "occulter_radii" are inconsistent!')

        new = TransferMatrix()

        new.source_radii = self.source_radii
        new.occulter_radii = self.occulter_radii

        # if other has lower distances, add it first
        if np.max(self.distances) > np.max(other.distances):
            self, other = other, self

        # if start and end points are the same, clip the last value
        # to prevent redundancy
        if self.distances[-1] == other.distances[0]:
            end_idx = self.distances.size - 1
        else:
            end_idx = self.distances.size

        new.input_flux = self.input_flux
        new.source_radii = self.source_radii
        new.occulter_radii = self.occulter_radii

        new.distances = np.append(self.distances[:end_idx], other.distances)
        new.flux = np.append(self.flux[:, :, :end_idx], other.flux, axis=2)
        new.magnification = np.append(self.magnification[:, :, :end_idx], other.magnification, axis=2)
        new.moment = np.append(self.moment[:, :, :end_idx], other.moment, axis=2)

        new.min_dist = self.min_dist
        new.max_dist = other.max_dist
        new.step_dist = self.step_dist
        if self.step_dist != other.step_dist:
            new.step_dist = [self.step_dist, other.step_dist]

        new.min_source = self.min_source
        new.max_source = self.max_source
        new.step_source = self.step_source

        new.max_occulter = self.max_occulter
        new.step_occulter = self.step_occulter

        new.num_points = min(self.num_points, other.num_points)
        new.calc_time = max(self.calc_time, other.calc_time)
        new.timestamp = max(self.timestamp, other.timestamp)

        new.complete = True
        new.update_notes()

        return new

    def dwindle_data(self):
        """
        Reduce memory of the saved data by getting rid of
        every other row, column and page of the matrix outputs
        "flux", "input_flux", "magnification" and "moment".
        This will also remove every other value in the
        1D arrays of "distances", "source_radii" and
        "occulter_radii", and double the step sizes.
        """
        N = [int(np.floor(s / 2) * 2) for s in self.flux.shape]
        self.flux = self.flux[:, 0:N[1]:2, :] + self.flux[:, 1::2, :]
        self.flux = self.flux[0:N[0]:2, :, 0:N[2]:2]

        self.input_flux = self.input_flux[:, 0:N[1]:2, :] + self.input_flux[:, 1::2, :]
        self.input_flux = self.input_flux[0:N[0]:2, :, 0:N[2]:2]

        self.moment = self.moment[:, 0:N[1]:2, :] + self.moment[:, 1::2, :]
        self.moment = self.moment[0:N[0]:2, :, 0:N[2]:2]

        # self.magnification = self.magnification[0:N[0]:2, 0:N[1]:2, 0:N[2]:2]
        self.magnification = self.flux / self.input_flux

        self.occulter_radii = self.occulter_radii[0:N[0]:2]
        self.source_radii = (self.source_radii[0:N[1]:2] + self.source_radii[1::2]) / 2
        self.distances = self.distances[0:N[2]:2]

        self.step_dist *= 2
        self.step_source *= 2
        self.occulter_radii *= 2

        self.update_notes()


class StarProfile:
    """
    A class to produce stellar brightness profiles to be ingested by the TransferMatrix.
    Each profile must have an array of values of the same length as the source_radii
    array on the TransferMatrix object that we want to use.
    Each value in the profile represents the relative surface brightness of an annulus
    given at that radius on the star.

    For a uniform profile it just has ones up to the radius of the star and then zeros.
    Other distributions (e.g., using limb darkening) will be added later.
    The flux from each value in the surface_brightness array is multiplied by the amount
    of light from each annulus in the source surface and the image surface.
    The sum of these flux values is used to calculate the magnification.
    The same can be done for flux and 1st moments to get the offset.
    """

    def __init__(self, radii, profile="uniform", source_radius=1, **kwargs):
        self.radii = radii
        self.surface_brightness = np.zeros(radii.size)
        profile = profile.lower()
        if profile == "uniform":
            self.surface_brightness[source_radius >= self.radii] = 1
        else:
            raise KeyError(
                f'Unknown star profile "{profile}". ' 'Use "unform", etc... '
            )

    def get_matrix(self):
        """
        Prepare the surface brightness values as a 3D matrix that can be broadcast
        to the flux values in the TransferMatrix object.
        """
        return np.expand_dims(self.surface_brightness, axis=[0, 2])


def draw_contours(distance, source_radius, points=1e5, plotting=False):
    """
    Draw the contours of a lensed circular disk star,
    given a lens at a certain distance from the star.
    All sizes (distance and source disk radius) are
    given in units of the Einstein radius
    (that means the lens has unit radius).

    Parameters
    ----------
    :param distance: scalar float
        The distance between source and lens centers in units
        of the Einstein radius

    :param source_radius: scalar float
        Size of the source (star) in units of the Einstein radius

    :param points: scalar integer
        How many points should be sampled on the circumference of the source disk

    :param plotting: boolean
        If True, will show a plot of the two contours

    :return z1x:
        The x values of points along the smaller contour

    :return z1y:
        The y values of points along the smaller contour

    :return z2x:
        The x values of points along the larger contour

    :return z2y:
        The y values of points along the larger contour

    Reference
    ---------
    https://iopscience.iop.org/article/10.3847/0004-637X/820/1/53/pdf

    """
    epsilon = np.abs(distance - source_radius)

    if epsilon < 0.3:
        th1 = (
            np.linspace(0, 1 - 2 * epsilon, int(np.ceil(points / 2)), endpoint=False)
            * np.pi
        )
        th1 = np.append(-th1, th1, axis=0)
        th2 = (
            np.geomspace(1 - 2 * epsilon, 1, int(np.ceil(points / 2)), endpoint=False)
            * np.pi
        )
        th2 = np.append(-th2, th2, axis=0)
        th = np.append(th1, th2, axis=0)
        th = np.sort(th)
    else:
        th = np.linspace(-1, 1, int(np.ceil(points / 2)), endpoint=False) * np.pi

    # vector connecting the center of the lens to the point on the perimeter of the source
    ux = distance + source_radius * np.cos(th)
    uy = source_radius * np.sin(th)
    u = np.sqrt(ux ** 2 + uy ** 2)

    # draw the small circle
    factor = 0.5 * (u - np.sqrt(u ** 2 + 4)) / u
    z1x = factor * ux
    z1y = factor * uy

    # draw the big circle
    factor = 0.5 * (u + np.sqrt(u ** 2 + 4)) / u
    z2x = factor * ux
    z2y = factor * uy

    # handle cases very close to zero by
    # adding a contour along one side of the lens
    # for each image, either internally or externally
    if epsilon < 0.001:
        th_right = np.linspace(-1, 1, int(np.ceil(points / 2)), endpoint=False) * np.pi / 2
        flip = 1
        if distance < source_radius:
            flip = -1
        z1x = np.append(z1x, -flip*np.cos(th_right))
        z1y = np.append(z1y, np.sin(th_right))
        z2x = np.append(z2x, flip*np.cos(th_right))
        z2y = np.append(z2y, np.sin(th_right))

    if plotting:
        plt.plot(z1x, z1y, "*", label="small image")
        plt.plot(z2x, z2y, "*", label="large image")

        lx = np.cos(th)
        ly = np.sin(th)
        plt.plot(lx, ly, ":", label="lens")

        plt.plot(ux, uy, "--", label="source")

        plt.gca().set_aspect("equal")
        plt.legend()
        plt.show()

    return z1x, z1y, z2x, z2y


def make_surface(
    distance,
    source_radius,
    z1x,
    z1y,
    z2x,
    z2y,
    pixels=1e6,
    plotting=False,
):

    """
    Take the contours of the two images of the source and plot them on a 2D binary map.
    On this map, True means there is light, and False means no light.
    The total number of True pixels represents the total amount of flux
    (since lensing preserves surface brightness).

    If the two images are seprate, we will get one map with two blobs of light.
    If the two images are overlapping, we will get a map with one blob that has
    a hole inside it (for light reflected away from the observer).

    For each set of contours the function will find the correct resolution
    (i.e., the number of pixels along a line of length of one Einstein radius),
    such that the total image will have a total number of pixels equal to the
    "pixels" input. This makes sure that even for small sources we maintain
    sufficient resolution to avoid aliasing on the pixelated edges of the images.
    For small sources and large distances, the gap between images requires a lot
    of pixels so the resolution will go down, potentially affecting the precision.
    It is best to avoid such combinations or to give bigger values of "pixels".

    The contours are placed on the map and then the area of the images is
    "colored in" line by line. If the contours are too sparsely sampled,
    there will be gaps between lines and the function will raise an exception.

    :param distance: scalar float
        Distance between center of source and lens.
    :param source_radius: scalar float
        Size of the source in units of the Einstein radius.
    :param z1x: float array
        The x-values of the contour of the first image (the small one).
    :param z1y: float array
        The y-values of the contour of the first image (the small one).
    :param z2x: float array
        The x-values of the contour of the second image (the big one).
    :param z2y: float array
        The x-values of the contour of the second image (the big one).
    :param pixels: scalar integer
        The total number of pixels for each image used in the calculation
        (there are three binary images and two float images with this
        number of pixels allocated by this function).
        The function will use this number to determine the resolution
        (pixels per Einstein radius).
        For smaller sources, where the image contours do not cover much area,
        the resolution will be higher than for large sources.
    :param plotting: scalar boolean
        If True, will show the image of the two light blobs. Default is False.

    :return im: binary 2D array
        A 2D binary map of the image(s) of the source after lensing.
        The total number of pixels in this map represents the amount of flux
        from the image(s), that can be compared to the flux from an unlensed source.
    :return x_grid: float 2D array
        A grid of values of the same size as "im" with the x coordinate value saved
        in each pixel. Useful for calculating moments, etc.
    :return y_grid: float 2D array
        A grid of values of the same size as "im" with the y coordinate value saved
        in each pixel. Useful for calculating moments, etc.
    :return resolution: scalar integer
        The number of pixels inside a line of length of the Einstein radius.
        This is important to have in order to compare flux values from
        different calculations: just comparing the number of "light pixels"
        in the image is not enough, as the resolution may cause each pixel
        to represent a different amount of surface area in physical space.

    """
    internal = distance < source_radius  # is the lens inside the disk of the source

    # extent of the required image, in Einstein units
    left = min(np.min(z1x), np.min(z2x))
    right = max(np.max(z1x), np.max(z2x))
    top = max(np.max(z1y), np.max(z2y))

    # print(f'left= {left}, right= {right}, top= {top}')

    total_area = top * (right - left)
    resolution = int(np.floor(np.sqrt(pixels/total_area)))

    # extent of the image in pixels
    height = int(np.ceil(top * resolution) + 1)
    width = int(np.ceil((right - left) * resolution) + 1)
    origin_idx = int(np.ceil(-left * resolution))

    # print(f'height= {height}, width= {width}, origin= {origin_idx}')

    x_axis = np.linspace(left, right, width, endpoint=True)
    y_axis = np.linspace(0, top, height, endpoint=True)

    im1 = np.zeros((height, width), dtype=bool)
    im2 = np.zeros((height, width), dtype=bool)

    # print(f'time to allocate: {timer() - t0:.3f}s'); t0 = timer()

    z1x_ind = np.round((z1x - left) * resolution).astype(int)
    z1y_ind = np.round(z1y * resolution).astype(int)
    good_indices = (z1x_ind >= 0) & (z1x_ind < width) & (z1y_ind >= 0) & (z1y_ind < height)
    z1x_ind = z1x_ind[good_indices]
    z1y_ind = z1y_ind[good_indices]

    im1[z1y_ind, z1x_ind] = True

    z2x_ind = np.round((z2x - left) * resolution).astype(int)
    z2y_ind = np.round(z2y * resolution).astype(int)
    good_indices = (z2x_ind >= 0) & (z2x_ind < width) & (z2y_ind >= 0) & (z2y_ind < height)
    z2x_ind = z2x_ind[good_indices]
    z2y_ind = z2y_ind[good_indices]

    im2[z2y_ind, z2x_ind] = True

    # print(f'time to loop over contours: {timer()-t0:.3f}s'); t0 = timer()

    for i in range(im1.shape[0]):
        indices = np.nonzero(im1[i])[0]  # where the contours are non-zero
        if indices.size:
            mx = np.max(indices)
            mn = np.min(indices)
            im1[i, mn:mx] = True

    for i in range(im2.shape[0]):
        indices = np.nonzero(im2[i])[0]  # where the contours are non-zero
        if indices.size:
            mx = np.max(indices)
            mn = np.min(indices)
            im2[i, mn:mx] = True

    if np.sum(np.diff(im2[:, 0])) > 2:
        raise RuntimeError(f'Number of transitions ({np.sum(np.diff(im2[:, 0])) }) is too high. '
                           f'Increase number of points on the circle. '
                           f'distance= {distance}, source radius= {source_radius}')

    # print(f'time to loop over height of images: {timer() - t0:.3f}s')

    # t0 = timer()
    #
    # im1 = ndimage.binary_closing(im1, border_value=0, iterations=1)
    # im2 = ndimage.binary_closing(im2, border_value=0, iterations=1)
    #
    # print(f'time to dilate/erode: {timer() - t0:.3f}s')

    # t0 = timer()

    if internal:
        im = np.bitwise_xor(im1, im2)
    else:
        im = np.bitwise_or(im1, im2)

    # print(f'time to subtract/add images: {timer() - t0:.3f}s')

    if plotting:
        plt.figure()
        ex = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
        plt.imshow(im, extent=ex)
        plt.show()

    x_grid = np.repeat(np.expand_dims(x_axis, 0), y_axis.size, axis=0)
    y_grid = np.repeat(np.expand_dims(y_axis, 1), x_axis.size, axis=1)

    return im, x_grid, y_grid, resolution


def remove_occulter(im, occulter_radius, r_squared_grid, plotting=False, return_changed_flag=False):
    """
    Find areas of the image of the source that are supposed to be occulted
    by the physical size of the lens, and replace any True values in that image
    with False. This represents the occultation by the lens.

    If the new occulter size does not remove any new pixels from "im",
    it just returns the same matrix. If any of the pixels are affected,
    the change is done in-place and the modified image is returned.

    The reason we need to discern between the two cases where the image
    is changed or not, is that if the image is not changed, there is no
    need to recalculate the sum of pixels in the image, which is expensive.
    If we are iterating over occulter sizes, often for many steps the occulter
    is too small to touch any of the images, so that could save us many computation.

    :param im: binary 2D array
        The boolean map used to denote where there is light from the two images of the source.
    :param occulter_radius: scalar float
        The physical size of the lensing object, in units of Einstein radius.
    :param r_squared_grid: float 2D array
        An image of the same size as "im" where each pixel value contains
        the x and y coordinates, squared and added.
        The values represent the square of the distance between the center of the lens
        and each point on the binary image.
    :param plotting: boolean scalar
        If True, will show the image. Default is False.
    :param return_changed_flag: boolean scalar
        If True, will return the image "im" and also a "changed_flag" boolean scalar, as a tuple.
        This is useful if you also want to know if the image has changed since the last calculation.
        This saves having to re-sum the image even though it did not change,
        e.g., when iterating over slowly increasing occulter radius.
    :return:
        Either just the "im" array (modified in place) or a tuple with (im, changed_flag).
        If changed_flag is True, that means there was at least one pixel changed in "im".
    """

    idx = r_squared_grid[0, :] < occulter_radius ** 2
    changed_flag = False

    if np.any(im[0, idx]):
        idx = r_squared_grid < occulter_radius ** 2
        im[idx] = 0
        changed_flag = True

    if plotting:
        plt.imshow(im)
        plt.show()

    if return_changed_flag:
        return im, changed_flag
    else:
        return im


def single_geometry(
    source_radius,
    distance,
    occulter_radius=0,
    circle_points=1e5,
    pixels=1e6,
    get_offsets=False,
    plotting=False,
):
    """
    Directly calculate the magnification from a specific micro-lensing alignment.
    The function uses the same internal calculations that are used to make a
    TransferMatrix object, but instead of pre-calculating and figuring out
    how the magnification looks byu interpolating and wighing with a star profile,
    this function just directly generates contours and surface maps and finds
    the magnification from that image.
    Utilizes draw_contours(), make_surface() and remove_occulter() in sequence.

    :param source_radius: scalar float
        The size of the source that is being lensed (in units of the Einstein radius).
    :param distance: scalar float
        The distance between the center of the lens and the center of the source (Einstein units).
    :param occulter_radius: scalar float
        The physical size of the lensing object, that can hide some of the light of the image
         (Einstein units). Default is zero (no occultation, e.g., a black hole lens).
    :param circle_points: scalar integer
        The number of points on the circumference of the source, used to make a 2D contour of the images.
        For very large sources this should be increased from the default 1e5 to 1e6 or more.
    :param pixels: scalar integer
        The total number of pixels in the generated image. The higher this number,
        the higher the resolution of the image. Default is 1e7.
    :param get_offsets: boolean scalar
        If True, will also calculate the offset of the image's center of light
        from the true source position (in Einstein units). Default is False.
    :param plotting: scalar boolean
        If True, will plot the image and show the positions of the source and lens.
        This uses the plot_geometry() function. Default is False.

    :return:
        Either return the magnification for this geometry as a scalar float,
        or return a tuple with the magnification and centroid offset,
        if requested by setting get_offsets=True.
        The offset is given in units of the Einstein radius.
    """
    # t0 = timer()
    (z1x, z1y, z2x, z2y) = draw_contours(distance, source_radius, points=circle_points)
    # print(f'time to draw contours: {timer() - t0:.3f}s')

    (im, x_grid, y_grid, resolution) = make_surface(distance, source_radius, z1x, z1y, z2x, z2y, pixels=pixels)

    if occulter_radius > 0:
        im2 = remove_occulter(im, occulter_radius, x_grid ** 2 + y_grid ** 2)
    else:
        im2 = im

    mag = np.sum(im2) * 2
    offset = np.sum(im2 * (x_grid - distance)) * 2 / mag
    mag /= np.pi * source_radius ** 2 * resolution ** 2

    if plotting:
        plot_geometry(im2, x_grid, y_grid, source_radius, distance, occulter_radius, mag, offset)

    if get_offsets:
        return mag, offset
    else:
        return mag


def radial_lightcurve(
    source_radius,
    distances=None,
    occulter_radius=0,
    circle_points=1e5,
    pixels=1e6,
    get_offsets=False,
    verbosity=0,
    plotting=False,
):
    """
    Produce a lightcurve (measuring the relative magnification) for each radial distance
    of the lens from the source. Calls single_geometry() iteratively to do this.


    :param source_radius: scalar float
        The size of the source that is being lensed (in units of the Einstein radius).
    :param distances: list or array of floats
        The set of distances used to calculate the magnifications (in Einstein units).
    :param occulter_radius: scalar float
        The physical size of the lensing object, that can hide some of the light of the image
         (Einstein units). Default is zero (no occultation, e.g., a black hole lens).
    :param circle_points: scalar integer
        The number of points on the circumference of the source, used to make a 2D contour of the images.
        For very large sources this should be increased from the default 1e5 to 1e6 or more.
    :param pixels: scalar integer
        The total number of pixels in the generated image. The higher this number,
        the higher the resolution of the image. Default is 1e7.
    :param get_offsets: boolean scalar
        If True, will also calculate the offset of each image's center of light
        from the true source position (in Einstein units). Default is False.
    :param verbosity: scalar integer
        If non-zero, will output more information to terminal while doing the calculations.
        Default is zero (quiet execution).
    :param plotting: scalar boolean
        If True, will plot each image for each geometry calculated.
        This uses the plot_geometry() function. Default is False.

    :return:
        Either returns one array of the same length as "distances"
        or a tuple with two such arrays. The first will have the
        magnification values and the optional second array will
        have the center of light offset from the real source position.
        To get both array, input get_offsets=True.
    """
    if distances is None:
        distances = np.linspace(0, 10, 100, endpoint=False)

    mag = np.ones(distances.shape)
    offset = np.ones(distances.shape)

    for i, d in enumerate(distances):

        (mag[i], offset[i]) = single_geometry(
            source_radius=source_radius,
            distance=d,
            occulter_radius=occulter_radius,
            circle_points=circle_points,
            pixels=pixels,
            get_offsets=True,
            plotting=plotting,
        )

        if verbosity > 0:
            print(
                f"i= {i}/{distances.size}: d= {d:.2f} "
                f"| mag= {mag[i]:.3f} | offset= {offset[i]:.3f}"
            )

    if get_offsets:
        return mag, offset
    else:
        return mag


def plot_geometry(
        im,
        x_grid,
        y_grid,
        source_radius,
        distance,
        occulter_radius,
        mag,
        offset,
        num_points=1e5,
        pause_time=1e-3
):
    """
    Show the results of calculations of a single geometry of source and lens.
    Will display the image(s) of the source on a 2D map, where the True values
    represent the area where light arrives from the source.
    Will also show the center of the source, the center of light of the lensed image,
    and the circumferences of the true, unlensed source and that of the lens
    (i.e., the Einstein ring).

    :param im: boolean 2D array
        The 2D binary image of the light areas of the lens image.
    :param x_grid: float 2D array
        The same size as "im", where each pixel value holds the x-coordinate of that pixel.
    :param y_grid: float 2D array
        The same size as "im", where each pixel value holds the x-coordinate of that pixel.
    :param source_radius: scalar float
        The source size in units of the Einstein radius.
    :param distance: scalar float
        The distance between source and lens centers, in Einstein units.
    :param occulter_radius: scalar float
        The physical size of the lens, that can occult some of the light, in Einstein units.
        If zero (the default), no part of the image is occulted (as is the case for, e.g., black holes).
    :param mag: scalar float
        The calculated magnification of this geometry. Will be printed on the axis label.
    :param offset: scalar float
        The calculated center of light offset of this geometry. Will be printed on the axis label.
    :param num_points: scalar integer
        The number of points on the circumferences of the lens and source,
        that are used to plot them as dotted lines on top of the image.
        Default is 1e5.
    :param pause_time: scalar float
        The number of seconds of pause that is added after plotting,
        in case we want to loop over many geometries and stop for a short
        while to look at the plot for each one. Default is 1e-3 seconds.

    """
    num_points = int(num_points)
    im_left = np.min(x_grid)
    im_right = np.max(x_grid)
    im_bottom = -np.max(y_grid)
    im_top = np.max(y_grid)
    plt.ion()
    plt.cla()
    plt.imshow(
        np.concatenate((np.flip(im, axis=0), im), axis=0),
        vmin=0,
        vmax=1,
        extent=(
            im_left,
            im_right,
            im_bottom,
            im_top,
        ),
        origin="lower",
    )

    x = np.cos(np.linspace(0, 2 * np.pi, num_points))
    y = np.sin(np.linspace(0, 2 * np.pi, num_points))
    out_of_bounds = (im_left > x) | (x > im_right) | (im_bottom > y) | (y > im_top)
    x[out_of_bounds] = np.NAN
    y[out_of_bounds] = np.NAN

    plt.plot(x, y, ":r", label="lens")

    x2 = source_radius * np.cos(np.linspace(0, 2 * np.pi, num_points)) + distance
    y2 = source_radius * np.sin(np.linspace(0, 2 * np.pi, num_points))

    out_of_bounds = (im_left > x2) | (x2 > im_right) | (im_bottom > y2) | (y2 > im_top)
    x2[out_of_bounds] = np.NAN
    y2[out_of_bounds] = np.NAN

    plt.plot(x2, y2, ":g", label="source radius")

    if occulter_radius > 0:
        x3 = occulter_radius * np.cos(np.linspace(0, 2 * np.pi, num_points))
        y3 = occulter_radius * np.sin(np.linspace(0, 2 * np.pi, num_points))

        out_of_bounds = (im_left > x3) | (x3 > im_right) | (im_bottom > y3) | (y3 > im_top)
        x3[out_of_bounds] = np.NAN
        y3[out_of_bounds] = np.NAN

        plt.plot(x3, y3, ":m", label="occulter")

    plt.plot(distance, 0, "go", label="source center")
    plt.plot(distance + offset, 0, "r+", label="center of light")

    plt.xlabel(
        f"d= {distance:.2f} | source r= {source_radius} | occulter r= {occulter_radius} | "
        f"mag= {mag:.2f} | offset= {offset:.2f}"
    )
    plt.gca().set_aspect("equal")
    plt.legend(bbox_to_anchor=(0.0, 1.04), loc="lower left")
    plt.show()
    plt.pause(pause_time)


def point_source_approximation(distances):
    """
    Calculate the magnification for a point source, given some distances.

    :param distances: scalar or array
        The distances between the lens and the source, in units of the Einstein radius.
    :return:
        The magnification using the point-source approximation.
        The output is the same size as the "distances" input. 
    """
    sq = np.sqrt(1 + 4 / distances ** 2)
    return 0.5 * (sq + 1 / sq)


if __name__ == "__main__":

    T = TransferMatrix.from_file('saved/matrix_SR0.100-1.000_D0.000-10.000.npz')
    d = np.linspace(0, 30, 300, endpoint=True)
    mag = T.radial_lightcurve(source=0.01, distances=d)
    plt.plot(d, mag, '*')
    plt.plot(d, point_source_approximation(d), '-')
    # single_geometry(occulter_radius=0.6, source_radius=1, plotting=True, distance=0.5, circle_points=1e5)

    # T = TransferMatrix()
    # T.min_source = 0.1
    # T.max_source = 2.0
    # T.step_source = 0.1
    # T.max_dist = 6
    # T.step_dist = 0.1
    # T.max_occulter = 3
    # T.step_occulter = 0.1
    # T.make_matrix()
    # T.save()

    # distance = 1.3
    # source_radius = 2.0
    # occulter_radius = 0.4
    #
    # t1 = timer()
    # (z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution) = draw_contours(distance, source_radius)
    # t2 = timer()
    # print(f"Draw contours: elapsed time is {t2 - t1}s. ")
    #
    # t1 = timer()
    # (im, x_grid, y_grid) = make_surface(z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution)
    # t2 = timer()
    # print(f"Make surface: elapsed time is {t2 - t1}s. ")
    #
    # t1 = timer()
    # im2 = remove_occulter(im, occulter_radius, x_grid, y_grid)
    # t2 = timer()
    # print(f"Remove occulter: elapsed time is {t2 - t1}s. ")
