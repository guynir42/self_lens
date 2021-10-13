import numpy as np
from scipy import ndimage
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
    and centroids offset for all combinations of source and occulter size,
    for all distances, given the ranges (min/max values) given.

    Then the existing matrix is saved/loaded from disk, and
    used to make quick lightcurve estimates.
    Any values between the given source/occulter radii or distances
    are interpolated linearly, which is reasonable if the step sizes
    are not too big and the functions don't change too rapidly.

    The name TransferMatrix refers to the transformation of the
    light from each annulus on the source into the light seen
    after it was lensed (i.e., magnified). The

    The data saved in the output arrays of the TransferMatrix
    is arranged into four arrays: "flux", "input_flux", "magnification"
    and "moment". The input_flux is 1D and the rest are 3D, with
    axis 0 for occulter radii, axis 1 for source radii, and
    axis 2 for distances (between source and lens centers).
    The "input_flux" dimension is only for the source radii,
    and it specifies the surface area of each annulus (between
    r_i and r_i+1) in units of number of pixels.
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
    the different annuli in the source surface.
    The moments can be summed over the different radii,
    and the "flux" array can be summed as well.
    Then their ratio gives the normalized offset
    of the center of light from the center of the source.

    USAGE: choose min/max/step for distances, source radii,
           and occulter radii. Call make_matrix() and wait.
           Once done, use save() to keep the matrix
           for later use. Retreive the matrix from file
           using the load() function.
           Once ready, use radial_lightcurve() to produce
           a set of magnifications and offsets for a list
           of distances, interpolated from the existing data.


    All units are normalized to the Einstein radius.
    """

    def __init__(self):
        self.resolution = 1e2
        self.num_points = 1e5

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

        self.notes = f'distances= {self.min_dist}-{self.max_dist} ({self.step_dist} step) | ' \
                     f'source radii= {self.min_source}-{self.max_source} ({self.step_source} step) | ' \
                     f'occulter radii= 0-{self.max_occulter} ({self.step_occulter} step) ' \
                     f'| resolution= {self.resolution} | num points= {self.num_points}'

        self.flux = np.zeros(
            (self.occulter_radii.size, self.source_radii.size, self.distances.size),
            dtype=np.single,
        )
        self.input_flux = np.zeros(self.source_radii.size, dtype=np.single)
        self.moment = np.zeros(
            (self.occulter_radii.size, self.source_radii.size, self.distances.size),
            dtype=np.single,
        )

        if plotting:
            plt.ion()

        t0 = timer()

        for i, d in enumerate(self.distances):

            for j, source_radius in enumerate(self.source_radii):

                (z1x, z1y, z2x, z2y) = draw_contours(
                    d,
                    source_radius,
                    points=self.num_points,

                )
                (im, x_grid, y_grid) = make_surface(
                    d,
                    source_radius,
                    z1x,
                    z1y,
                    z2x,
                    z2y,
                    resolution=self.resolution,
                )

                for k, occulter_radius in enumerate(self.occulter_radii):
                    im2 = remove_occulter(im, occulter_radius, x_grid, y_grid)
                    self.input_flux[j] = (
                        np.pi * source_radius ** 2 * self.resolution ** 2
                    )
                    self.flux[k, j, i] = np.sum(im2)
                    self.moment[k, j, i] = np.sum(im2 * (x_grid - d))

                    if plotting:
                        offset = self.moment[k, j, i] / self.flux[k, j, i]
                        plt.cla()
                        plt.imshow(
                            im2,
                            vmin=0,
                            vmax=1,
                            extent=(
                                np.min(x_grid),
                                np.max(x_grid),
                                np.min(y_grid),
                                np.max(y_grid),
                            ),
                        )
                        x = np.cos(np.linspace(0, 2 * np.pi))
                        y = np.sin(np.linspace(0, 2 * np.pi))

                        plt.plot(x, y, "--r", label="lens")

                        x2 = source_radius * np.cos(np.linspace(0, 2 * np.pi)) + d
                        y2 = source_radius * np.sin(np.linspace(0, 2 * np.pi))
                        plt.plot(x2, y2, ":b", label="source radius")

                        if occulter_radius > 0:
                            x3 = occulter_radius * np.cos(np.linspace(0, 2 * np.pi))
                            y3 = occulter_radius * np.sin(np.linspace(0, 2 * np.pi))
                            plt.plot(x3, y3, ":g", label="occulter")

                        plt.plot(d, 0, "go", label="source center")
                        plt.plot(d + offset, 0, "r+", label="center of light")

                        plt.xlabel(
                            f"d= {d:.2f} | source r= {source_radius} | occulter r= {occulter_radius} "
                            f"| mag= {self.flux[k, j, i] / self.input_flux[j]:.2f} | moment= {offset:.2f}"
                        )
                        plt.gca().set_aspect("equal")
                        plt.legend()
                        plt.show()
                        plt.pause(0.0001)

            t1 = timer()
            self.calc_time = t1 - t0

            print(
                f"{i+1:3d}/{self.distances.size} | Distance= {d} | runtime = {t1-t0:.1f}s"
            )

        self.flux = np.diff(self.flux, axis=1, prepend=0)
        self.input_flux = np.diff(self.input_flux, axis=0, prepend=0)
        self.magnification = self.flux / np.expand_dims(self.input_flux, axis=[0, 2])

        self.moment = np.diff(self.moment, axis=1, prepend=0)

        self.complete = True
        self.timestamp = str(datetime.utcnow())

    def radial_lightcurve(
        self, source=3, distances=None, occulter_radius=0, get_offsets=False
    ):
        """
        :param source:
            Give the size of the source,
            or a brightness profile (array or StarProfile object)
            that specifies the surface brightness of each annulus.
            Such an array must match in size the matrix's source_radii array,
            and each value of surface brightness in the profile must correspond
            to that radius in source_radii.
        :param distances:
            A list of distances (between lens and source centers)
            where the results should be calculated.
            If None (default) will return the values at the matrix's distances array.
            Otherwise, the values are linearly interpolated into the distances
            that were calculated by the matrix. None of the distances can exceed
            the maximum of the matrix's distances array
        :param occulter_radius:
            Provide the radius of the physical extent of the lensing object.
            For very tiny objects (e.g., black holes) this can be effectively zero (default).
            Otherwise, it the results are interpolated from the results calculated
            using the matrix's occulter_radii array.
            The value requested cannot exceed the maximum of that array.
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

        # figure out the stellar light profile
        N = self.flux.shape[1]  # number of source radii
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
            star_profile = np.zeros((1, N, 1))
            star_profile[0, source >= self.source_radii, 0] = 1
        else:
            raise TypeError(
                'Must provide "source" input as a numeric scalar, '
                f"a list, or an array (of size {N}). "
            )

        # multiply each profile's brightness
        # by the surface area of the annulus
        dr = np.diff(self.source_radii)
        dr = np.append(dr, dr[-1])
        # area = 2 * np.pi * self.source_radii * dr
        area = np.pi * self.source_radii ** 2
        area = np.diff(area, prepend=0)  # get the annulus area
        area = np.expand_dims(area, axis=[0, 2])
        star_profile *= area  # the flux from each annulus scaled to the area

        star_profile /= np.sum(star_profile)  # normalize to unity

        if occulter_radius > np.max(self.occulter_radii):
            raise ValueError(
                f"requested occulter radius {occulter_radius} is out of range "
                f"for this matrix ({np.min(self.occulter_radii)}-{np.max(self.occulter_radii)})"
            )

        occulter_idx = occulter_radius == self.occulter_radii

        if np.any(occulter_idx):
            mag = self.magnification[occulter_idx, :, :]
            mag = np.sum(mag * star_profile, axis=1)  # apply the source profile
            mag = mag[0, :]
        else:
            # index of occulter radius below requested value
            idx = np.argmax(self.occulter_radii < occulter_radius)

            x1 = self.occulter_radii[idx]
            x2 = self.occulter_radii[idx + 1]
            y1 = self.magnification[idx]
            y2 = self.magnification[idx + 1]

            # interpolate between the occulter radii above and below
            mag = y1 + ((occulter_radius - x1) / (x2 - x1)) * (y2 - y1)
            mag = np.sum(mag * star_profile, axis=1)  # apply the source profile
            mag = mag[0, :]

            if distances is not None:
                mag = np.interp(d, self.distances, mag, right=1)

        if not get_offsets:
            return mag
        else:

            if np.any(occulter_idx):
                moments = self.moment[occulter_idx, :, :]
                # apply the source size / profile
                moments = np.sum(moments * star_profile, axis=1)

                flux = self.flux[occulter_idx, :, :]
                # apply the source size / profile
                flux = np.sum(flux * star_profile, axis=1)

            else:
                y1 = self.moment[idx]
                y2 = self.moment[idx + 1]
                moments = y1 + ((occulter_radius - x1) / (x2 - x1)) * (y2 - y1)

                # apply the source profile
                moments = np.sum(moments * star_profile, axis=1)

                y1 = self.flux[idx]
                y2 = self.flux[idx + 1]
                flux = y1 + ((occulter_radius - x1) / (x2 - x1)) * (y2 - y1)

                # apply the source profile
                flux = np.sum(flux * star_profile, axis=1)

            offsets = moments / flux

            if distances is not None:
                offsets = np.interp(d, self.distances, offsets, right=0)
            return mag, offsets

    def save(self, filename="matrix"):
        np.savez_compressed(filename, **vars(self))
        with open(f'{filename}.txt', 'w') as text_file:
            text_file.write(self.notes)

    def load(self, filename):
        A = np.load(filename)
        for k in vars(self).keys():
            setattr(self, k, A[k])

    def __add__(self, other):
        pass  # TODO: need to finish this


class StarProfile:
    def __init__(self, radii, profile="uniform", source_radius=1, **kwargs):
        self.radii = radii
        self.values = np.zeros(radii.size)
        profile = profile.lower()
        if profile == "uniform":
            self.values[source_radius >= self.radii] = 1
        else:
            raise KeyError(
                f'Unknown star profile "{profile}". ' 'Use "unform", etc... '
            )

        self.values /= np.sum(self.values)  # total light must be unity

    def get_matrix(self):
        return np.expand_dims(self.values, axis=[0, 2])


def draw_contours(distance, source_radius, points=1e4, plotting=False):
    """
    Draw the contours of a lensed circular disk star,
    given a lens at a certain distance from the star.
    All sizes (distance and source disk radius) are
    given in units of the Einstein radius
    (that means the lens has unit radius).

    Parameters
    ----------
    :param distance:
        the distance between source and lens centers in units
        of the Einstein radius

    :param source_radius:
        size of the source (star) in units of the Einstein radius

    :param points: scalar integer
        how many points should be sampled on the circumference of the source disk

    :param plotting: boolean
        if True, will show a plot of the two contours

    :return z1x:
        the x values of points along the smaller contour

    :return z1y:
        the y values of points along the smaller contour

    :return z2x:
        the x values of points along the larger contour

    :return z2y:
        the y values of points along the larger contour

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
    resolution,
    plotting=False,
):

    internal = distance < source_radius

    max_radius = max(1, source_radius)
    # x axis spans the range from -(2*max_radius) to (distance + 4*max_radius)
    width = int(np.ceil((6 * max_radius + distance) * resolution))
    x_axis = np.linspace(
        -2 * max_radius, 4 * max_radius + distance, width, endpoint=False
    )

    # y axis spans max_radius on either side
    height = int(np.ceil(4 * max_radius * resolution))
    y_axis = np.linspace(-2 * max_radius, 2 * max_radius, height, endpoint=False)

    im1 = np.zeros((height, width), dtype=bool)
    im2 = np.zeros((height, width), dtype=bool)

    for i in range(z1x.size):

        z1x_ind = int(np.round((z1x[i] - x_axis[0]) * resolution))
        z1y_ind = int(np.round((z1y[i] - y_axis[0]) * resolution))
        if 0 <= z1x_ind < width and 0 <= z1y_ind < height:
            im1[z1y_ind, z1x_ind] = True

        z2x_ind = int(np.round((z2x[i] - x_axis[0]) * resolution))
        z2y_ind = int(np.round((z2y[i] - y_axis[0]) * resolution))
        if 0 <= z2x_ind < width and 0 <= z2y_ind < height:
            im2[z2y_ind, z2x_ind] = True

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

    im1 = ndimage.binary_closing(im1, border_value=0, iterations=2)
    im2 = ndimage.binary_closing(im2, border_value=0, iterations=2)

    if internal:
        im = np.bitwise_xor(im1, im2)
    else:
        im = np.bitwise_or(im1, im2)

    if plotting:
        plt.figure()
        ex = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
        plt.imshow(im, extent=ex)
        plt.show()

    x_grid = np.repeat(np.expand_dims(x_axis, 0), y_axis.size, axis=0)
    y_grid = np.repeat(np.expand_dims(y_axis, 1), x_axis.size, axis=1)

    return im, x_grid, y_grid


def remove_occulter(im, occulter_radius, x_grid, y_grid, plotting=False):
    idx = (x_grid ** 2 + y_grid ** 2) < occulter_radius ** 2

    im_new = im
    im_new[idx] = 0

    if plotting:
        plt.imshow(im_new)
        plt.show()

    return im_new


def single_geometry(
    source_radius,
    distance,
    occulter_radius=0,
    circle_points=1e4,
    resolution=1e2,
    get_offsets=False,
    verbosity=0,
    plotting=False,
):

    (z1x, z1y, z2x, z2y) = draw_contours(distance, source_radius, points=circle_points)

    (im, x_grid, y_grid) = make_surface(distance, source_radius, z1x, z1y, z2x, z2y, resolution)

    if occulter_radius > 0:
        im2 = remove_occulter(im, occulter_radius, x_grid, y_grid)
    else:
        im2 = im

    mag = np.sum(im2)
    offset = np.sum(im2 * (x_grid - distance)) / mag
    mag /= np.pi * source_radius ** 2 * resolution ** 2

    if plotting:
        plt.ion()
        plt.cla()
        plt.imshow(
            im2,
            vmin=0,
            vmax=1,
            extent=(
                np.min(x_grid),
                np.max(x_grid),
                np.min(y_grid),
                np.max(y_grid),
            ),
        )

        x = np.cos(np.linspace(0, 2 * np.pi))
        y = np.sin(np.linspace(0, 2 * np.pi))

        plt.plot(x, y, "--r", label="lens")

        x2 = source_radius * np.cos(np.linspace(0, 2 * np.pi)) + distance
        y2 = source_radius * np.sin(np.linspace(0, 2 * np.pi))
        plt.plot(x2, y2, ":g", label="source radius")

        if occulter_radius > 0:
            x3 = occulter_radius * np.cos(np.linspace(0, 2 * np.pi))
            y3 = occulter_radius * np.sin(np.linspace(0, 2 * np.pi))
            plt.plot(x3, y3, ":g", label="occulter")

        plt.plot(distance, 0, "go", label="source center")
        plt.plot(distance + offset, 0, "r+", label="center of light")

        plt.xlabel(
            f"d= {distance:.2f} | source r= {source_radius} | "
            f"occulter r= {occulter_radius} | mag= {mag:.2f} | "
            f"offset= {offset:.2f}"
        )
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.show()
        plt.pause(0.0001)

    if get_offsets:
        return mag, offset
    else:
        return mag


def radial_lightcurve(
    source_radius,
    distances=None,
    occulter_radius=0,
    circle_points=1e4,
    resoution=1e2,
    get_offsets=False,
    verbosity=0,
    plotting=False,
):

    if distances is None:
        distances = np.linspace(0, 10, 100, endpoint=False)

    mag = np.ones(distances.shape)
    offset = np.ones(distances.shape)

    for i, d in enumerate(distances):

        (mag[i], offset[i]) = single_geometry(
            source_radius,
            d,
            occulter_radius,
            circle_points,
            resoution,
            True,
            verbosity,
            plotting,
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


if __name__ == "__main__":

    T = TransferMatrix()

    T.max_source = 2
    T.max_dist = 3
    T.max_occulter = 1

    # T.load("matrix.npz")

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
