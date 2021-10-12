import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer


class TransferMatrix:
    def __init__(self):
        self.resolution = 1e2
        self.num_points = 1e5

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

    def make_matrix(self, plotting=False):
        print("calculating matrix! ")
        self.distances = np.round(np.linspace(
            0, self.max_dist, int(np.round(self.max_dist / self.step_dist) + 1), endpoint=True
        ), 5)
        self.source_radii = np.round(np.linspace(
            self.min_source,
            self.max_source,
            int(np.round((self.max_source - self.min_source) / self.step_source) + 1),
            endpoint=True,
        ), 5)

        self.occulter_radii = np.round(np.linspace(
            0,
            self.max_occulter,
            int(np.round(self.max_occulter / self.step_occulter) + 1),
            endpoint=True,
        ), 5)

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

                # when they are the same (caustic crossing), use the previously calculated contour
                if d != source_radius:
                    (
                        z1x,
                        z1y,
                        z2x,
                        z2y,
                        height,
                        width,
                        x_axis,
                        y_axis,
                        internal,
                        resolution,
                    ) = draw_contours(
                        d,
                        source_radius,
                        points=self.num_points,
                        resolution=self.resolution,
                    )
                (im, x_grid, y_grid) = make_surface(
                    z1x,
                    z1y,
                    z2x,
                    z2y,
                    height,
                    width,
                    x_axis,
                    y_axis,
                    internal,
                    resolution,
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

                    # keep only the values from the annulus, not the entire disk
                    if j > 0:
                        self.input_flux[j] -= self.input_flux[j - 1]
                        self.flux[k, j, i] -= self.flux[k, j - 1, i]
                        self.moment[k, j, i] -= self.moment[k, j - 1, i]

            t1 = timer()
            print(
                f"{i+1:3d}/{self.distances.size} | Distance= {d} | runtime = {t1-t0:.1f}s"
            )

        self.magnification = self.flux / np.expand_dims(self.input_flux, axis=[0, 2])

    def radial_lightcurve(
        self, source=3, distances=None, occulter_radius=0, get_moments=False
    ):
        if distances is None:
            d = self.distances
        else:
            d = distances

        # figure out the stellar light profile
        N = self.flux.shape[0]  # number of occulter radii
        if isinstance(source, StarProfile):
            star_profile = source.get_matrix()
        elif type(source) is np.ndarray:
            if source.ndim == 1 and source.shape[0] == N:
                star_profile = np.expand_dims(np.expand_dims(source, axis=0), axis=2)
            elif source.ndim == 2 and source.shape[1] == N:
                star_profile = np.expand_dims(source, axis=2)
            elif source.ndim == 3 and source.shape[1] == N:
                star_profile = source
        elif hasattr(source, "__len__") and len(source) == N:
            star_profile = np.expand_dims(np.expand_dims(np.array(source), axis=0), axis=2)
        elif np.isscalar(source):
            star_profile = np.zeros((1, N, 1))
            star_profile[0, source >= self.occulter_radii, 0] = 1
        else:
            raise TypeError('Must provide "source" input as a numeric scalar, '
                            f'a list, or an array (of size {N}). ')

        star_profile /= np.sum(star_profile)  # normalize to unity

        if occulter_radius > np.max(self.occulter_radii):
            raise ValueError(
                f"requested occulter radius {occulter_radius} is out of range "
                f"for this matrix ({np.min(self.occulter_radii)}-{np.max(self.occulter_radii)}"
            )
        if occulter_radius == np.max(self.occulter_radii):
            mag = self.magnification[-1, :, :]
            mag = np.sum(mag * star_profile, axis=1)  # apply the source size / profile

        else:
            # index of occulter radius below requested value
            idx = np.argmax(self.occulter_radii < occulter_radius)

            x1 = self.occulter_radii[idx]
            x2 = self.occulter_radii[idx + 1]
            y1 = self.magnification[idx]
            y2 = self.magnification[idx + 1]

            # interpolate between the occulter radii above and below
            mag = y1 + ((occulter_radius - x1) / (x2 - x1)) * (y2 - y1)
            mag = np.sum(mag * star_profile, axis=1)  # apply the source size / profile

            if distances is not None:
                mag = np.interp(d, self.distances, mag, right=1)

        if not get_moments:
            return mag
        else:

            if occulter_radius == np.max(self.occulter_radii):
                moments = self.moment[-1, :, :]
                moments = np.sum(moments * star_profile, axis=1)  # apply the source size / profile

            else:
                y1 = self.moment[idx]
                y2 = self.moment[idx + 1]
                moments = y1 + ((occulter_radius - x1) / (x2 - x1)) * (y2 - y1)
                moments = np.sum(moments * star_profile, axis=1)  # apply the source size / profile

            if distances is not None:
                moments = np.interp(d, self.distances, moments, right=0)
            return mag, moments

    def save(self, filename="transfer_matrix"):
        np.savez_compressed(filename, **vars(self))

    def load(self, filename):
        A = np.load(filename)
        for k in vars(self).keys():
            setattr(self, k, A[k])

    def __add__(self, other):
        pass  # need to finish this


class StarProfile:
    def __init__(self):
        pass

def draw_contours(distance, source_radius, points=1e4, resolution=1e2, plotting=False):
    """
    Draw the contours of a lensed circular disk star,
    given a lens at a certain distance from the star.
    All sizes (distance and source disk radius) are
    given in units of the Einstein radius
    (that means the lens has unit radius).

    Parameters
    ----------
    * distance:

    * source_radius:

    * points: scalar integer
        how many points should be sampled on the circumference of the source disk

    * resolution: scalar integer
        how many points in the grid should represent one unit (=Einstein radius)

    * plotting: boolean
        if True, will show a plot of the two contours

    Reference
    ---------
    https://iopscience.iop.org/article/10.3847/0004-637X/820/1/53/pdf

    """
    epsilon = np.abs(distance - source_radius)
    if epsilon == 0:
        raise ValueError(
            "Cannot find the contours for exact match distance==source_radius"
        )

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

    max_radius = max(1, source_radius)
    # x axis spans the range from -(2*max_radius) to (distance + 4*max_radius)
    width = int(np.ceil((6 * max_radius + distance) * resolution))
    x_axis = np.linspace(
        -2 * max_radius, 4 * max_radius + distance, width, endpoint=False
    )

    # y axis spans max_radius on either side
    height = int(np.ceil(4 * max_radius * resolution))
    y_axis = np.linspace(-2 * max_radius, 2 * max_radius, height, endpoint=False)

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

    internal = distance < source_radius

    if plotting:
        plt.plot(z1x, z1y, "-", label="small image")
        plt.plot(z2x, z2y, "-", label="large image")

        lx = np.cos(th)
        ly = np.sin(th)
        plt.plot(lx, ly, ":", label="lens")

        plt.plot(ux, uy, "--", label="source")

        plt.gca().set_aspect("equal")
        plt.legend()
        plt.show()

    return z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution


def make_surface(
    z1x,
    z1y,
    z2x,
    z2y,
    height,
    width,
    x_axis,
    y_axis,
    internal,
    resolution,
    plotting=False,
):

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


def radial_lightcurve(
    source_radius,
    occulter_radius=0,
    distances=None,
    circle_points=1e4,
    resoution=1e2,
    plotting=False,
):

    if distances is None:
        distances = np.linspace(0, 10, 100, endpoint=False)

    mag = np.ones(distances.shape)
    moment = np.ones(distances.shape)

    for i, d in enumerate(distances):

        if d == source_radius:
            if i == 0:
                mag[i] = np.NAN
                moment[i] = np.NAN
            else:
                mag[i] = mag[i - 1]
                moment[i] = moment[i - 1]
        else:
            (
                z1x,
                z1y,
                z2x,
                z2y,
                height,
                width,
                x_axis,
                y_axis,
                internal,
                resolution,
            ) = draw_contours(d, source_radius, points=circle_points)

            (im, x_grid, y_grid) = make_surface(
                z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution
            )

            if occulter_radius > 0:
                im2 = remove_occulter(im, occulter_radius, x_grid, y_grid)
            else:
                im2 = im

            mag[i] = np.sum(im2)
            moment[i] = np.sum(im2 * (x_grid - d)) / mag[i]
            mag[i] /= np.pi * source_radius ** 2 * resoution ** 2

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

                x2 = source_radius * np.cos(np.linspace(0, 2 * np.pi)) + d
                y2 = source_radius * np.sin(np.linspace(0, 2 * np.pi))
                plt.plot(x2, y2, ":g", label="source radius")

                if occulter_radius > 0:
                    x3 = occulter_radius * np.cos(np.linspace(0, 2 * np.pi))
                    y3 = occulter_radius * np.sin(np.linspace(0, 2 * np.pi))
                    plt.plot(x3, y3, ":g", label="occulter")

                plt.plot(d, 0, "go", label="source center")
                plt.plot(d + moment[i], 0, "r+", label="center of light")

                plt.xlabel(
                    f"d= {d:.2f} | source r= {source_radius} | occulter r= {occulter_radius} | mag= {mag[i]:.2f} | moment= {moment[i]:.2f}"
                )
                plt.gca().set_aspect("equal")
                plt.legend()
                plt.show()
                plt.pause(0.0001)

        print(
            f"i= {i}/{distances.size}: d= {d:.2f} | mag= {mag[i]:.3f} | moment= {moment[i]:.3f}"
        )

    return mag, moment


if __name__ == "__main__":

    T = TransferMatrix()
    T.load('transfer_matrix.npz')
    # T.max_source = 2
    # T.max_dist = 3
    # T.max_occulter = 1

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
