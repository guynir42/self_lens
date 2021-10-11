import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer


class TransferMatrix:

    def __init__(self):
        self.resolution = 1e2
        self.num_points = 1e5

        self.distances = np.arange(0, 200) / 10
        self.source_radii = np.arange(1, 100) / 10
        self.lens_radii = np.arange(0, 30) / 10

        # the following are 3D matrices, corresponding
        # to the amount of flux and 1st moments,
        # that are produced by one uniform annulus of
        # the source, with surface brightness = 1
        # the axes of each data matrix correspond to
        # 0: lens radius, 1: source size, 2: distance
        self.flux = None
        self.input_flux = None
        self.magnification = None
        self.moment = None

    def calculate(self, plotting=False):

        self.flux = np.zeros((self.lens_radii.size, self.source_radii.size, self.distances.size), dtype=np.single)
        self.input_flux = np.zeros(self.source_radii.size, dtype=np.single)
        self.moment = np.zeros((self.lens_radii.size, self.source_radii.size, self.distances.size), dtype=np.single)

        if plotting:
            plt.ion()

        t0 = timer()

        for i, d in enumerate(self.distances):

            for j, sr in enumerate(self.source_radii):

                if d != sr:  # when they are the same (caustic crossing), use the previously calculated contour
                    (z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution) = draw_contours(
                        d,
                        sr,
                        points=self.num_points,
                        resolution=self.resolution,
                    )
                (im, x_grid, y_grid) = make_surface(z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution)

                for k, lr in enumerate(self.lens_radii):
                    im2 = remove_occulter(im, lr, x_grid, y_grid)
                    self.input_flux[j] = np.pi * sr ** 2 * self.resolution ** 2
                    self.flux[k, j, i] = np.sum(im2)
                    self.moment[k, j, i] = np.sum(im2 * (x_grid - d))

                    if plotting:
                        offset = self.moment[k, j, i] / self.flux[k, j, i]
                        plt.cla()
                        plt.imshow(im2, vmin=0, vmax=1, extent=(np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)))
                        plt.xlabel(f'd= {d} | sr= {sr} | lr= {lr} | mag= {self.flux[k, j, i] / self.input_flux[j]:.1f} | offset= {offset:.2f}')
                        plt.plot(d, 0, 'go')
                        plt.plot(d + offset, 0, 'r+')
                        plt.gca().set_aspect('equal')
                        plt.show()
                        plt.pause(0.0001)

                    if j > 0:  # keep only the values from the annulus, not the entire disk
                        self.input_flux[j] -= self.input_flux[j - 1]
                        self.flux[k, j, i] -= self.flux[k, j - 1, i]
                        self.moment[k, j, i] -= self.moment[k, j - 1, i]


            t1 = timer()
            print(f'{i:3d}/{self.distances.size} | Distance= {d} | runtime = {t1-t0:.1f}s')

        self.magnification = self.flux / np.expand_dims(self.input_flux, axis=[0, 2])


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
        raise ValueError('Cannot find the contours for exact match distance==source_radius')

    if epsilon < 0.3:
        th1 = np.linspace(0, 1 - 2*epsilon, int(np.ceil(points / 2)), endpoint=False) * np.pi
        th1 = np.append(-th1, th1, axis=0)
        th2 = np.geomspace(1 - 2*epsilon, 1, int(np.ceil(points / 2)), endpoint=False) * np.pi
        th2 = np.append(-th2, th2, axis=0)
        th = np.append(th1, th2, axis=0)
        th = np.sort(th)
    else:
        th = np.linspace(-1, 1, int(np.ceil(points / 2)), endpoint=False) * np.pi

    max_radius = max(1, source_radius)
    # x axis spans the range from -(2*max_radius) to (distance + 4*max_radius)
    width = int(np.ceil((6 * max_radius + distance) * resolution))
    x_axis = np.linspace(- 2 * max_radius, 4 * max_radius + distance, width, endpoint=False)

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
        plt.plot(z1x, z1y, '-', label='small image')
        plt.plot(z2x, z2y, '-', label='large image')

        lx = np.cos(th)
        ly = np.sin(th)
        plt.plot(lx, ly, ':', label='lens')

        plt.plot(ux, uy, '--', label='source')

        plt.gca().set_aspect('equal')
        plt.legend()
        plt.show()

    return z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution


def make_surface(z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution, plotting=False):

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


def radial_lightcurve(source_radius, occulter_radius=0, distances=None, circle_points=1e4, resoution=1e2, plotting=False):

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
            (z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution) = draw_contours(d, source_radius, points=circle_points)

            (im, x_grid, y_grid) = make_surface(z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution)

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
                plt.imshow(im2, vmin=0, vmax=1, extent=(np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)))

                x = np.cos(np.linspace(0, 2*np.pi))
                y = np.sin(np.linspace(0, 2*np.pi))

                plt.plot(x, y, '--r', label="lens")

                x2 = source_radius * np.cos(np.linspace(0, 2 * np.pi)) + d
                y2 = source_radius * np.sin(np.linspace(0, 2 * np.pi))
                plt.plot(x2, y2, ':g', label="source radius")

                if occulter_radius>0:
                    x3 = occulter_radius * np.cos(np.linspace(0, 2*np.pi))
                    y3 = occulter_radius * np.sin(np.linspace(0, 2*np.pi))
                    plt.plot(x3, y3, ':g', label="occulter")

                plt.plot(d, 0, 'go', label="source center")
                plt.plot(d + moment[i], 0, 'r+', label="center of light")

                plt.xlabel(
                    f'd= {d:.2f} | sr= {source_radius} | lr= {occulter_radius} | mag= {mag[i]:.2f} | offset= {moment[i]:.2f}')
                plt.gca().set_aspect('equal')
                plt.legend()
                plt.show()
                plt.pause(0.0001)

        print(f'i= {i}/{distances.size}: d= {d:.2f} | mag= {mag[i]:.3f} | moment= {moment[i]:.3f}')

    return mag, moment

if __name__ == "__main__":

    distance = 1.3
    source_radius = 2.0
    occulter_radius = 0.4

    t1 = timer()
    (z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution) = draw_contours(distance, source_radius)
    t2 = timer()
    print(f"Draw contours: elapsed time is {t2 - t1}s. ")

    t1 = timer()
    (im, x_grid, y_grid) = make_surface(z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution)
    t2 = timer()
    print(f"Make surface: elapsed time is {t2 - t1}s. ")

    t1 = timer()
    im2 = remove_occulter(im, occulter_radius, x_grid, y_grid)
    t2 = timer()
    print(f"Remove occulter: elapsed time is {t2 - t1}s. ")
