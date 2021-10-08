import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

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

    if epsilon < 0.1:
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
    x_axis = np.linspace(-1 - 2 * max_radius, 4 * max_radius + distance, width, endpoint=False)

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


def make_surface(z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution, plotting=True):

    im1 = np.zeros((height, width))
    im2 = np.zeros((height, width))

    for i in range(z1x.size):

        z1x_ind = int(np.round((z1x[i] - x_axis[0]) * resolution))
        z1y_ind = int(np.round((z1y[i] - y_axis[0]) * resolution))
        if 0 <= z1x_ind < width and 0 <= z1y_ind < height:
            im1[z1y_ind, z1x_ind] = 1

        z2x_ind = int(np.round((z2x[i] - x_axis[0]) * resolution))
        z2y_ind = int(np.round((z2y[i] - y_axis[0]) * resolution))
        if 0 <= z2x_ind < width and 0 <= z2y_ind < height:
            im2[z2y_ind, z2x_ind] = 1

    for i in range(im1.shape[0]):
        indices = np.nonzero(im1[i])[0]  # where the contours are non-zero
        if indices.size:
            mx = np.max(indices)
            mn = np.min(indices)
            im1[i, mn:mx] = 1

    for i in range(im2.shape[0]):
        indices = np.nonzero(im2[i])[0]  # where the contours are non-zero
        if indices.size:
            mx = np.max(indices)
            mn = np.min(indices)
            im2[i, mn:mx] = 1

    if internal:
        im = im2 - im1
    else:
        im = im2 + im1

    if plotting:
        plt.figure()
        ex = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
        plt.imshow(im, extent=ex)
        plt.show()

    return im


if __name__ == "__main__":

    t1 = timer()
    (z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution) = draw_contours(2+0.5, 2.0)
    t2 = timer()
    print(f"Draw contours: elapsed time is {t2 - t1}s. ")

    t1 = timer()
    im = make_surface(z1x, z1y, z2x, z2y, height, width, x_axis, y_axis, internal, resolution)
    t2 = timer()
    print(f"Make surface: elapsed time is {t2 - t1}s. ")

