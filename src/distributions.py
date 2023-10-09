import os

import numpy as np
import xarray as xr


ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def marginalize_declinations(ds):
    """
    Marginalize over declinations to get the effective volume.
    This assumes the declinations are uniformly sampled in space,
    so we can get a weighted average of the effective volume
    over all contributions from all declinations.

    Parameters
    ----------
    ds: xarray dataset
        The dataset with the effective volume.
        It will produce a new effective volume dataset where the declination
        has been marginalized over.

    Returns
    -------
    xarray DataArray
        The data array with the marginalized effective volume.
    """
    dec = ds.declination * np.pi / 180  # convert to radians
    # dec_step = dec[1] - dec[0]  # do something more complicated if uneven steps
    dec_step = np.diff(dec)  # each dec value accounts for the range of declinations back down to previous point
    new_ev = ds.effective_volume.isel(declination=slice(1, None))
    new_ev = (new_ev * np.cos(dec[1:]) * dec_step).sum(dim="declination")
    new_ev.attrs["name"] = "effective_volume"
    new_ev.attrs["long_name"] = "Effective volume"
    new_ev.attrs["units"] = "pc^3"
    ds["marginalized_volume"] = new_ev

    return new_ev


def get_prob_dataset(da, obj="WD", temp_index="mid", mass_index="mid", sma_index="mid"):
    """
    Generate a dataset with a fine coordinate grid
    for lens_mass, star_mass, lens_temp and star_temp,
    either for white dwarf or a black hole/neutron star objects.

    The dataset will have a normalized probability distribution
    across the different parameters.

    Parameters
    ----------
    da: xarray DataArray
        The data array with the effective volume, to use as a template for
        the limits of the coordinates of the probability distribution.
        The semimajor axis coordinate will be used as is (it should
        be well sampled).
    obj: str
        The type of object to get the probability distribution for.
        Can be "WD" for white dwarfs or "BH" for black holes and neutron stars.
    temp_index: str or float
        The temperature model to use. Can be "low", "mid" or "high" or a float
        value for the temperature power law index.
    mass_index: str or float or tuple of floats
        The mass model to use. Can be "low", "mid" or "high" or a float value
        for the mass power law index, or a tuple of floats for a Gaussian
        distribution with given mean and standard deviation (in solar mass).
    sma_index: str or float
        The semimajor axis model to use. Can be "low", "mid" or "high" or a float
        value for the semimajor axis power law index.

    Returns
    -------
    prob_da: xarray DataArray
        A new data array with the normalized probability distribution.
    """
    if obj not in ["WD", "BH"]:
        raise ValueError('obj must be "WD" or "BH"')

    dm = 0.05
    dT = 500

    # the sources are white dwarfs anyway
    star_masses = np.arange(da.star_mass.min(), da.star_mass.max(), dm)
    star_temps = np.arange(da.star_temp.min(), da.star_temp.max(), dT)

    # regardless of object type, we want a fine grid on the masses
    lens_masses = np.arange(da.lens_mass.min(), da.lens_mass.max(), dm)

    if obj == "WD":
        lens_temps = np.arange(da.lens_temp.min(), da.lens_temp.max(), dT)
    elif obj == "BH":
        lens_temps = da.lens_temp.values  # should be only one value anyway

    sma = da.semimajor_axis.values

    # get the probability distribution for the different parameters
    # and normalize it
    prob = np.zeros((len(lens_temps), len(star_temps), len(lens_masses), len(star_masses), len(sma)), dtype=float)

    # create the dataset
    prob_da = xr.DataArray(
        data=prob,
        coords={
            "lens_temp": lens_temps,
            "star_temp": star_temps,
            "lens_mass": lens_masses,
            "star_mass": star_masses,
            "semimajor_axis": sma,
        },
    )

    # the probability distribution calculation:
    get_probability(prob_da, obj=obj, temp_index=temp_index, mass_index=mass_index, sma_index=sma_index)

    return prob_da


def get_probability(da, obj="WD", temp_index="mid", mass_index="mid", sma_index="mid"):
    """
    Use the parameters in reference Maoz et al 2018
    (https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2584M/abstract)
    to generate a probability density for white dwarfs.
    Each parameter (temp, mass, sma->semimajor axis)
    controls if we want to get the "mid" (best estimate
    value) or "low" or "high" for the lower or higher
    models.

    Parameters
    ----------
    da: xarray DataArray
        The data array that would contain the normalized probability distribution,
        in the coordinates that we expect to work with (semimajor_axis, lens_mass,
        star_mass, lens_temp, star_temp). These will have a tighter grid than the
        coordinates of the effective volume, so we will have to interpolate
        that one to this grid.
        This data array is modified in place to hold the probability distribution.
    obj: str
        The type of object to get the probability distribution for.
        Can be "WD" for white dwarfs or "BH" for black holes and neutron stars.
    temp_index: str or float
        The temperature model to use. Can be "low", "mid" or "high" or a float
        value for the temperature power law index.
    mass_index: str or float or tuple of floats
        The mass model to use. Can be "low", "mid" or "high" or a float value
        for the mass power law index, or a tuple of floats for a Gaussian
        distribution with given mean and standard deviation (in solar mass).
    sma_index: str or float
        The semimajor axis model to use. Can be "low", "mid" or "high" or a float
        value for the semimajor axis power law index.
    """
    if obj not in ["WD", "BH"]:
        raise ValueError('obj must be "WD" or "BH"')

    # less negative slope of the temperature power law
    # indicates more hot WDs and so, more detections.
    if isinstance(temp_index, (int, float)):
        temp_index = float(temp_index)
    elif temp_index == "mid":
        temp_index = -2
    elif temp_index == "low":
        temp_index = -3
    elif temp_index == "high":
        temp_index = -1
    else:
        raise ValueError(f'Unknown option "{temp_index}". Use (low|high|mid) or a float.')

    # right now, only have one mass model for WDs
    star_mass = (0.6, 0.2)

    # the lens mass could be a WD or a BH/NS
    if isinstance(mass_index, (int, float)):
        lens_mass = float(mass_index)
    elif isinstance(mass_index, (list, tuple)) and len(mass_index) == 2:
        lens_mass = tuple(float(m) for m in mass_index)
    elif isinstance(mass_index, str):
        if obj == "WD":
            lens_mass = (0.6, 0.2)
        elif obj == "BH":
            if mass_index == "mid":
                lens_mass = -2.3
            elif mass_index == "low":
                lens_mass = -2.7
            elif mass_index == "high":
                lens_mass = -2.0
            else:
                raise ValueError(f'Unknown option "{mass_index}". Use (low|high|mid) or a float.')
    else:
        raise ValueError(f'Unknown option "{mass_index}". Use (low|high|mid) or a float or a 2-tuple of floats.')

    if isinstance(sma_index, (int, float)):
        sma_index = float(sma_index)
    elif sma_index == "mid":
        sma_index = -1.3
    elif sma_index == "low":
        sma_index = -1.5
    elif sma_index == "high":
        sma_index = -1.0
    else:
        raise ValueError(f'Unknown option "{sma_index}". Use (low|high|mid) or a float.')

    # done parsing inputs, now make the actual probability matrices:
    if len(da.lens_temp) == 1:
        prob_lens_temp = np.array([1])  # in case of BH/NS lens, the temperature is zero, so we just ignore this
    else:
        prob_lens_temp = da.lens_temp.values ** float(temp_index)
    prob_star_temp = da.star_temp.values ** float(temp_index)

    prob_star_mass = np.exp(-0.5 * (da.star_mass.values - star_mass[0]) ** 2 / star_mass[1] ** 2)

    if isinstance(lens_mass, (tuple, list)) and len(lens_mass) == 2:
        prob_lens_mass = np.exp(-0.5 * (da.lens_mass.values - lens_mass[0]) ** 2 / lens_mass[1] ** 2)
    elif isinstance(lens_mass, (int, float)):
        prob_lens_mass = da.lens_mass.values ** float(lens_mass)
    else:
        raise ValueError(f"lens_mass must be a float or 2-tuple of floats, got {lens_mass}.")

    prob_sma = semimajor_axis_distribution(
        sma=da.semimajor_axis.values,
        lens_masses=da.lens_mass.values,
        star_masses=da.star_mass.values,
        alpha=sma_index,
    )

    # add dimensions to allow broadcasting:
    prob_lens_temp = prob_lens_temp.reshape((len(prob_lens_temp), 1, 1, 1, 1))
    prob_star_temp = prob_star_temp.reshape((1, len(prob_star_temp), 1, 1, 1))
    prob_lens_mass = prob_lens_mass.reshape((1, 1, len(prob_lens_mass), 1, 1))
    prob_star_mass = prob_star_mass.reshape((1, 1, 1, len(prob_star_mass), 1))

    # this output array is not a vector, but has meaningful dimensions in lens/star mass as well!
    prob_sma = prob_sma.reshape(1, 1, len(da.lens_mass), len(da.star_mass), len(da.semimajor_axis))

    # multiply all the probabilities together
    prob = prob_lens_temp * prob_star_temp * prob_lens_mass * prob_star_mass * prob_sma

    # normalize the probability distribution
    prob /= prob.sum()

    da.data = prob


def semimajor_axis_distribution(sma, star_masses, lens_masses, alpha=-1.3):
    """
    Calculate the probability to find a system with a given semimajor axis (sma).
    Uses the parametrization given in reference Maoz et al 2018
    (https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2584M/abstract)
    This tells us the present day distribution of semimajor axis
    for binary WDs with an initial sma distribution with power law
    index alpha.

    Parameters
    ----------
    sma: array-like
        The semimajor axis values to calculate the probability for.
    star_masses: array-like
        The masses of the stars in solar masses.
    lens_masses: array-like
        The masses of the lenses in solar masses.
    alpha: scalar float
        The power law index of the zero age, semimajor axis power law index
        for binary white dwarfs that emerge from the common envelope phase.

    Returns
    -------
    prob: array-like
        The distribution of the number density (not normalized to anything)
        of the binary white dwarfs, based on their masses and semimajor axis.
        The shape of the output array is (len(lens_masses), len(star_masses), len(sma)).
    """
    # G_over_c = 1.2277604981899157e-73  # = G**3/c**5 = (6.6743e-11) ** 3 / (299792458) ** 5 in MKS!
    # const = 6.286133750732368e-72  # = 256 / 5 * G_over_c
    # const = const / (1.496e11) ** 4 * 3.15576e16 * (1.989e30) ** 3  # unit conversions
    const = 3.116488722396829e-09  # in units of AU^4 Gyr^-1 solar_mass ^-3

    t0 = 13.5  # age of the galaxy in Gyr
    m1 = star_masses.reshape(1, len(star_masses), 1)
    m2 = lens_masses.reshape(len(lens_masses), 1, 1)
    sma = sma.reshape(1, 1, len(sma))
    x = sma / (const * m1 * m2 * (m1 + m2) * t0) ** (1 / 4)

    if alpha == -1:
        return x**3 * np.log(1 + x ** (-4))
    else:
        return -(x ** (4 + alpha)) * ((1 + x ** (-4)) ** ((alpha + 1) / 4) - 1)


def marginalize_mass_temp(ev, prob, use_loop=True):
    """
    Marginalize over lens and star temperatures and masses.
    Returns a new dataset that holds the effective volume in one
    data array (ev) and the probability to find a system in each
    semimajor axis value in another data array (prob).
    In both cases these are the weighted average over the
    mass and temperature of lens and star.

    The ev data array is usually on a less fine grid than the prob data array.
    For that reason we must interpolate ev unto prob.
    However, because these arrays are very large, we will
    do this in each individual semimajor axis step separately,
    and do the sum over the interpolated product in a loop.
    To skip the loop set use_loop=False, but be careful as that
    can easily blow up the RAM.

    To calculate the total number of detections, multiply ev * prob
    and integrate over semimajor axis (including the variable step size dsma).
    This sum is the number of detections assuming a single system per 1pc^3
    including all values of semimajor axis, temperatures and masses in range.

    Parameters
    ----------
    ev: xarray DataArray
        The effective volume, marginalized over declinations.
    prob: xarray DataArray
        The probability distribution, on a fine grid over lens/star mass/temperature
        and semimajor axis.
    use_loop: bool
        Whether to use a loop to calculate the weighted average.
        This is slower but uses less RAM.
        Default is True.

    Returns
    -------
    ds: xarray Dataset
        The dataset with the effective volume and probability distribution,
        as a function of semimajor axis only.
    """
    sma = prob.semimajor_axis.values
    if not np.array_equal(sma, ev.semimajor_axis.values):
        raise ValueError("Expected semimajor axis coordinates to be the same on ev and prob! ")

    if use_loop is False:
        new_ev = ev.interp_like(prob, method="cubic") * prob
        new_ev = new_ev.sum(["lens_mass", "star_mass", "lens_temp", "star_temp"])

    else:  # the default uses a loop to avoid memory overflow
        ev_per_sma = np.zeros(len(sma), dtype=float)
        prob_per_sma = np.zeros(len(sma), dtype=float)
        for i, a in enumerate(sma):
            prob_sma = prob.isel(semimajor_axis=i)  # the probability distribution for this semimajor axis value
            new_ev = ev.isel(semimajor_axis=i)  # the effective volume for this semimajor axis value
            if len(ev.lens_temp) == 1:
                new_ev = new_ev.isel(lens_temp=0)
                prob_sma = prob_sma.isel(lens_temp=0)
            new_ev = new_ev.interp_like(prob_sma, method="cubic")
            if len(ev.lens_temp) == 1:
                coords = ["star_temp", "lens_mass", "star_mass"]
            else:
                coords = ["lens_temp", "star_temp", "lens_mass", "star_mass"]
            prob_per_sma[i] = prob_sma.sum(coords)
            ev_per_sma[i] = (new_ev * prob_sma).sum(coords) / prob_per_sma[i]

        new_ev = xr.DataArray(data=ev_per_sma, coords={"semimajor_axis": sma})
        new_prob = xr.DataArray(data=prob_per_sma, coords={"semimajor_axis": sma})

    ds = xr.Dataset(data_vars={"effective_volume": new_ev, "probability": new_prob})

    return ds


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    filename = os.path.join(ROOT_FOLDER, f"saved/simulate_LSST_WD.nc")
    ds_wd = xr.load_dataset(filename, decode_times=False)
    da_wd = marginalize_declinations(ds_wd)
    prob_wd = get_prob_dataset(da_wd, obj="WD")
    # ds_sma_wd = marginalize_mass_temp(da_wd, prob_wd, use_loop=True)

    filename = os.path.join(ROOT_FOLDER, f"saved/simulate_LSST_BH.nc")
    ds_bh = xr.load_dataset(filename, decode_times=False)
    da_bh = marginalize_declinations(ds_bh)
    prob_bh = get_prob_dataset(da_bh, obj="BH", mass_index="mid")
    ds_sma_bh = marginalize_mass_temp(da_bh, prob_bh, use_loop=True)
