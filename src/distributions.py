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
    obj = obj.upper()
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
    calc_probability(prob_da, obj=obj, temp_index=temp_index, mass_index=mass_index, sma_index=sma_index)

    return prob_da


def calc_probability(da, obj="WD", temp_index="mid", mass_index="mid", sma_index="mid"):
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
    obj = obj.upper()
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
    m1 = np.array(star_masses).reshape(1, len(star_masses), 1)
    m2 = np.array(lens_masses).reshape(len(lens_masses), 1, 1)
    sma = np.array(sma).reshape(1, 1, len(sma))
    x = sma / (const * m1 * m2 * (m1 + m2) * t0) ** (1 / 4)

    if alpha == -1:
        return x**3 * np.log(1 + x ** (-4))
    else:
        return -(x ** (4 + alpha)) * ((1 + x ** (-4)) ** ((alpha + 1) / 4) - 1)


def marginalize_mass_temp(ev, prob):
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

    Returns
    -------
    ds: xarray Dataset
        The dataset with the effective volume and probability distribution,
        as a function of semimajor axis only.
    """
    sma = prob.semimajor_axis.values
    if not np.array_equal(sma, ev.semimajor_axis.values):
        raise ValueError("Expected semimajor axis coordinates to be the same on ev and prob! ")

    ev_per_sma = np.zeros(len(sma), dtype=float)
    prob_per_sma = np.zeros(len(sma), dtype=float)
    for i, a in enumerate(sma):
        prob_sma = prob.isel(semimajor_axis=i)  # the probability distribution for this semimajor axis value
        new_ev = ev.isel(semimajor_axis=i)  # the effective volume for this semimajor axis value
        if len(ev.lens_temp) == 1:
            new_ev = new_ev.isel(lens_temp=0)
            prob_sma = prob_sma.isel(lens_temp=0)
        new_ev = new_ev.interp_like(prob_sma, method="cubic")
        new_ev = new_ev.where(new_ev > 0, 0)  # remove negative values (that could arise from interpolation)
        if len(ev.lens_temp) == 1:
            coords = ["star_temp", "lens_mass", "star_mass"]
        else:
            coords = ["lens_temp", "star_temp", "lens_mass", "star_mass"]
        prob_per_sma[i] = prob_sma.sum(coords)
        ev_per_sma[i] = (new_ev * prob_sma).sum(coords) / prob_per_sma[i]

    new_ev = xr.DataArray(data=ev_per_sma, coords={"semimajor_axis": sma})
    new_prob = xr.DataArray(data=prob_per_sma, coords={"semimajor_axis": sma})

    ds = xr.Dataset(data_vars={"effective_volume": new_ev, "probability": new_prob})
    ds.effective_volume.attrs["name"] = "effective_volume"
    ds.effective_volume.attrs["long_name"] = "Effective volume"
    ds.effective_volume.attrs["units"] = "pc^3"
    ds.semimajor_axis.attrs["name"] = "semimajor_axis"
    ds.semimajor_axis.attrs["long_name"] = "Semimajor axis"
    ds.semimajor_axis.attrs["units"] = "AU"

    return ds


def density_model(ds, space_density_pc3=1 / 1000):
    """
    Calculate a density model (how many systems per pc^3
    in each semimajor axis bin, and the reciprocal of
    how many pc3 are required to find one system).
    This is added to the dataset as the two new variables
    "density" and "pc3_per_system".
    Also adds the number of detections in each bin into
    the dataset as "detections".

    This modifies the "ds" parameter in place and returns nothing.

    Parameters
    ----------
    ds: xarray Dataset
        The dataset with the effective volume and probability distribution,
        as a function of semimajor axis only. This is the output of
        marginalize_mass_temp.
    space_density_pc3: float
        The space density of the objects in pc^-3.
        Default is 1/1000.

    """
    ds["density"] = ds.probability * space_density_pc3
    ds["pc3_per_system"] = 1 / ds.density
    ds["detections"] = ds.effective_volume * ds.density
    ds["detections_followup"] = ds.effective_volume_followup * ds.density

    ds.density.attrs["name"] = "density"
    ds.density.attrs["long_name"] = "Density"
    ds.density.attrs["units"] = "pc^-3"

    ds.pc3_per_system.attrs["name"] = "pc3_per_system"
    ds.pc3_per_system.attrs["long_name"] = "Average volume per system"
    ds.pc3_per_system.attrs["units"] = "pc^3"

    ds.detections.attrs["name"] = "detections"
    ds.detections.attrs["long_name"] = "Detections"

    ds.detections_followup.attrs["name"] = "detections_followup"
    ds.detections_followup.attrs["long_name"] = "Detections with followup"


def add_followup_prob(ds, frac=0.1, years=1):
    """
    Add the probability that a followup campaign would successfully
    identify repeat occurrences of the system's flare in X years
    of followup time.

    The result is added as an additional data var to ds in place.

    Parameters
    ----------
    ds: xarray Dataset
        The dataset with the effective volume, the duty cycle and period.
    frac: float
        Fraction of the time that a telescope is observing the source.
        Default is 0.25 which represents a single telescope observing
        the target for an average of 6 hours each night.
    years: float
        The number of years of followup time to consider.
        The default is 5 years (which is approximately one grad-length).

    """
    # how many times do we stand to see the flare in X years?
    number = np.round(years / (ds.orbital_period / 24 / 365.25))

    # probability to see the second flare during followup:
    prob = 1 - (1 - frac) ** number

    ds["followup_prob"] = prob
    ds["effective_volume"] = ds["effective_volume"] * prob


def fetch_distributions(
    survey,
    obj="WD",
    temp_index="mid",
    mass_index="mid",
    sma_index="mid",
    space_density_pc3=1 / 1000,
    clear_cache=False,
    verbose=False,
):
    """
    Run all the above functions, starting from loading the raw dataset
    from file and down to calculating the semimajor axis based
    distributions for effective volume, probability, density, and so on.

    Parameters
    ----------
    survey: str
        The survey to get the distributions for.
        Can be "TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY" or "LAST".
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
        This is only for the lens mass, and mostly relevant for the NS/BH case.
    sma_index: str or float
        The semimajor axis model to use. Can be "low", "mid" or "high" or a float
        value for the semimajor axis power law index.
        This is the power law index of the separation when leaving the common envelope
        stage, with evolution given by gravitational radiation,
        see reference in Maoz et al 2018
        (https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.2584M/abstract)
    space_density_pc3: float
        The space density of the objects in pc^-3.
        Default is 1/1000.
    clear_cache: bool
        Wether to clear the results from previous calculations.
        Default is False, use the cached results.
        We cache the marginalize_mass_temp() results in the "saved" folder
        in files that correspond to all the mass/temp/sma indices and object type.
    verbose: bool
        Wether to print out some information about the calculations.

    Returns
    -------
    out_ds: xarray Dataset
        The dataset with the effective volume and probability distribution,
        as a function of semimajor axis only. Also adds the density of systems,
        the number of pc^3 required to find one system, the number of detections,
        and the number of detections (and effective volume) accounting for
        which systems can be followed up (with the default parameters
        given in followup_prob()).
    """
    obj = obj.upper()
    if obj not in ["WD", "BH"]:
        raise ValueError('obj must be "WD" or "BH"')

    survey = survey.upper()
    if survey not in ["TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]:
        raise ValueError('survey must be "TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY" or "LAST"')

    # try to load the cached value:
    cache_filename = f"saved/marginalized_{survey}_{obj}_{temp_index}_{mass_index}_{sma_index}.nc"
    cache_filename = os.path.join(ROOT_FOLDER, cache_filename)
    if clear_cache or not os.path.isfile(cache_filename):
        # cannot (or not allowed to use) the cached value, so calculate it:
        if verbose and clear_cache:
            print(f"Not allowed to use cache, recaclulating {cache_filename}")
        if verbose and not os.path.isfile(cache_filename):
            print(f"No cache file {cache_filename}, calculating new values")

        # get the raw data from file
        filename = os.path.join(ROOT_FOLDER, f"saved/simulate_{survey}_{obj}.nc")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Cannot find file {filename}. Try using utils.fetch_volumes first!")

        ds = xr.load_dataset(filename, decode_times=False)

        # do the marginalizations and probability distributions:
        da = marginalize_declinations(ds)
        prob = get_prob_dataset(da, obj=obj, temp_index=temp_index, mass_index=mass_index, sma_index=sma_index)
        ds_sma = marginalize_mass_temp(da, prob)

        add_followup_prob(ds)  # this will reduce the effective volume with the followup probability
        da2 = marginalize_declinations(ds)
        prob2 = get_prob_dataset(da2, obj=obj, temp_index=temp_index, mass_index=mass_index, sma_index=sma_index)
        ds_sma2 = marginalize_mass_temp(da2, prob2)

        # add the effective volume weighted by followup probability to the dataset
        ds_sma["effective_volume_followup"] = ds_sma2["effective_volume"]

        density_model(ds_sma, space_density_pc3=space_density_pc3)

        # save the result:
        ds_sma.to_netcdf(cache_filename)

    else:  # can just load the cached value
        if verbose:
            print(f"loading cached {cache_filename}")
        ds_sma = xr.load_dataset(cache_filename, decode_times=False)

        # rescale to the new space density:
        old_density = float(ds_sma.density.sum())
        rescale = space_density_pc3 / old_density
        ds_sma["density"] *= rescale
        ds_sma["pc3_per_system"] /= rescale
        ds_sma["detections"] *= rescale
        ds_sma["detections_followup"] *= rescale

    return ds_sma


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    filename = os.path.join(ROOT_FOLDER, f"saved/simulate_LSST_WD.nc")
    ds_wd = xr.load_dataset(filename, decode_times=False)
    da_wd = marginalize_declinations(ds_wd)
    prob_wd = get_prob_dataset(da_wd, obj="WD")
    ds_sma_wd = marginalize_mass_temp(da_wd, prob_wd)

    add_followup_prob(ds_wd)  # this will reduce the effective volume with the followup probability
    da_wd2 = marginalize_declinations(ds_wd)
    prob2 = get_prob_dataset(da_wd2, obj="WD")
    ds_sma_wd2 = marginalize_mass_temp(da_wd2, prob2)

    # add the effective volume weighted by followup probability to the dataset
    ds_sma_wd["effective_volume_followup"] = ds_sma_wd2["effective_volume"]
    density_model(ds_sma_wd)

    filename = os.path.join(ROOT_FOLDER, f"saved/simulate_LSST_BH.nc")
    ds_bh = xr.load_dataset(filename, decode_times=False)
    da_bh = marginalize_declinations(ds_bh)
    prob_bh = get_prob_dataset(da_bh, obj="BH", mass_index="mid")
    ds_sma_bh = marginalize_mass_temp(da_bh, prob_bh)

    add_followup_prob(ds_bh)  # this will reduce the effective volume with the followup probability
    da_bh2 = marginalize_declinations(ds_bh)
    prob2 = get_prob_dataset(da_bh2, obj="BH", mass_index="mid")
    ds_sma_bh2 = marginalize_mass_temp(da_bh2, prob2)

    # add the effective volume weighted by followup probability to the dataset
    ds_sma_bh["effective_volume_followup"] = ds_sma_bh2["effective_volume"]
    density_model(ds_sma_bh)
