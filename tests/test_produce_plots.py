import os
import pytest

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from num2tex import num2tex, configure

from src.transfer_matrix import single_geometry, radial_lightcurve
from src.simulator import Simulator
from src.grid_scan import Grid
from src.distributions import fetch_distributions, semimajor_axis_distribution


CLEAR_CACHE = False

# ref: https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython
matplotlib.use("Agg")

if not os.path.isdir("plots"):
    os.mkdir("plots")


def test_make_individual_geometries():
    if not os.path.isdir("plots/individual_geometries"):
        os.mkdir("plots/individual_geometries")

    plt.figure(num=0, figsize=[12, 6])

    distances = np.linspace(0, 5, 20)
    for i, d in enumerate(distances):
        single_geometry(distance=d, source_radius=0.8, occulter_radius=0.25, plotting=True, legend=True)
        plt.show()
        plt.pause(0.2)

        plt.savefig(f"plots/individual_geometries/geometry_{i}.pdf")
        plt.savefig(f"plots/individual_geometries/geometry_{i}.png")


def test_make_multiple_geometries():
    fig, axes = plt.subplots(1, 3, num=1, figsize=[12, 4])
    distances = [0, 0.75, 1.05]

    for i, d in enumerate(distances):
        single_geometry(
            distance=d,
            source_radius=0.8,
            occulter_radius=0.25,
            plotting=True,
            legend=False,
            axes=axes[i],
            left=-2.0 + 0.6 * d,
            right=2.0 + 0.6 * d,
            top=1.5,
        )
        axes[i].set_title(f"Separation= {d}")
        if i != 1:
            axes[i].set_xlabel("")
        # if i > 0:
        axes[i].set_yticklabels([])

    axes[0].set_position([0.05, 0.1, 0.225, 0.8])
    axes[1].set_position([0.30, 0.1, 0.225, 0.8])
    axes[2].set_position([0.55, 0.1, 0.225, 0.8])
    axes[2].legend(bbox_to_anchor=(1.05, 1.05), loc="upper left", fontsize=14)

    # save the figure
    plt.savefig("plots/multiple_geometries.pdf")
    plt.savefig("plots/multiple_geometries.png")


def test_make_lightcurves_sources():
    distances = np.linspace(0, 2.0, 20)
    # occulters = [0, 0.85, 1.0, 1.2]
    sources = [0.25, 0.5, 1.0, 2.0]
    mag = []
    for i, s in enumerate(sources):
        mag.append(radial_lightcurve(distances=distances, source_radius=s, occulter_radius=0))

    # plot these
    fig = plt.figure(num=2, figsize=[12, 6])
    fig.clf()
    # markers = ['*', 'v', 's', 'o', 's', 'x']
    line_styles = ["-", "--", "-.", ":"]
    for i, s in enumerate(sources):
        plt.plot(
            distances,
            mag[i],
            lw=4.0,
            ms=10,
            marker=None,
            ls=line_styles[i],
            fillstyle="full",
            label=f"{s:.2f}",
        )

    plt.legend(fontsize=24, title="source radius", title_fontsize=22, framealpha=0.0)
    plt.xlabel("Source-lens distance [Einstein radii]", fontsize=24)
    plt.ylabel("magnification", fontsize=24)
    [l.set_fontsize(24) for l in plt.gca().get_xticklabels()]
    [l.set_fontsize(24) for l in plt.gca().get_yticklabels()]

    plt.gca().set_position([0.1, 0.15, 0.85, 0.8])
    plt.show()

    # save the plots
    plt.savefig("plots/example_lcs_sources.pdf")
    plt.savefig("plots/example_lcs_sources.png")


def test_make_lightcurves_occulters():
    distances = np.linspace(0, 2.0, 20)
    occulters = [0, 0.85, 1.0, 1.2]
    source = 0.5
    mag = []
    for i, occ in enumerate(occulters):
        mag.append(radial_lightcurve(distances=distances, source_radius=source, occulter_radius=occ))

    # plot these
    fig = plt.figure(num=3, figsize=[12, 6])
    fig.clf()
    # markers = ['*', 'v', 's', 'o', 's', 'x']
    line_styles = ["-", "--", "-.", ":"]

    for i, occ in enumerate(occulters):
        plt.plot(
            distances,
            mag[i],
            lw=4.0,
            ms=10,
            marker=None,
            ls=line_styles[i],
            fillstyle="full",
            label=f"{occ:.2f}",
        )

    plt.legend(fontsize=24, title="occulter radius", title_fontsize=22, framealpha=0.0)
    plt.xlabel("Source-lens distance [Einstein radii]", fontsize=24)
    plt.ylabel("magnification", fontsize=24)
    [l.set_fontsize(24) for l in plt.gca().get_xticklabels()]
    [l.set_fontsize(24) for l in plt.gca().get_yticklabels()]

    plt.gca().set_position([0.1, 0.15, 0.85, 0.8])
    plt.show()

    # save the plots
    plt.savefig("plots/example_lcs_occulters.pdf")
    plt.savefig("plots/example_lcs_occulters.png")


def test_make_astrometry_curves():
    distances = np.linspace(0, 2.0, 20)
    sources = [0.25, 0.5, 1.0, 2.0]
    offsets = []
    for i, s in enumerate(sources):
        _, offset = radial_lightcurve(distances=distances, source_radius=s, occulter_radius=0, get_offsets=True)
        offsets.append(offset)

    # plot these
    fig = plt.figure(num=3, figsize=[12, 6])
    fig.clf()
    # markers = ['*', 'v', 's', 'o', 's', 'x']
    line_styles = ["-", "--", "-.", ":"]
    for i, s in enumerate(sources):
        plt.plot(
            distances,
            offsets[i],
            lw=4.0,
            ms=10,
            marker=None,
            ls=line_styles[i],
            fillstyle="full",
            label=f"{s:.2f}",
        )
    plt.xlim(-0.15, 2.0)
    plt.legend(fontsize=24, loc="upper left", title="source radius", title_fontsize=22, framealpha=0.0)
    plt.xlabel("Source-lens distance [Einstein radii]", fontsize=24)
    plt.ylabel("astrometric offset [Einstein radii]", fontsize=24)
    [l.set_fontsize(24) for l in plt.gca().get_xticklabels()]
    [l.set_fontsize(24) for l in plt.gca().get_yticklabels()]

    plt.gca().set_position([0.13, 0.14, 0.82, 0.80])
    plt.show()

    # save the plots
    plt.savefig("plots/example_astrometric_shifts.pdf")
    plt.savefig("plots/example_astrometric_shifts.png")


def test_make_example_system_wd_wd(sim):
    sim.calculate(
        star_mass=0.3,
        lens_mass=0.6,
        declination=0.02,
        semimajor_axis=0.15,
        star_temp=8000,
        lens_temp=4000,
    )
    sim.syst.plot(font_size=24)

    plt.gcf().set_size_inches([16, 8])

    # save the plots
    plt.savefig("plots/example_wd_wd_system.pdf")
    plt.savefig("plots/example_wd_wd_system.png")


def test_make_example_system_eclipse(sim):
    sim.calculate(
        star_mass=0.6,
        lens_mass=0.6,
        declination=0.05,
        semimajor_axis=0.08,
        star_temp=8000,
        lens_temp=4000,
    )
    sim.syst.plot(font_size=24)

    plt.gcf().set_size_inches([16, 8])

    # save the plots
    plt.savefig("plots/example_wd_wd_system_eclipse.pdf")
    plt.savefig("plots/example_wd_wd_system_eclipse.png")


def test_make_example_system_wd_ns(sim):
    sim.calculate(
        star_mass=0.6,
        lens_mass=1.6,
        declination=0.05,
        semimajor_axis=0.08,
        star_temp=8000,
        lens_temp=4000,
    )
    sim.syst.plot(font_size=24)

    plt.gcf().set_size_inches([16, 8])

    # save the plots
    plt.savefig("plots/example_wd_ns_system.pdf")
    plt.savefig("plots/example_wd_ns_system.png")


def test_make_example_system_wd_bh(sim):
    sim.calculate(
        star_mass=0.6,
        lens_mass=10.0,
        declination=0.5,
        semimajor_axis=0.0005,
        star_temp=8000,
        lens_temp=4000,
    )
    sim.syst.plot(font_size=24)

    plt.gcf().set_size_inches([16, 8])

    # save the plots
    plt.savefig("plots/example_wd_bh_system.pdf")
    plt.savefig("plots/example_wd_bh_system.png")


def test_make_magnification_vs_period_plot(sim):
    original_font_size = matplotlib.rcParams["font.size"]
    try:
        matplotlib.rcParams["font.size"] = 22

        sim.star_mass = 0.6
        sim.star_temp = 7000
        sim.declination = 0

        fig = plt.figure(num=4, figsize=[23, 14])
        plt.clf()
        # ax = fig.add_axes([0.08, 0.08, 0.65, 0.87])
        ax = fig.add_axes([0.08, 0.08, 0.87, 0.87])

        thresholds = [0.03, 0.3, np.inf]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # for WD lenses
        masses = np.array([0.3, 0.6, 1.0, 1.2])
        styles = ["*", "v", "s", "p"]
        orbits = np.geomspace(0.0001, 1, 120)
        peak_mags = np.zeros((len(masses), len(orbits)))
        source_sizes = np.zeros(peak_mags.shape)
        periods = np.zeros(peak_mags.shape)
        prob_low = np.zeros(peak_mags.shape)
        prob_high = np.zeros(peak_mags.shape)
        roche = np.zeros(peak_mags.shape)

        for i, m in enumerate(masses):
            t = 0
            sim.lens_mass = m
            for j, a in enumerate(orbits):
                sim.semimajor_axis = a
                mag = sim.calculate()
                source_sizes[i, j] = sim.source_size
                periods[i, j] = sim.orbital_period * 3600
                peak_mags[i, j] = np.max(mag - 1)
                prob_low[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[0], threshold=1)
                prob_high[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[1], threshold=1)
                roche[i, j] = sim.roche_lobe / sim.star_size

                if peak_mags[i, j] > thresholds[t]:
                    vertical = "top" if i % 2 == 0 else "bottom"
                    horizontal = "center" if i % 2 == 1 else "left"
                    prob = sim.best_prob_all_declinations_estimate(precision=thresholds[t], threshold=1)
                    # text_str = f'P={prob:.2g}'
                    text_str = f"1/{int(np.round(1 / prob, -1))}"
                    ax.text(
                        periods[i, j],
                        peak_mags[i, j],
                        text_str,
                        va=vertical,
                        ha=horizontal,
                        color="k",
                        bbox=dict(facecolor=colors[i], alpha=0.8),
                    )
                    t += 1

            ax.plot(
                periods[i],
                peak_mags[i],
                "-",
                markersize=10,
                marker=styles[i],
                color=colors[i],
                label=f"WD-WD ({sim.star_mass}$M_\u2609$, {sim.lens_mass}$M_\u2609$)",
            )

            idx = roche[i] < 1
            ax.plot(periods[i, idx], peak_mags[i, idx], "bo", fillstyle="none", markersize=12)

        # NS and BH lenses
        orbits = np.geomspace(0.0001, 1, 60)
        masses = np.array([1.5, 3, 10.0, 30.0])
        colors = [np.ones(3) * c for c in [0.6, 0.4, 0.2, 0.0]]
        styles = ["X", "D", "o", "^"]
        peak_mags = np.zeros((len(masses), len(orbits)))
        periods = np.zeros(peak_mags.shape)
        prob_low = np.zeros(peak_mags.shape)
        prob_high = np.zeros(peak_mags.shape)
        roche = np.zeros(peak_mags.shape)

        for i, m in enumerate(masses):
            t = 0
            sim.lens_mass = m
            for j, a in enumerate(orbits):
                sim.semimajor_axis = a
                mag = sim.calculate()
                periods[i, j] = sim.orbital_period * 3600
                peak_mags[i, j] = np.max(mag - 1)
                prob_low[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[0], threshold=1)
                prob_high[i, j] = sim.best_prob_all_declinations_estimate(precision=thresholds[1], threshold=1)
                roche[i, j] = sim.roche_lobe / sim.star_size

                if peak_mags[i, j] > thresholds[t]:
                    prob = sim.best_prob_all_declinations_estimate(precision=thresholds[t], threshold=1)
                    # text_str = f'P={prob:.2g}'
                    text_str = f"1/{int(np.round(1 / prob))}"
                    ax.text(
                        periods[i, j],
                        peak_mags[i, j],
                        text_str,
                        ha="right",
                        color="k",
                        bbox=dict(facecolor="w", alpha=0.8),
                    )

                    t += 1

            ax.plot(
                periods[i],
                peak_mags[i],
                "-",
                markersize=10,
                marker=styles[i],
                color=colors[i],
                label=f"WD-{sim.lens_type} ({sim.star_mass}$M_\u2609$, {sim.lens_mass}$M_\u2609$)",
            )

            idx = roche[i] < 1
            ax.plot(periods[i, idx], peak_mags[i, idx], "bo", fillstyle="none", markersize=12)

        ax.set_xlabel("Orbital period [s]")
        ax.set_ylabel("Max magnification")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot([0], [0], "bo", markersize=12, fillstyle="none", label="Roche lobe overflow")

        ax_min = 2.5e-3
        ax_max = 50

        ax.set_ylim(ax_min, ax_max)
        ax.set_xlim(0.45, 1.5e8)

        ax.plot(
            ax.get_xlim(),
            np.ones(2) * thresholds[0],
            "-.k",
            label=f"{int(thresholds[0] * 1000)}mmag threshold",
            alpha=0.5,
        )
        ax.plot(
            ax.get_xlim(),
            np.ones(2) * thresholds[1],
            "--k",
            label=f"{int(thresholds[1] * 1000)}mmag threshold",
            alpha=0.5,
        )

        ax.plot(np.ones(2) * 60, ax.get_ylim(), ":k", alpha=0.3)
        ax.text(60 * 1.1, ax_max / 2, "one minute")
        ax.plot(np.ones(2) * 3600, ax.get_ylim(), ":k", alpha=0.3)
        ax.text(3600 * 1.1, ax_max / 2, "one hour")
        ax.plot(np.ones(2) * 3600 * 24, ax.get_ylim(), ":k", alpha=0.3)
        ax.text(3600 * 24 * 1.1, ax_max / 2, "one day")
        ax.plot(np.ones(2) * 3600 * 24 * 365, ax.get_ylim(), ":k", alpha=0.3)
        ax.text(3600 * 24 * 365 * 1.1, ax_max / 2, "one year")

        ax.plot([0], [0], linestyle="none", label='"1/10": Duty cycle')

        # ax.legend(loc="lower right")
        # ax.set_position([0.08, 0.08, 0.64, 0.9])
        # ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
        ax.legend(fontsize=20)

        ## save the plot

        plt.savefig("plots/mag_vs_period.pdf")
        plt.savefig("plots/mag_vs_period.png")

    finally:
        matplotlib.rcParams["font.size"] = original_font_size


def test_plot_survey_precisions():
    from src.survey import Survey

    plt.rc("font", size=24)

    survey_names = ["ZTF", "TESS", "LSST", "DECAM", "CURIOS", "LAST"]
    # markers = ['x', 'v', '*', '^', 's']
    line_styles = ["-", ":", "--", "-."]
    colors = ["k", "r", "g", "b"]

    fig = plt.figure(num=5, figsize=[12, 6])
    plt.clf()
    axes = fig.add_axes([0.2, 0.14, 0.78, 0.82])

    # fig = plt.figure(num=5).clf()
    # fig, axes = plt.subplots(num=5, figsize=[18, 12])

    for i, s in enumerate(survey_names):
        new_survey = Survey(s)
        if new_survey.prec_list is not None:
            m = new_survey.mag_list
            p = new_survey.prec_list
        else:
            m = np.linspace(12, new_survey.limmag)
            p = np.ones(m.shape) * new_survey.precision
        line_style = line_styles[i // len(line_styles)]
        color = colors[i % len(colors)]
        axes.plot(m, p, ls=line_style, color=color, label=s, lw=4.0)

    axes.set_xlabel("Magnitude")
    axes.set_ylabel("Photometric precision")
    axes.set_yscale("log")
    axes.grid(axis="y", linestyle="--", linewidth=0.8, alpha=1.0)

    # labels = [item.get_text() for item in axes.get_yticklabels()]
    labels = []
    for value in axes.get_yticks():
        if value < 0.001:
            labels.append(f"{value*1000:.1f}mmag")
        elif value < 0.01:
            labels.append(f"{int(value*1000)}mmag")
        else:
            labels.append(f"{int(value*100)}%")

    axes.set_yticklabels(labels)

    axes.legend(loc="lower right", fontsize=20, ncol=2, framealpha=0)

    # save the plot
    plt.savefig("plots/photometric_precision.pdf")
    plt.savefig("plots/photometric_precision.png")


def test_make_duration_declination_plots(wd_ds):
    wd_ds.flare_duration.sel(
        lens_mass=0.6,
        star_mass=0.6,
        star_temp=8000,
        lens_temp=8000,
        survey=["TESS", "LSST", "CURIOS"],
    ).isel(semimajor_axis=[50, 65], declination=slice(0, 120)).plot(
        col="semimajor_axis", hue="survey", marker="*", figsize=[14, 6]
    )

    plt.xscale("log")

    # save the plot
    plt.savefig("plots/duration_vs_declination.pdf")
    plt.savefig("plots/duration_vs_declination.png")


def test_plot_effective_volume_wd():
    import matplotlib

    font_size = 20
    matplotlib.rcParams.update({"font.size": font_size})

    fig, axes = plt.subplots(num=6, figsize=[12, 8])

    surveys = ["TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]
    # surveys = surveys[:2]  # debug only
    markers = ["X", "v", "*", "^", "s", "D", "P"]

    for i, s in enumerate(surveys):
        ds = fetch_distributions(s, obj="WD", space_density=1 / 1818, verbose=False, clear_cache=CLEAR_CACHE)
        sma = ds.semimajor_axis.values
        ev = ds.ev_sma.values
        p = axes.plot(sma[::2], ev[::2], label=s, linestyle=None, marker=markers[i], linewidth=0, markersize=8)
        axes.plot(sma, ev, "-", linewidth=2.0, color=p[0].get_color())

    # axes.set_ylim((1e-3, 3e6))
    # axes.set_xlim((8e-3, 15))
    axes.set_yscale("log")
    axes.set_xscale("log")

    axes.set_xlabel("Semimajor axis [AU]", fontsize=font_size)
    axes.set_ylabel("Effective volume [pc$^3$]", fontsize=font_size)

    # Kepler's law using two equal 0.6 solar mass WDs
    period = lambda x: x ** (3 / 2) * 365.25 / np.sqrt(1.2)
    ax2 = axes.twiny()
    ax2.set_xlabel("Period [days]", fontsize=font_size)
    ax2.set_xlim([period(x) for x in axes.get_xlim()])
    ax2.set_xscale("log")

    # add the model density
    axes.plot(sma, ds.volume_sma, "--k", label="WD-WD density", lw=2.5)

    lgnd = axes.legend(loc="lower center", fontsize=font_size, framealpha=0, ncol=2)
    for lh in lgnd.legendHandles:
        lh.set_markersize(12.0)
        lh.set_linewidth(2.0)

    # following https://stackoverflow.com/questions/44078409/matplotlib-semi-log-plot-minor-tick-marks-are-gone-when-range-is-large
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    axes.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    axes.yaxis.set_minor_locator(locmin)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.show()

    # save the plot
    plt.savefig("plots/effective_volume_WD.pdf")
    plt.savefig("plots/effective_volume_WD.png")


def test_plot_effective_volume_bh():
    import matplotlib

    font_size = 20
    matplotlib.rcParams.update({"font.size": font_size})

    fig, axes = plt.subplots(num=7, figsize=[12, 8])

    surveys = ["TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]
    # surveys = surveys[:2]  # debug only
    markers = ["X", "v", "*", "^", "s", "D", "P"]

    for i, s in enumerate(surveys):
        ds = fetch_distributions(s, obj="BH", space_density=10**-5, verbose=False, clear_cache=CLEAR_CACHE)
        sma = ds.semimajor_axis.values
        ev = ds.ev_sma.values
        p = axes.plot(sma[::2], ev[::2], label=s, linestyle=None, marker=markers[i], linewidth=0, markersize=8)
        axes.plot(sma, ev, "-", linewidth=2.0, color=p[0].get_color())

    # axes.set_ylim((1e-3, 3e6))
    # axes.set_xlim((8e-3, 15))
    axes.set_yscale("log")
    axes.set_xscale("log")

    axes.set_xlabel("Semimajor axis [AU]", fontsize=font_size)
    axes.set_ylabel("Effective volume [pc$^3$]", fontsize=font_size)

    # minutes (using Kepler's law) assuming a 10 solar mass BH with a 0.6 WD
    period = lambda x: x ** (3 / 2) * 365.25 * 24 * 60 / np.sqrt(10.6)
    ax2 = axes.twiny()
    ax2.set_xlabel("Period [minutes]", fontsize=font_size)
    ax2.set_xlim([period(x) for x in axes.get_xlim()])
    ax2.set_xscale("log")

    # add the model density
    axes.plot(sma, ds.volume_sma, "--k", label="WD-NS/BH density", lw=2.5)

    lgnd = axes.legend(loc="lower right", fontsize=font_size, framealpha=0, ncol=2)
    for lh in lgnd.legendHandles:
        lh.set_markersize(12.0)
        lh.set_linewidth(2.0)

    # following https://stackoverflow.com/questions/44078409/matplotlib-semi-log-plot-minor-tick-marks-are-gone-when-range-is-large
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    axes.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    axes.yaxis.set_minor_locator(locmin)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.show()

    # save the plot
    plt.savefig("plots/effective_volume_BH.pdf")
    plt.savefig("plots/effective_volume_BH.png")


@pytest.mark.skip(reason="This plot is replaced by the effective volume plot.")
def test_plot_sensitivity_and_model(wd_ds):
    fig, axes = plt.subplots(num=10, figsize=[10, 6])

    surveys = ["TESS", "CURIOS", "LSST", "CURIOS_ARRAY"]
    markers = ["x", "v", "*", "^", "s"]
    sens = 1 / (wd_ds.marginalized_volume * wd_ds.probability_density_flat).sum(
        dim=["lens_mass", "star_mass", "lens_temp", "star_temp"]
    )

    sma = sens.semimajor_axis.values

    sens_curve = {}
    min_sma = np.inf
    for i, s in enumerate(surveys):
        sens_curve[s] = sens.sel(survey=s).values
        min_sma = min(min_sma, sma[np.argmin(np.isinf(sens_curve[s]))])
        axes.plot(sma, sens_curve[s], label=s, marker=markers[i])

    g = Grid()
    g.dataset = wd_ds

    space_density = 0.0055  # WDs per pc^3
    binary_fraction = 0.1
    prob = {}
    for m in ["mid", "low", "high"]:
        model = g.get_default_probability_density(temp=m, mass=m, sma=m).sum(
            dim=[
                "star_mass",
                "lens_mass",
                "star_temp",
                "lens_temp",
            ]
        )
        prob[m] = model.values * space_density * binary_fraction

    axes.fill_between(sma, prob["low"], prob["high"], color="k", alpha=0.3, label=f"WD model")
    # axes.plot(sma, prob['mid'], color='k', label=f'{mass}M$_\odot$, {temp}$^\circ$K')
    axes.plot(sma, prob["mid"], color="k")

    axes.legend(loc="upper right", fontsize=14, framealpha=0)

    axes.set_xlim((min_sma / 2, 0.4))
    # plt.ylim((float(sens.min() * 0.5), max(prob['mid']) * 1e2))

    axes.set_yscale("log")
    axes.set_xscale("log")

    axes.set_xlabel("Semimajor axis [AU]", fontsize=14)
    axes.set_ylabel("WD binary density [pc$^{-3}$]", fontsize=14)

    period = lambda x: x ** (3 / 2) * 365.25
    ax2 = axes.twiny()
    ax2.set_xlabel("Period [days]", fontsize=14)
    ax2.set_xlim([period(x) for x in axes.get_xlim()])
    ax2.set_xscale("log")

    plt.show()

    # save the plot
    plt.savefig("plots/sensitivity_vs_model.pdf")
    plt.savefig("plots/sensitivity_vs_model.png")


@pytest.mark.skip(reason="This plot is replaced by the new effective volume plot.")
def test_plot_effective_volume_wd_old_version(wd_ds):
    import matplotlib

    font_size = 20
    matplotlib.rcParams.update({"font.size": font_size})

    g = Grid()
    g.dataset = wd_ds
    g.marginalize_declinations()
    prob_flat = g.get_probability_density(lens_temp=-2, star_temp=-2, lens_mass=(0.6, 0.2), star_mass=(0.6, 0.2))
    g.dataset["probability_density_flat"] = prob_flat

    prob = g.get_default_probability_density()
    total_vol = float(g.get_total_volume())
    print(f"total volume: {total_vol:.1f}pc^3")

    fig, axes = plt.subplots(num=11, figsize=[12, 8])
    fig.clf()
    fig, axes = plt.subplots(num=11, figsize=[12, 8])

    surveys = ["ZTF", "TESS", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]
    markers = ["X", "v", "*", "^", "s", "D", "P"]
    ev = (wd_ds.marginalized_volume * wd_ds.probability_density).sum(
        dim=["lens_mass", "star_mass", "lens_temp", "star_temp"]
    )
    sma = ev.semimajor_axis.values

    ev_curve = {}
    min_sma = np.inf
    for i, s in enumerate(surveys):
        ev_curve[s] = ev.sel(survey=s).values
        min_sma = min(min_sma, sma[np.argmin(ev_curve[s] > 0)])
        axes.plot(sma, ev_curve[s], label=s, marker=markers[i], linewidth=2.0, markersize=8)

    axes.set_ylim((1e-3, 3e6))
    axes.set_xlim((8e-3, 15))
    axes.set_yscale("log")
    axes.set_xscale("log")

    axes.set_xlabel("Semimajor axis [AU]", fontsize=font_size)
    axes.set_ylabel("effective volume [pc$^3$]", fontsize=font_size)

    # Kepler's law using two equal 0.6 solar mass WDs
    period = lambda x: x ** (3 / 2) * 365.25 / np.sqrt(1.2)
    ax2 = axes.twiny()
    ax2.set_xlabel("Period [days]", fontsize=font_size)
    ax2.set_xlim([period(x) for x in axes.get_xlim()])
    ax2.set_xscale("log")

    space_density = 0.0055
    binary_fraction = 0.1

    need_volume = 1 / (space_density * binary_fraction)
    axes.plot(
        sma,
        np.ones(sma.shape) * need_volume,
        "--k",
        label=f"DWD/{int(need_volume)} pc$^3$",
        lw=2.5,
    )
    axes.legend(loc="upper right", fontsize=font_size, framealpha=0, ncol=2)

    # following https://stackoverflow.com/questions/44078409/matplotlib-semi-log-plot-minor-tick-marks-are-gone-when-range-is-large
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    axes.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    axes.yaxis.set_minor_locator(locmin)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.show()

    # save the plot
    plt.savefig("plots/effective_volume_WD.pdf")
    plt.savefig("plots/effective_volume_WD.png")


@pytest.mark.skip(reason="This plot is replaced by the new effective volume plot.")
def test_plot_effective_volume_bh_old_version(bh_ds):
    import matplotlib

    font_size = 20
    matplotlib.rcParams.update({"font.size": font_size})

    g = Grid()
    g.dataset = bh_ds
    g.marginalize_declinations()
    prob_flat = g.get_probability_density(lens_temp=-2, star_temp=-2, lens_mass=-2, star_mass=(0.6, 0.2))
    g.dataset["probability_density_flat"] = prob_flat

    # prob = g.get_default_probability_density()
    # print(bh_ds.probability_density)
    # total_vol = float(g.get_total_volume())
    # print(f"total volume: {total_vol:.1f}pc^3")

    fig, axes = plt.subplots(num=11, figsize=[12, 8])
    fig.clf()
    fig, axes = plt.subplots(num=11, figsize=[12, 8])

    surveys = ["TESS", "ZTF", "LSST", "DECAM", "CURIOS", "CURIOS_ARRAY", "LAST"]
    markers = ["X", "v", "*", "^", "s", "D", "P"]
    ev = (bh_ds.marginalized_volume * bh_ds.probability_density_flat).sum(
        dim=["lens_mass", "star_mass", "lens_temp", "star_temp"]
    )
    sma = ev.semimajor_axis.values

    ev_curve = {}
    min_sma = np.inf
    for i, s in enumerate(surveys):
        ev_curve[s] = ev.sel(survey=s).values
        min_sma = min(min_sma, sma[np.argmin(ev_curve[s] > 0)])
        axes.plot(sma, ev_curve[s], label=s, marker=markers[i], linewidth=2.0, markersize=8)

    axes.set_ylim((1e-6, 1e8))
    axes.set_xlim((8e-5, 1.5))

    axes.set_yscale("log")
    axes.set_xscale("log")

    axes.set_xlabel("Semimajor axis [AU]", fontsize=font_size)
    axes.set_ylabel("effective volume [pc$^3$]", fontsize=font_size)

    # minutes (using Kepler's law) assuming a 10 solar mass BH with a 0.6 WD
    period = lambda x: x ** (3 / 2) * 365.25 * 24 * 60 / np.sqrt(10.6)
    ax2 = axes.twiny()
    ax2.set_xlabel("Period [minutes]", fontsize=font_size)
    ax2.set_xlim([period(x) for x in axes.get_xlim()])
    ax2.set_xscale("log")

    # following https://stackoverflow.com/questions/44078409/matplotlib-semi-log-plot-minor-tick-marks-are-gone-when-range-is-large
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    axes.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    axes.yaxis.set_minor_locator(locmin)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax3 = axes.twinx()
    ax3.set_ylabel("Flare FWHM [seconds]", fontsize=font_size)
    # ax3.set_xlim([period(x) for x in axes.get_xlim()])
    ax3.set_yscale("log")

    # fwhm1 = ds.fwhm.sel(star_mass=0.6).isel(lens_mass=0, declination=0, lens_temp=0, star_temp=0, survey=0)
    # fwhm2 = ds.fwhm.sel(star_mass=0.6).isel(lens_mass=-1, declination=0, lens_temp=0, star_temp=0, survey=0)
    s_low = Simulator()
    s_high = Simulator()

    fwhm1 = []
    fwhm2 = []
    for a in sma:
        s_low.timestamps = None
        s_low.calculate(
            star_mass=0.6,
            lens_mass=min(bh_ds.lens_mass.values),
            semimajor_axis=a,
            declination=0,
            star_temp=8000,
            lens_temp=0,
        )
        fwhm1.append(s_low.fwhm)

        s_high.timestamps = None
        s_high.calculate(
            star_mass=0.6,
            lens_mass=max(bh_ds.lens_mass.values),
            semimajor_axis=a,
            declination=0,
            star_temp=8000,
            lens_temp=0,
        )
        fwhm2.append(s_high.fwhm)

    ax3.fill_between(sma, fwhm1, fwhm2, color="k", alpha=0.3)

    axes.fill_between([np.nan, np.nan], [np.nan, np.nan], color="k", alpha=0.3, label="FWHM range")
    axes.legend(loc="lower right", fontsize=font_size, framealpha=0, ncols=2)

    plt.show()

    # save the plot
    plt.savefig("plots/effective_volume_BH.pdf")
    plt.savefig("plots/effective_volume_BH.png")


def test_plot_detections_per_mass_index():
    import matplotlib

    font_size = 20
    matplotlib.rcParams.update({"font.size": font_size})

    survey = "LSST"
    mass_indices = ["low", "mid", "high"]
    markers = ["X", "v", "s"]
    styles = ["-", "-.", "--"]

    fig, axes = plt.subplots(num=8, figsize=[12, 8])
    det = []
    for i, mass in enumerate(mass_indices):
        ds = fetch_distributions(survey, obj="BH", space_density=1e-5, mass_index=mass, clear_cache=CLEAR_CACHE)
        sma = ds.semimajor_axis.values
        det.append(ds.detections_sma.values)
        # p = axes.plot(sma, det, linestyle=None, marker=markers[i], label=mass)
        # axes.plot(sma, det, '-', linewidth=2.0, color=p[0].get_color())
        axes.plot(sma, det[-1], linestyle=styles[i], label=mass, linewidth=2.0)

        axes.set_ylim((1e-2, 2e2))
        axes.set_xlim((8e-4, 2))
        axes.set_yscale("log")
        axes.set_xscale("log")
        axes.set_xlabel("Semimajor axis [AU]", fontsize=font_size)
        axes.set_ylabel("Number of detections", fontsize=font_size)

    # axes.plot(sma, det[2]/det[0])

    lgnd = axes.legend(loc="upper right", fontsize=font_size, framealpha=0)
    for lh in lgnd.legendHandles:
        lh.set_markersize(12.0)
        lh.set_linewidth(2.0)

    plt.show()

    # save the plot
    plt.savefig("plots/detections_mass_index.pdf")
    plt.savefig("plots/detections_mass_index.png")


def test_plot_effective_volume_per_mass():
    import matplotlib

    font_size = 20
    matplotlib.rcParams.update({"font.size": font_size})

    # surveys = ["TESS", "ZTF", "LSST"]
    surveys = ["LSST", "ZTF", "TESS"]
    markers = ["X", "v", "*", "^", "s", "D", "P"]

    fig, ax = plt.subplots(1, 1, num=9, figsize=[12, 8])
    fig.subplots_adjust(wspace=0)

    for i, survey in enumerate(surveys):
        ds = fetch_distributions(survey, obj="BH", space_density=1e-5, clear_cache=CLEAR_CACHE)
        mass = ds.lens_mass.values
        ev = ds.ev_mass.values
        ax.plot(mass, ev, linewidth=0, marker=markers[i])

        # fit to a sqrt
        # forcing an exact match to sqrt
        # adjusted = ev / np.sqrt(mass)
        # coeff = np.mean(adjusted)
        # ev_fit = coeff * np.sqrt(mass)
        # coeff_power = int(np.floor(np.log10(coeff)))
        # coeff_number = f"{coeff / 10 ** coeff_power:.2f}"
        # coeff_power = str(coeff_power)
        # coeff_power = r"$\times 10^{" + coeff_power + "}$"
        # tag = "V= " + coeff_number + coeff_power + r"${M_L}^{1/2}$"

        # using a log fit (assumes approximate sqrt dependence)
        # p = np.polyfit(np.log10(mass), np.log10(ev), 1)
        # ev_fit = 10 ** (p[0] * np.log10(mass) + p[1])
        # tag = f'{p[0]:.2f}'
        # tag = '*M^{' + tag + '}$'
        # tag = f'V= ${10 ** p[1]:.1e}' + tag

        # using a polynomial with rank 2
        # p = np.polyfit(mass[20:], ev[20:], 2)

        # using a linear fit but only after some value
        fit_mass = 10
        idx = np.argmin(abs(mass - 10))
        p = np.polyfit(mass[idx:], ev[idx:], 1)
        ev_fit = np.poly1d(p)(mass)

        # render the fit results
        configure(exp_format="cdot", help_text=False)
        p2 = p.copy()
        p2 = p2[::-1]

        tag = "V= "
        for j, coeff in enumerate(p2):
            if coeff > 0 and j > 0:
                tag += "+"
            tag += f"${num2tex(coeff, precision=2)}$"
            if j > 0:
                tag += r"$\times M$"
                if j > 1:
                    tag += f"$^{j}$"

        tag = f"{survey +':':5s} {tag}"
        ax.plot(mass[idx:], ev_fit[idx:], "-", linewidth=2.0, color=ax.lines[-1].get_color(), label=tag)
        ax.plot(mass[: idx + 1], ev_fit[: idx + 1], "--", linewidth=1.5, color=ax.lines[-1].get_color())

        ax.set_ylim((2e4, 1e9))
        ax.set_yscale("log")
        # ax.set_xscale("log")
        ax.set_xlabel(r"Lens mass [M$_\odot$]", fontsize=font_size)
        ax.set_ylabel(r"Effective volume [pc$^3$]", fontsize=font_size)

    lgnd = ax.legend(loc="lower right", fontsize=font_size - 2, framealpha=0)

    for i, h in enumerate(lgnd.legendHandles):
        h.set_markersize(12.0)
        h.set_marker(markers[i])

    plt.show()

    # save the plot
    plt.savefig("plots/effective_volume_mass.pdf")
    plt.savefig("plots/effective_volume_mass.png")


def test_plot_number_detections_vs_sma():
    import matplotlib

    font_size = 20
    matplotlib.rcParams.update({"font.size": font_size})

    surveys = ["TESS", "ZTF", "LSST"]
    markers = ["X", "v", "*", "^", "s", "D", "P"]

    densities = {"WD": 1 / 1818, "BH": 1e-5}

    fig, axes = plt.subplots(1, 2, num=10, figsize=[12, 8])
    fig.subplots_adjust(wspace=0)
    lgnd = []
    for j, obj in enumerate(["WD", "BH"]):
        for i, survey in enumerate(surveys):
            ds = fetch_distributions(
                survey, obj=obj, space_density=densities[obj], verbose=False, clear_cache=CLEAR_CACHE
            )
            sma = ds.semimajor_axis.values
            det = ds.detections_sma.values
            p = axes[j].plot(sma, det, "-", linewidth=2.0, marker=markers[i], label=survey)
            det_f = ds.detections_sma_followup.values
            axes[j].plot(sma, det_f, "--", linewidth=2.0, color=p[0].get_color())

            axes[j].set_ylim((1e-6, 2e2))
            axes[j].set_yscale("log")
            axes[j].set_xscale("log")
            axes[j].set_xlabel("Semimajor axis [AU]", fontsize=font_size)
            axes[j].set_ylabel("Number of detections", fontsize=font_size)

        lgnd.append(axes[j].legend(loc="lower left", fontsize=font_size, framealpha=0))
        for lh in lgnd[-1].legendHandles:
            lh.set_markersize(12.0)
            lh.set_linewidth(2.0)

    axes[0].get_xaxis().set_ticks([1e-2, 1e-1, 1])
    axes[0].set_title("WD-WD")

    axes[1].get_yaxis().set_ticks([])
    axes[1].set_ylabel(None)
    ax2 = axes[1].twinx()
    ax2.set_ylim(axes[1].get_ylim())
    ax2.set_yscale("log")
    lgnd[1].set(loc="lower center")

    axes[1].set_title("WD-NS/BH")

    plt.show()

    # save the plot
    plt.savefig("plots/detection_numbers.pdf")
    plt.savefig("plots/detection_numbers.png")


def test_plot_semimajor_axis_population():
    fig, axes = plt.subplots(num=11, figsize=[8, 6])

    space_density = 0.0055
    binary_fraction = 0.1
    sma = np.geomspace(1e-4, 50, 1000)

    # g = Grid()
    # g.setup_demo_scan(wd_lens=True)
    # g.star_masses = [0.6]
    # number = g.semimajor_axis_distribution(alpha=-1.3, sma=sma)
    number = semimajor_axis_distribution(sma=sma, star_masses=[0.6], lens_masses=[0.6], alpha=-1.3)
    number = number.reshape(len(sma))
    number = number / np.sum(number)

    number *= space_density * binary_fraction

    axes.plot(sma, number, lw=3, label="total number of binary WDs")

    # add a second plot with the duty cycle weighed number of WDs
    constants = 220961382691907.84  # = G*M_sun*AU/c^2 = 6.67408e-11 * 1.989e30 * 1.496e11 / 299792458 ** 2
    r = np.sqrt(4 * constants * 0.6 * sma) / 1.496e11  # einstein radius in AU

    duty_cycle = r / sma  # in either space or time (total prob is this squared)

    number2 = number * duty_cycle

    axes.plot(
        sma,
        number2,
        lw=3,
        label="with geometry and constant monitoring",
    )

    num_observations = 50  # number of observations on each target (estimate)
    number3 = number * duty_cycle**2 * num_observations
    number3 = np.minimum(number2, number3)

    axes.plot(
        sma,
        number3,
        lw=3,
        ls="--",
        color=axes.lines[-1].get_color(),
        label=f"with geometry $\\times$ duty cycle $\\times$ {num_observations} visits",
    )

    num_observations = 1000  # number of observations on each target (estimate)
    number4 = number * duty_cycle**2 * num_observations
    number4 = np.minimum(number2, number4)

    axes.plot(
        sma,
        number4,
        lw=3,
        ls=":",
        color=axes.lines[-1].get_color(),
        label=f"with geometry $\\times$ duty cycle $\\times$ {num_observations} visits",
    )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Semimajor axis [AU]", fontsize=16)
    axes.set_ylabel("Number of DWDs per pc$^3$", fontsize=16)
    axes.tick_params(axis="x", labelsize=12)
    axes.tick_params(axis="y", labelsize=12)

    # days (using Kepler's law) assuming a 0.6 solar mass WD with a 0.6 WD
    period = lambda x: x ** (3 / 2) * 365.25 / np.sqrt(1.2)
    ax2 = axes.twiny()
    ax2.set_xlabel("Period [days]", fontsize=14)
    ax2.set_xlim([period(x) for x in axes.get_xlim()])
    ax2.tick_params(axis="x", labelsize=12)
    ax2.set_xscale("log")

    ax3 = axes.twinx()
    ax3.set_ylabel("Fraction of all DWDs", fontsize=14)
    ax3.set_ylim([y / space_density for y in axes.get_ylim()])
    ax3.tick_params(axis="y", labelsize=12)
    ax3.set_yscale("log")

    axes.legend(loc="lower left", fontsize=14, framealpha=0)

    plt.show()

    # save the plot
    plt.savefig("plots/semimajor_axis_population.pdf")
    plt.savefig("plots/semimajor_axis_population.png")
