import glob
import os
import re
import itertools
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from numpy import dtype, ndarray, float64
from pandas import DataFrame
from scipy.integrate import quad
from scipy.spatial import cKDTree
from sklearn import linear_model, ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from doubleml import DoubleMLData
from doubleml import DoubleMLPLR, DoubleMLIRM

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colorbar as plt_cbar
from matplotlib import colors
from matplotlib.patches import Rectangle
import matplotlib
from plotly import graph_objects as go

import cProfile


plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "sans-serif",      # Use sans serif font family
    "font.serif": ["Computer Modern"],# Match default LaTeX font
    "font.sans-serif": ["Latin Modern"],
    "axes.labelsize": 14,             # Font size for axis labels
    "font.size": 13,                  # General font size
    "legend.fontsize": 11,             # Legend font size
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
        "ytick.direction": "out",
    "xtick.direction": "out",
    'figure.constrained_layout.use': True,
    "legend.frameon":    False,
    "figure.dpi": 200
})
temp_df = pd.read_csv(r"temp_input\Tpeak_tconv_values\temp_input_values.txt", dtype=float, delimiter=" ", comment="#")
temp_idx_df = temp_df.set_index(["T_lim", "T_peak", "t_conv"])


def hatch_nan_cells(ax, values, hatch="////", edgecolor="white"):
    nan_rows, nan_cols = np.where(np.isnan(values))
    for row, col in zip(nan_rows, nan_cols):
        ax.add_patch(Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            facecolor="none",
            edgecolor=edgecolor,
            hatch=hatch,
            linewidth=0.0,
            antialiased=False,
            zorder=3,
        ))

def set_plot_size(width_type, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    match width_type:
        case "article":
            width = 418.25
        case "thesis":
            width = 398.33864
        case "report":
            width = 345.0
        case "prb":
            width = (3+3/8) * 72.27
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2 * 1

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * np.sqrt(subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def forcing_function(T_0, mu_0, mu_1, T_lim, R):
    """Returns forcing functions (here just a parametric temperature curve)

    Args:
        T_0 (float): _description_
        mu_0 (float): _description_
        mu_1 (float): _description_
        T_lim (float): _description_
        R (float): _description_

    Returns:
        callable: forcing function, scalar
    """
    y = R + mu_0*(T_0-T_lim)
    f = lambda t: (T_0 + y*t - (1 - np.exp(-(mu_0+mu_1*t)*t))*(y*t - (T_lim - T_0)))
    return f

def calculate_clean_ticks(vmin, vmax, target_ticks=5):
    """Calculate clean tick positions between vmin and vmax. Now that its hand written it actually works"""
    bounds = np.asarray([vmin, vmax], dtype=float)
    if not np.all(np.isfinite(bounds)):
        return np.array([])
    vmin, vmax = np.sort(bounds)
    if np.isclose(vmin, vmax):
        return np.array([vmin])

    data_range = vmax - vmin
    acceptable_intervals = [0.1, 0.2, 0.25, 0.5, 1]
    preferred_interval = acceptable_intervals[0]
    for interval in acceptable_intervals:
        if interval >= data_range:
            break
        if np.ceil(data_range / interval) >= target_ticks:
            preferred_interval = interval
    start = np.ceil(vmin / preferred_interval) * preferred_interval
    all_ticks = np.round(np.array([vmin, *np.arange(start, vmax, preferred_interval), vmax]), 2)
    all_ticks = all_ticks[np.isfinite(all_ticks)]
    all_ticks = np.unique(all_ticks)
    if len(all_ticks) > 2 and all_ticks[1] - vmin < preferred_interval / 4:
        all_ticks = np.delete(all_ticks, 1)
    if len(all_ticks) > 2 and vmax - all_ticks[-2] < preferred_interval / 4:
        all_ticks = np.delete(all_ticks, -2)
    return all_ticks

def imshow_grid(
    matrices,
    titles,
    x_ticks=None,
    x_ticklabels=None,
    y_ticks=None,
    y_ticklabels=None,
    nrows=1,
    ncols=1,
    figsize=(6, 4),
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    cbar_axis=None):
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    vmin = min(np.min(matrix) for matrix in matrices[:nrows * ncols])
    vmax = max(np.max(matrix) for matrix in matrices[:nrows * ncols])
    axes = [axes] if isinstance(axes, Axes) else list(axes.flat)

    diverging_map = "inferno"  # shiftedColorMap(plt.cm.coolwarm, 0, 1 - vmax / (vmax + abs(vmin)), 1)
    limit = max(np.abs(vmin), np.abs(vmax))
    divnorm = colors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            # vmin=vmin,
            # vmax=vmax, #TODO ugly hack
            cmap=diverging_map,
            norm=divnorm,
            #    interpolation="bilinear"
        )
        # ax.set_title(title, loc='left', fontsize='small')

    if xlabel:
        fig.supxlabel(xlabel)
    if ylabel:
        fig.supylabel(ylabel)

    if x_ticks is not None:
        for ax in axes:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)

    if y_ticks is not None:
        axes[0].set_yticks(y_ticks)
        axes[0].set_yticklabels(y_ticklabels)
    cbar = fig.colorbar(im, ax=axes[:cbar_axis] if cbar_axis else axes[:len(axes) // 2])

    y_ticks = calculate_clean_ticks(vmin, vmax) # AI developed
    cbar.ax.set_yticks(y_ticks)
    cbar.ax.set_ylim(vmin, vmax)
    cbar.ax.set_yticklabels([f'{tick:.2g}' for tick in y_ticks])
    if cbar_label:
        cbar.set_label(cbar_label, fontsize="small")

        return fig, axes

def prepare_impact_plot(impact_df, T_lim=None):
    # avg_impact = impact_df.xs(0.5, level="strength").droplevel("integral")
    Tlim_options = impact_df.index.get_level_values("T_lim").unique()
    if not T_lim:
        T_lim = Tlim_options
    tconv = (
        impact_df.index
        .get_level_values("t_conv")
        .unique()
        .sort_values()
    )

    matrices = []
    titles = []

    for tlim in T_lim:
        df_slice = impact_df.xs(tlim, level="T_lim")
        mat = (
            df_slice
            .unstack("t_conv")
            .sort_index()
            .values
        )
        matrices.append(mat)
        decimal = int(tlim-int(tlim)!=0)
        titles.append(fr"$T_{{\mathrm{{lim}}}}={tlim:.{decimal}f}^\circ C$")

    Tpeak = (
        df_slice.index
        .get_level_values("T_peak")
        .unique()
        .sort_values()
    )
    x_ticks = np.int16(np.linspace(0, len(tconv)-1, 4, endpoint=True))
    y_ticks = np.int16(np.linspace(0, len(Tpeak)-1, 5, endpoint=True))
    return {
        "matrices": matrices,
        "titles": titles,
        "x_ticks": x_ticks,
        "x_ticklabels": np.int16(tconv.round(0))[x_ticks],
        "y_ticks": y_ticks,
        "y_ticklabels": Tpeak.round(1)[y_ticks],
    }

def prepare_tipping_plot(df, components):
    if not isinstance(components, list):
        components = list(components)

    todrop = [
        name for name in df.index.names
        if name not in ["integral", "strength", "component"]
    ]
    df_mod = df.droplevel(todrop)

    matrices = []
    titles = []

    for comp in components:
        df_c = df_mod.xs(comp, level="component")
        mat = (
            df_c
            .unstack("integral")
            .sort_index()
            .values
        )
        matrices.append(mat)
        titles.append(comp)

    strength = (
        df_mod.index
        .get_level_values("strength")
        .unique()
        .sort_values()
    )

    return {
        "matrices": matrices,
        "titles": titles,
        "y_ticks": np.arange(len(strength)),
        "y_ticklabels": strength.round(2),
        "vmin": df_mod.loc[(slice(None), slice(None), components)].min(),
        "vmax": df_mod.loc[(slice(None), slice(None), components)].max(),
    }

def read_files(suffix, time):
    filenames = glob.glob(f"../numerical_data/results/no_feedbacks/network_1.0_1.0_1.0/*{suffix}.npy")
    n_temps = len(filenames)
    results_array = np.zeros((n_temps, *np.load(filenames[0]).shape))
    temperature_props = np.zeros((n_temps, 4))
    for i, filename in enumerate(filenames):
        data = np.load(filename)

        m = re.search(
            r'Tlim(?P<T_lim>[0-9]+(?:\.[0-9]+)?)_'
            r'Tpeak(?P<T_peak>[0-9]+(?:\.[0-9]+)?)_'
            r'tconv(?P<t_conv>[0-9]+(?:\.[0-9]+)?)',
            filename
        )
        vals = {k: float(v) for k, v in m.groupdict().items()}
        # Calculate integrated temperature overrun for x-axis ordering
        row = temp_idx_df.loc[tuple(vals[k] for k in temp_idx_df.index.names)]
        forcing = forcing_function(1.0, row["mu_0"], row["mu_1"], vals["T_lim"], row["R"])
        integral = quad(forcing, 0, time)[0]
        results_array[i] = data
        temperature_props[i] = np.array([*list(vals.values()),integral])

    strengths = np.arange(0.0, 1.01, 0.1)
    components = ["time", "GIS", "AMOC", "WAIS", "Amazonas", "nino", "assi", "total"]
    temperature_index = pd.MultiIndex.from_arrays(
        temperature_props.T,
        names=["Tlim", "Tpeak", "tconv", "integral"]
    )
    strength_index = pd.Index(strengths, name="strength")
    component_index = pd.Index(components, name="component")
    # TODO name value column "value"
    long_df = (
        pd.DataFrame(
            results_array.reshape(len(temperature_props), -1),
            index=temperature_index,
            columns=pd.MultiIndex.from_product(
                [strength_index, component_index]
            )
        )
        .stack(["strength", "component"])
        .drop(index="time", level="component")
    )
    return long_df

def loess(X_plot, X_np, y_np, radius=0.2):
    pass # Awaiting implementation if it ever gets useful

def weighted_avg(values, idxs, dists, sigma=0.05):
    w = np.exp(-(dists**2) / (2*sigma**2))
    return np.sum(w * values[idxs]) / np.sum(w)

def return_neighbors(X_grid, X_data, radius=0.2):
    X_norm, norm = normalize(X_data, norm="max", axis=0, return_norm=True)
    X_grid_norm = X_grid / norm
    tree = cKDTree(X_norm)
    # find neighbors within radius
    neighbors = tree.query_ball_point(X_grid_norm, r=radius)
    return neighbors

def plot_legend(ax, cax=None):
    # --- sRGB chromaticity coordinates (CIE 1931) ---
    R = np.array([0.64, 0.33])
    G = np.array([0.30, 0.60])
    B = np.array([0.15, 0.06])

    primaries = np.vstack([R, G, B])

    # --- Generate barycentric samples inside the triangle ---
    n = 300
    colors = []
    points = []

    for i in range(n):
        for j in range(n - i):
            wR = i / n
            wG = j / n
            wB = 1 - wR - wG

            rgb = np.clip([wR, wG, wB], 0, 1)
            xy = wR * R + wG * G + wB * B

            colors.append(rgb)
            points.append(xy)
    points = np.array(points)
    colors = np.array(colors)

    # --- Map triangle to an equilateral triangle ---

    # Target equilateral triangle
    R_t = np.array([-0.5, -1/3])
    G_t = np.array([0.5, -1/3])
    B_t = np.array([0.0, 2/3])

    target = np.vstack([R_t, G_t, B_t])

    # Compute affine transform: xy → equilateral
    A = np.vstack([primaries.T, np.ones(3)])
    B_aff = np.vstack([target.T, np.ones(3)])

    M = B_aff @ np.linalg.inv(A)

    def affine_transform(p):
        p_h = np.c_[p, np.ones(len(p))]
        return (M @ p_h.T).T[:, :2]

    points_eq = affine_transform(points)
    primaries_eq = affine_transform(primaries)
    # taken from colorbar
    if not cax:
        fig = (  # Figure of first Axes; logic copied from make_axes.
            [*ax.flat] if isinstance(ax, np.ndarray)
            else [*ax] if np.iterable(ax)
            else [ax])[0].get_figure(root=False)
        current_ax = fig.gca()

        cax, kwargs = plt_cbar.make_axes(ax, aspect=1, pad=0)
        # make_axes calls add_{axes,subplot} which changes gca; undo that.
        fig.sca(current_ax)
    else:
        cax.cla()
    cax.grid(visible=False, which='both', axis='both')

    # --- Plot ---
    cax.scatter(points_eq[:, 0],
                points_eq[:, 1],
                marker=',', edgecolors='none', facecolors=colors)

    # Draw triangle edges
    cycle = np.vstack([primaries_eq, primaries_eq[0]])
    cax.plot(cycle[:, 0]*1.4, cycle[:, 1]*1.4+0.01, 'k', linewidth=2)

     # Labels
    cax.text(primaries_eq[0,0]*1.47+0.05, primaries_eq[0,1]*1.15 - 0.35, "AR", ha="left", fontsize="small")
    cax.text(primaries_eq[1,0]*1.47-0.44, primaries_eq[1,1]*1.15, "AMOC", rotation=301, fontsize="small")
    cax.text(primaries_eq[2,0]*1.47-0.62, primaries_eq[2,1]*1.15 - 0.5, "WAIS", rotation=61, fontsize="small")

    cax.axis('equal')
    cax.axis('off')
    return cax

def interaction_difference(df, component):
    feature = df.xs(component, level="component")
    return (feature.xs(1.0, level="strength") - feature.xs(0.0, level="strength")).droplevel("integral")

def overshoot_plot(ax:Axes):
    vals = {"T_lim": 1.0, "t_conv": 300, "T_peak":4.0}
    row = temp_idx_df.loc[tuple(vals[k] for k in temp_idx_df.index.names)]
    forcing = forcing_function(1.0, row["mu_0"], row["mu_1"], vals["T_lim"], row["R"])
    time = np.linspace(0, 500, 100)
    inner_color = 'w' # "#30736A"
    arrowprops = dict(arrowstyle="-", color=inner_color, shrinkA=0, shrinkB=1, lw=0.5) #dict(arrowstyle="->") #, connectionstyle="angle,angleA=0,angleB=90,rad=10")
    ax.plot(time, forcing(time), c=inner_color)
    ax.vlines(vals["t_conv"], ymin=0, ymax = forcing(vals["t_conv"]), color=inner_color, ls=':')
    ax.hlines(vals["T_peak"], xmin = 0, xmax = time[np.argmax(forcing(time))], color=inner_color, ls=':')
    ax.axhline(vals["T_lim"], ls=':', c=inner_color)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 4.6)
    ax.annotate(r"$T_\mathrm{lim}$", (500, vals["T_lim"]), (440, vals["T_lim"] +  0.55), arrowprops=arrowprops, fontsize="x-small",  ha="right", color=inner_color)
    ax.annotate(r"$T_\mathrm{peak}$", (time[np.argmax(forcing(time))], vals["T_peak"]), (time[np.argmax(forcing(time))]+90, vals["T_peak"]-0.1),
                arrowprops=arrowprops, fontsize="x-small", color=inner_color)
    ax.annotate(r"$t_\mathrm{conv}$", (vals["t_conv"], 0), (vals["t_conv"]-80, 0.22), arrowprops=arrowprops, fontsize="x-small", ha="right", color=inner_color)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Overshoot", fontsize="x-small", c="w", pad=0)
    outer_color = inner_color # "#14312d"  #"#30736A"
    ax.set_xlabel(r"$t$", fontsize="x-small", labelpad=1, color=outer_color)
    ax.set_ylabel(r"$\Delta$GMT", fontsize="x-small", labelpad=-0.1, color=outer_color)
    ax.spines['bottom'].set_color(inner_color)
    ax.spines['top'].set_color(inner_color)
    ax.spines['right'].set_color(inner_color)
    ax.spines['left'].set_color(inner_color)

def load_longform_df(path):
    df = pd.read_csv(fr"{path}")
    index_value_demarcation = df.columns.get_loc('component') + 1
    long_idx = pd.MultiIndex.from_frame(df.drop(columns=df.columns[index_value_demarcation:]))
    long_df = pd.DataFrame({key: df[key].to_numpy() for key in df.columns[index_value_demarcation:]}, index=long_idx)
    long_df = long_df.loc[~long_df.index.duplicated(), :] # sometimes happens due to only 2 sigfigs each
    # forces all df into a format with years as columns
    try:
        long_df.columns = np.int32(np.float32(long_df.columns))
    except ValueError:
        if "year" in long_idx.names:
            long_df = long_df.unstack("year")
            long_df.columns = long_df.columns.get_level_values(1)
        else: # make it explicit
            pass
    return long_df

def cartesian_product(*arrays):
    """
    Thank you StackOverflow
    Args:
        *arrays:

    Returns:

    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def state_plot(state_df:pd.DataFrame):
    X_plot, neighbors = lay_plot_grid(state_df)
    cfgs = {}
    for component in state_df.index.get_level_values("component").unique():
        y_grid = {}
        for key in state_df.index.get_level_values("strength").unique():
            # y_component = state_df.xs((key, component), level=["strength", "component"])#.to_numpy().flatten()
            # y_grid[key] = np.array([y_component[idx].mean()
            #                         for idx in neighbors])
            if component == "total":
                y_grid[key] = (state_df.xs((key, component), level=["strength", "component"])
                                .groupby(OVERSHOOT_PROPERTIES).mean())
            else:
                y_grid[key] = ((state_df.xs((key, component), level=["strength", "component"])>0)
                                .groupby(OVERSHOOT_PROPERTIES).mean())
            print(f"{component},{key}:{np.mean(y_grid[key])}")
        impact_component = y_grid[1.0] - y_grid[0.0]
        impact_df = pd.DataFrame({"value": impact_component},
                               index=pd.MultiIndex.from_arrays(X_plot.T, names=OVERSHOOT_PROPERTIES))
        cfgs[component] = prepare_impact_plot(impact_df)

    cfgs["titles"] = [f"{chr(97+i)}) {cfgs["total"]["titles"][i]}" for i in range(3)]  # Look ma, I took a C course!
    fig, axes = imshow_grid(
        **cfgs["total"],
        nrows=2,
        ncols=3,
        figsize=set_plot_size("article"),
        xlabel="Convergence time / a",
        ylabel=r"Peak temperature / $^\circ C$",
        cbar_label=r"Impact on \# tipped",
    )
    elements = ["AMOC", "Amazonas", "GIS"]
    element_Tlim_index = 2
    state_plots = np.array([cfgs[element]["matrices"] for element in elements])
    vmax = state_plots.max()
    vmin = state_plots.min()
    diverging_map = "managua"  # shiftedColorMap(plt.cm.coolwarm, 0, 1 - vmax / (vmax + abs(vmin)), 1)
    limit = np.nanmax(np.abs(state_plots[np.isfinite(state_plots)]))
    divnorm = colors.TwoSlopeNorm(vcenter=0, vmin= - limit, vmax=limit)
    for i, (ax, component) in enumerate(zip(axes[3:], elements)):
        im = ax.imshow(cfgs[component]["matrices"][element_Tlim_index], origin="lower", aspect="auto",
                   cmap=diverging_map, norm=divnorm)
        cfgs["titles"].append(rf"{chr(100+i)}) {component}")

    for ax, title in zip(axes, cfgs["titles"]):
        ax.set_title(title, loc='left', fontsize='medium')

    colorbar = fig.colorbar(im, ax=axes[3:])
    colorbar.set_label("Impact on element", fontsize="small")
    colorbar.ax.set_ylim(vmin - 0.03, vmax)  # cant go above b/c colorspace ends there
    # here be problems with rtol???
    colorbar.ax.set_yticks(calculate_clean_ticks(vmin, vmax)[np.isfinite(calculate_clean_ticks(vmin, vmax))])
    # y_ticks = [np.round(vmin, 1), *np.arange(0, 0.6, 0.5), np.round(vmax, 1)]
    # colorbar.ax.set_yticks(y_ticks, y_ticks)  # get_yticks is useless as ever, so this needs to be hardcoded

    plot_overshoot_inset(axes[-1])
    plt.show()


def plot_overshoot_inset(ax: Axes):
    overshoot_ax = ax.inset_axes((0.49, 0.15, 0.5, 0.5))
    overshoot_ax.patch.set_alpha(0.0)
    inset_indicator = ax.indicate_inset((1.5, 1.5, 1, 1), inset_ax=overshoot_ax, edgecolor='k', alpha=1, lw=0.5,
                                              transform=ax.transData)
    for connector in inset_indicator.connectors:
        connector.set(color="w")
    inset_indicator.connectors[0].set(visible=True)
    inset_indicator.connectors[1].set(visible=True)
    inset_indicator.connectors[3].set(visible=False)
    inset_indicator.rectangle.set(edgecolor="w")
    overshoot_plot(overshoot_ax)


def lay_plot_grid(state_df: DataFrame) -> tuple[list[int] | Any, ndarray[tuple[int, int], dtype[Any]]]:
    X_temp = state_df.index.to_frame(index=False)[OVERSHOOT_PROPERTIES].drop_duplicates().to_numpy()
    X_plot = cartesian_product(np.arange(0, 2.1, 1),
                               np.arange(2, 6.1, 1),
                               np.arange(100, 1001, 100))
    neighbors = return_neighbors(X_plot, X_temp, radius=0.16)  # avoids taking points at other Tlims
    return X_plot, neighbors


def cascade_analysis(df, X_temp):
    tipping_properties = pd.read_csv(f"{FOLDER}/tipping_properties.csv", index_col=0)
    components = df.index.get_level_values("component").unique()
    component_series = {component: df.xs(component, level="component").reindex(X_temp, fill_value=np.inf)
                        for component in components}
    cascades = {}
    translator = {component: f"{component}_time" for component in components}
    for initial_component_name, initial_component_data in component_series.items():
        initial_component_data = initial_component_data.to_numpy().flatten()
        for secondary_component_name, secondary_component_data in component_series.items():
            if secondary_component_name == initial_component_name:
                continue
            lhc_lookup = secondary_component_data.index.get_level_values("lhc")
            secondary_component_data = secondary_component_data.to_numpy().flatten()
            second_tip_time = tipping_properties.loc[lhc_lookup][translator[secondary_component_name]].to_numpy()
            tip_after = initial_component_data < secondary_component_data
            tip_in_time = secondary_component_data - second_tip_time < initial_component_data
            cascades[(initial_component_name, secondary_component_name)] = tip_in_time * tip_after # logical and
    cascades = pd.DataFrame(cascades)
    ### Here Be ChatGPT ###
    paths_by_length = precompute_paths(cascades, components)

    # --------------------------------------------------
    # Longest chain for one mask
    # --------------------------------------------------

    def longest_chain_and_start(mask):

        for length in (3, 2, 1):
            for candidate in paths_by_length[length]:

                ok = True
                for b in candidate["bits"]:
                    if not (mask & (1 << b)):
                        ok = False
                        break

                if ok:
                    return length, candidate["start"]

        return 0, None

    # --------------------------------------------------
    # Precompute all 4096 possible masks
    # --------------------------------------------------

    lookup = {
        mask: longest_chain_and_start(mask)
        for mask in range(2 ** 12)
    }

    # --------------------------------------------------
    # Convert rows -> bitmasks
    # --------------------------------------------------

    powers = (1 << np.arange(12))
    masks = cascades.astype(np.uint8).to_numpy().dot(powers)

    # --------------------------------------------------
    # Apply lookup
    # --------------------------------------------------

    result = pd.Series(masks).map(lookup)
    max_chain_df = pd.DataFrame({"value": result.str[0].to_numpy()}, index=X_temp)
    max_chain_df["component"] = "max_chain"
    max_chain_df = max_chain_df.set_index("component", append=True)
    chain_start_df = pd.DataFrame({"value": result.str[1].to_numpy()}, index=X_temp)
    chain_start_df["component"] = "chain_start"
    chain_start_df = chain_start_df.set_index("component", append=True)
    return max_chain_df, chain_start_df


def precompute_paths(cascades, components):
    cols = cascades.columns  # the MultiIndex you showed

    # 12 directed edges in fixed order:
    # ('Amazonas','AMOC'), ('Amazonas','GIS'), ...
    nodes = components

    # map edge -> bit position
    edge_index = {edge: i for i, edge in enumerate(cols)}

    # --------------------------------------------------
    # Precompute all simple paths (cycles not allowed)
    # --------------------------------------------------

    paths_by_length = {1: [], 2: [], 3: []}

    for length in [1, 2, 3]:
        for path in itertools.permutations(nodes, length + 1):

            bits = []
            valid = True

            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])

                if edge not in edge_index:
                    valid = False
                    break

                bits.append(edge_index[edge])

            if valid:
                paths_by_length[length].append({
                    "start": path[0],
                    "bits": bits,
                    "path": path
                })

    return paths_by_length


def causal_analysis(timing_df, X_temp):
    """
    Investigate the causal effects within the timing_df (treated as a categorization task).
    Probably better to just intervene in the network simulation
    Args:
        timing_df: Reduced df of when tipping elements tip
        X_temp: MultiIndex, recording all temperature-configuration combinations to reconstruct full df

    Returns:

    """
    # Either DoubleMLPLR or DoubleMLIRM
    # RandomForestClassifiers for treatment (AMOC) and outcome (Amazon)
    # Bit of a question what I actually want to show with this: the result may just be the corrected effect from Kriegler
    # Get Events: tipping times of NINO, AMAZONAS and AMOC. Slight problem: NINO cannot tip, therefore doesn't show up
    # in the record. Well, so much for rigor. I cant adjust for NINO anyway because it's a descendant of AMOC
    # Also, AMOC-AMAZON connection is 0.5 - 4, so this will be very ambiguous
    # relevant_components = timing_df.query("component in ['AMOC', 'Amazonas', 'NINO']")
    treatment_df =  timing_df.xs((1.0, "AMOC"), level=["strength", "component"]).reindex(X_temp, fill_value=np.inf)
    outcome_df = timing_df.xs((1.0, "Amazonas"), level=["strength", "component"]).reindex(X_temp, fill_value=np.inf)
    treatment_before_outcome = (treatment_df < outcome_df).to_numpy().flatten()
    outcome = (outcome_df < np.inf).to_numpy().flatten()
    df = pd.DataFrame({"treatment": np.int16(treatment_before_outcome), "outcome": np.int16(outcome),
                       "T_lim": X_temp.get_level_values("T_lim"),
                       "T_peak": X_temp.get_level_values("T_peak"),
                       "t_conv": X_temp.get_level_values("t_conv")})
    data = DoubleMLData(df, y_col="outcome", d_cols="treatment", x_cols=["T_lim", "T_peak", "t_conv"])
    ml_g_rf = RandomForestClassifier(n_estimators=50, max_depth=7, max_features=3, min_samples_leaf=3)
    ml_m_rf = RandomForestClassifier(n_estimators=50, max_depth=5, max_features=4, min_samples_leaf=7)
    dml_plr_tree = (DoubleMLIRM(data, ml_g=ml_g_rf, ml_m=ml_m_rf))
    dml_plr_tree.fit()
    print(np.mean(outcome))
    print(np.mean(np.logical_and(outcome, treatment_before_outcome)))
    print(dml_plr_tree.summary)


def combine_cfg_matrices(cfgs, components, index):
    """Combine the nth matrix entry from selected configuration dictionaries.

    Args:
        cfgs (dict): Dictionary of configuration dictionaries
        components (list): List of component names to combine
        index (int): Index of the matrix to extract from each component

    Returns:
        dict: Combined configuration with matrices, titles, and other properties
    """
    combined = {
        "matrices": [cfgs[comp]["matrices"][index] for comp in components],
        "titles": [cfgs[comp]["titles"][index] for comp in components],
        "x_ticks": cfgs[components[0]]["x_ticks"],
        "x_ticklabels": cfgs[components[0]]["x_ticklabels"],
        "y_ticks": cfgs[components[0]]["y_ticks"],
        "y_ticklabels": cfgs[components[0]]["y_ticklabels"],
    }
    return combined

def extract_influences(state_series, mode = "ATE"):
    interventions = state_series.index.get_level_values("intervention").unique()
    components = state_series.index.get_level_values("component").unique()
    X_plot = state_series.index.to_frame(index=False)[OVERSHOOT_PROPERTIES].drop_duplicates().to_numpy()
    influences = {}
    if X_plot.shape[0] < 5:
        influence_matrix = np.empty((len(interventions), len(components), X_plot.shape[0]))
    else:
        influence_matrix = np.empty((len(interventions), len(components), 1))
    influence_matrix[:] = np.nan
    for intervention in interventions:
        influences[intervention] = {}
        for component in components:
            if component == intervention:
                             #"NINO"]:  # uninteresting, cuz component then isnt dynamic/not tipping element
                continue
            # WAIS and GIS tip too slowly to show much effect on each other after 1ka
            no_tip = state_series.xs((intervention, -1, component),
                                     level=["intervention", "state", "component"])
            tip = state_series.xs((intervention, 1, component),
                                  level=["intervention", "state", "component"])
            # free_run = state_series.xs((intervention, 0, component),
            #                            level=["intervention", "state", "component"])
            # if component == "total":  # remove the effect of the intervention on the total
            #     tip -= state_series.xs((intervention, 1, intervention),
            #                            level=["intervention", "state", "component"]) > 0
            # p_intervention = ((state_df[50000].xs((intervention, 0, intervention),
            #                                       level=["intervention", "state", "component"]) > 1)
            #                   .groupby(level=[OVERSHOOT_PROPERTIES]).mean())
            if mode == "PF":
                if component != "total":
                    ate_on_component = (tip > 0).groupby(
                    level=[OVERSHOOT_PROPERTIES]).mean() / (no_tip > 0).groupby(
                    level=[OVERSHOOT_PROPERTIES]).mean()
                else:
                    ate_on_component = (tip - no_tip).groupby(
                    level=[OVERSHOOT_PROPERTIES]).mean()
            elif mode == "ATE":
                if component != "total":
                    ate_on_component = ((tip > 0).astype(int) - (no_tip > 0).astype(int)).groupby(
                    level=[OVERSHOOT_PROPERTIES]).mean()
                else:
                    # for kind of scary reasons, in the case of no connections, this massively deviates from the
                    # recorded tipping in no systematic way...
                    # Note I am neglecting REEF and NINO here, but those are almost always 1 and 0 anyway
                    manual_total = (state_series.xs(intervention, level="intervention")
                            .drop("total", level="component").drop(intervention, level="component")) > 0
                    no_tip_total = ((manual_total.xs(-1, level="state"))
                                    .groupby(level=["lhc", *OVERSHOOT_PROPERTIES]).sum()
                                    .groupby(level=[OVERSHOOT_PROPERTIES]).mean())
                    tip_total = ((manual_total.xs(1, level="state"))
                                    .groupby(level=["lhc", *OVERSHOOT_PROPERTIES]).sum()
                                    .groupby(level=[OVERSHOOT_PROPERTIES]).mean())
                    ate_on_component = tip_total - no_tip_total
            else:
                y_component = tip - no_tip
                ate_on_component = y_component.groupby(
                    level=[OVERSHOOT_PROPERTIES]).mean()  # np.array([y_component[idx].mean() for idx in neighbors])

            print(
                f"ATE of {intervention} on {component}:{np.mean(ate_on_component):.2f}+-{np.std(ate_on_component):.2f}")
            impact_df = pd.DataFrame({"value": ate_on_component},
                                     index=pd.MultiIndex.from_arrays(X_plot.T, names=OVERSHOOT_PROPERTIES))
            if len(impact_df) < 5:
                influences[intervention][component] = impact_df
                influence_matrix[interventions.get_loc(intervention), components.get_loc(component), :] = (
                    impact_df.to_numpy().flatten())
            else:
                influences[intervention][component] = impact_df.mean()
                influence_matrix[interventions.get_loc(intervention), components.get_loc(component), :] = (
                    impact_df.mean().to_numpy())
    return influences, influence_matrix, X_plot

def intervention_analysis(state_df, timing_df):
    # X_plot, neighbors = lay_plot_grid(state_df)
    state_series = state_df[1000]
    influences, X_temp = extract_influences(state_series)
    for intervention in influences.keys():
        selected_components = ["AMOC", "Amazonas", "GIS", "WAIS"]
        cfgs = {}
        for component in influences[intervention].keys():
            cfgs[component] = prepare_impact_plot(influences[intervention][component])
        # Create combined configuration for selected components at index 2
        selected_components.remove(intervention)
        cfgs["combined"] = combine_cfg_matrices(cfgs, selected_components, 2)

        cfgs["titles"] = [f"{chr(97 + i)}) {cfgs["total"]["titles"][i]}" for i in
                          range(4)]  # Look ma, I took a C course!
        fig, axes = imshow_grid(
            **cfgs["total"],
            nrows=2,
            ncols=2,
            figsize=set_plot_size("article"),
            xlabel="Convergence time / a",
            ylabel=r"Peak temperature / $^\circ C$",
            cbar_label=rf"Impact of {intervention} on elements",
            cbar_axis=4
        )
        fig.suptitle(f"Impact of {intervention}", fontsize="medium")
        for ax, title in zip(axes, cfgs["titles"]):
            ax.set_title(title, loc='left', fontsize='medium')
        plot_overshoot_inset(axes[-1])

        # Plot combined configuration
        fig_combined, axes_combined = imshow_grid(
            **cfgs["combined"],
            nrows=1,
            ncols=3,
            figsize=set_plot_size("article", fraction=1, subplots=(1, 3)),
            xlabel="Convergence time / a",
            ylabel=r"Peak temperature / $^\circ C$",
            cbar_label=rf"Impact of {intervention} on element",
            cbar_axis=3
        )
        fig_combined.suptitle(fr"Impact of {intervention} at $T_\mathrm{{lim}}={cfgs['combined']['titles'][0].split('=')[1]}",
                              fontsize="medium")
    plt.show()

def plot_influence_matrix(influence_matrix: ndarray[tuple[int, int, int], dtype[float64]], components, interventions,
                          temperatures, plotname=None):
    if len(influence_matrix.shape) == 2:
        influence_matrix = np.expand_dims(influence_matrix, axis=-1)
        temperatures = [[temperatures]]
    for i in range(influence_matrix.shape[-1]):
        # gridspec = plt.GridSpec()
        fig, axes = plt.subplots(1,4, width_ratios=(influence_matrix.shape[-2] - 1, 0.7, 1, 0.7),
                                 figsize=set_plot_size("article"), layout="constrained")
        comp_ax = axes[0] #fig.add_subplot(gridspec[0])
        comp_cmap_ax = axes[1] #fig.add_subplot(gridspec[1])
        total_ax = axes[2] #fig.add_subplot(gridspec[2], sharey=comp_ax)
        total_cmap_ax = axes[3] #fig.add_subplot(gridspec[3])
        component_values = influence_matrix[:, :-1, i]
        component_finite = component_values[np.isfinite(component_values)]
        lim = np.max(np.abs(component_finite)) + 0.05 if component_finite.size else 0.05
        cmap = matplotlib.colormaps.get_cmap('seismic')  # viridis is the default colormap for imshow
        cmap.set_bad(color='k')
        component_heatmap = comp_ax.imshow(component_values, origin="upper", aspect="equal",
                                           cmap=cmap, interpolation="none",
                                           norm=colors.TwoSlopeNorm(vmin=-1.1, vmax=1.1, vcenter=0))
        hatch_nan_cells(comp_ax, component_values)
        comp_ax.set_xticks(range(len(components)-1), labels=components[:-1],
                      rotation=45, ha="right", rotation_mode="anchor")
        comp_ax.set_yticks(range(len(interventions)), labels=interventions)

        total_values = influence_matrix[:, [-1], i]
        total_finite = total_values[np.isfinite(total_values)]
        vmax_total = np.nanmax(total_finite) if total_finite.size else 0
        vmin_total = np.nanmin(total_finite) if total_finite.size else 0
        lim_total = np.nanmax(np.abs(total_finite)) + 0.05 if total_finite.size else 0.05
        total_heatmap = total_ax.imshow(total_values, origin="upper", aspect="equal",
                                        cmap="bwr", interpolation="none",
                                        norm = colors.TwoSlopeNorm(vcenter=0, vmin=-lim_total, vmax=lim_total))
        hatch_nan_cells(total_ax, total_values)
        total_ax.set_xticks([0], labels=["total"],rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(total_ax.get_yticklabels(), visible=False)
        fig.suptitle(f"ATE at T={temperatures[i][0]}°C", fontsize='medium')
        comp_cbar = fig.colorbar(component_heatmap, cax=comp_cmap_ax)
        comp_min = np.nanmin(influence_matrix[:, :-1, i])
        comp_max = np.nanmax(influence_matrix[:, :-1, i])
        comp_cbar.ax.set_ylim(comp_min - 0.01, comp_max)
        calculate_clean_ticks(comp_min, comp_max, 5)
        comp_cbar.ax.set_yticks(calculate_clean_ticks(comp_min, comp_max, 5))
        # comp_cbar.ax.set_yticks([comp_min, -0.1,  0, 0.1, comp_max])
        comp_cmap_ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        total_colorbar = fig.colorbar(total_heatmap, cax=total_cmap_ax)
        total_colorbar.ax.set_ylim(vmin_total - 0.03, vmax_total + 0.03)
        #fig.tight_layout()
        if plotname:
            fig.savefig(fr"C:\Users\lukas\Documents\PhD\numerical_data\analysis_results\Causal Effect\{temperatures[i][0]}{plotname}.png")
    plt.show()

def intervention_matrix(state_df):
    time = 50000
    state_series = (state_df[time]
                    .xs(1.0, level="strength")
                    .drop(["NINO", "REEF"], level="component")
                    .drop(["REEF"], level="intervention"))
    components = state_series.index.get_level_values("component").unique()
    interventions = state_series.index.get_level_values("intervention").unique()
    influences, influence_matrix, temperatures = extract_influences(state_series, mode="ATE")
    pairwise_series = (state_df[time]
                    .xs(0.0, level="strength")
                    .drop(["NINO", "REEF"], level="component")
                    .drop(["REEF"], level="intervention"))
    _, pairwise_influence_matrix, temperatures = extract_influences(pairwise_series, mode="ATE")
    plot_influence_matrix(influence_matrix, components,interventions, temperatures,
                          f"second_order_infmatr{int(time/1000)}ka")


def network_effects(state_df):
    state_series = state_df[50000].drop(labels="NINO", level="component")#.xs(strength, level="strength")
    tip_series = state_series > 0
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True,
                             figsize=set_plot_size("article", fraction=1, subplots=(3, 2)))
    # fig.suptitle(f"{"Network" if strength == 1 else "Pairwise"} effects of {intervention} ", fontsize="medium")
    interesting_combinations = [("GIS", "AMOC"), ("WAIS", "AMOC"), ("AMOC", "WAIS")]
    rows = []
    for ax_i, (intervention, component) in enumerate(interesting_combinations):
        for strength in [0.0, 1.0]:
            ax = axes[ax_i, int(strength)]
            component_series = tip_series.xs((intervention, component, strength),
                                             level=["intervention", "component", "strength"])
            nti_series = component_series.xs(-1, level="state")
            ti_series = component_series.xs(1, level="state")
            p00 = np.mean(~nti_series & ~ti_series)
            p01 = np.mean(~nti_series & ti_series)
            p10 = np.mean(nti_series & ~ti_series)
            p11 = np.mean(nti_series & ti_series)
            M = np.array([[p00, p01], [p10, p11]])
            ax.imshow(M, cmap="plasma", vmin=0, vmax=1) #, origin="upper", aspect="auto", cmap="magma", interpolation="none")
            ax.set_xticks(np.arange(2), labels=[f"B stable\nif A tips", f"B tips\nif A tips"], rotation=90, fontsize='small')
            ax.set_yticks(np.arange(2), labels=[f"B stable\nif A stable", f"B tips\nif A stable"], rotation=0, fontsize='small')
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        fr"{100*M[i, j]:.0f}\%",
                        ha="center",
                        va="center"
                    )
        rows.append(f"{intervention} on {component}")
    pad = 5  # in points
    cols = ["Pairwise Effect", "Network Effect"]
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad + pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='small', ha='right', va='center', rotation="vertical")
    for ax in axes[:, 0]:
        ax.set_anchor("E")
    for ax in axes[:, 1]:
        ax.set_anchor("W")
    plt.tight_layout(w_pad=1.1)
    fig.subplots_adjust(wspace=0.1)
    plt.show()
    sankey(tip_series)

def sankey(tip_series):
    component = "AMOC"
    intervention = "WAIS"
    strength = 1
    component_series = tip_series.xs((intervention, component, strength),
                                     level=["intervention", "component", "strength"])
    nti_series = component_series.xs(-1, level="state")
    ti_series = component_series.xs(1, level="state")
    p00 = np.mean(~nti_series & ~ti_series)
    p01 = np.mean(~nti_series & ti_series)
    p10 = np.mean(nti_series & ~ti_series)
    p11 = np.mean(nti_series & ti_series)
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=30,
                thickness=20,
                label=[
                    f"{component} stable when {intervention} stable (Y_0=0)",
                    f"{component} tips when {intervention} stable (Y_0=1)",
                    f"{component} stable when {intervention} tips (Y_1=0)",
                    f"{component} tips when {intervention} tips (Y_1=1)",
                ],
            ),
            link=dict(
                source=[0, 0, 1, 1],
                target=[2, 3, 2, 3],
                value=[p00, p01, p10, p11],
            ),
        )
        )

    fig.update_layout(
        title="Intervention transition structure",
        font_size=14,
    )
    # fig.write_image(fr"C:\Users\lukas\Documents\PhD\numerical_data\analysis_results\Causal Effect\intervention_transition_structure.jpg")
    fig.show()

def plot_pf_calibration():
    data = pd.read_csv("calibrations/interaction_calibration.csv", header=[0, 1], index_col=0)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=set_plot_size("article", fraction=1,
                                                                           subplots=(1, 2)))
    ax1.plot(data['GIS_to_AMOC']['pf'], data['GIS_to_AMOC']['interaction_fac'])
    ax1.set_xlabel("Probability factor")
    ax1.set_ylabel(r"Derivative coupling/$\tau_\mathrm{GIS}$")
    ax1.set_title("GIS to AMOC")

    ax2.plot(data['GIS_to_WAIS']['pf'], data['GIS_to_WAIS']['interaction_fac'])
    ax2.set_xlabel("Probability factor")
    ax2.set_ylabel(r"Linear coupling")
    ax2.set_title("GIS to WAIS")
    fig.suptitle("Calibration of interaction factors")
    plt.show()

def main():
    timeframes = {#"close": 100,
                "medium":1000,
                "far":50000}
    # long_df = read_files(keyword, timeframe)
    timing_df = load_longform_df(fr"{FOLDER}\timing_dataframe.csv")
    if "nino_state" in timing_df.columns:
        timing_df.drop(columns = "nino_state", inplace=True) # preliminary
    # plot_pf_calibration()
    state_df = load_longform_df(fr"{FOLDER}\dataframe.csv")
    # intervention_analysis(state_df, timing_df)
    intervention_matrix(state_df)
    network_effects(state_df)
    # snapshot_df = state_df[50_000]
    # snapshot_df.name = "value"
    # # snapshot_df = pd.concat([snapshot_df, max_chain_df]).sort_index()
    # state_plot(snapshot_df)

OVERSHOOT_PROPERTIES = ["T_lim", "T_peak", "t_conv"]
FOLDER = r"C:\Users\lukas\Documents\PhD\numerical_data\results\intervention\2026-08-25_1"
if __name__ == "__main__":
    main()


# No connection THC-to-AMAZ, indirect influence ATE: -0.0589 (0.27 tipping chance, 0.15 if thc tipped before. Oddly, additional
# negative correlation from THC tipping? Probably due to the messy cycles in that region of the model. Might also be with
# AMOC tipping slower than AMAZ, but exerting influence already)
# Full connection ATE: -0.1875 (0.22 tipping chance, 0.088 if thc tipped before)
# Read: if I set THC to tipped, AMAZ would not tip in an additional 18% of scenarios
# TODO: influence in flat trajectories, additional elements from bara
# Why do I have permafrost? Why do we use cusps without quadratic term?
