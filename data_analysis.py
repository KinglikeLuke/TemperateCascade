import glob
import os
import re
import itertools
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype, ndarray
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

import cProfile

from torchvision.datasets import folder

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
    "figure.dpi": 300
})
temp_df = pd.read_csv(r"temp_input\Tpeak_tconv_values\temp_input_values.txt", dtype=float, delimiter=" ", comment="#")
temp_idx_df = temp_df.set_index(["T_lim", "T_peak", "t_conv"])

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

def calculate_clean_ticks(vmin, vmax, target_ticks=6):
    """Calculate clean tick positions between vmin and vmax."""
    data_range = vmax - vmin

    # Candidate step sizes (clean numbers)
    step_candidates = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    # Scale candidates to appropriate magnitude
    magnitude = 10 ** np.floor(np.log10(data_range))
    scaled_candidates = [s * magnitude for s in step_candidates]
    scaled_candidates.extend([s * magnitude / 10 for s in step_candidates])
    scaled_candidates.extend([s * magnitude * 10 for s in step_candidates])

    # Find step size that gives closest to target number of ticks
    best_step = None
    best_diff = float('inf')

    for step in scaled_candidates:
        first_tick = np.ceil(vmin / step) * step
        last_tick = np.floor(vmax / step) * step
        n_ticks = int(np.round((last_tick - first_tick) / step)) + 1

        if 3 <= n_ticks <= 5:
            diff = abs(n_ticks - target_ticks)
            if diff < best_diff:
                best_diff = diff
                best_step = step

    # Fallback if no suitable step found
    if best_step is None:
        best_step = data_range / (target_ticks - 1)

    # Generate interior ticks
    first_tick = np.ceil(vmin / best_step) * best_step
    last_tick = np.floor(vmax / best_step) * best_step
    interior_ticks = np.arange(first_tick, last_tick + best_step / 2, best_step)

    # Always include exact min and max
    if abs(interior_ticks[0] - vmin) > best_step / 4:
        all_ticks = np.concatenate([[vmin], interior_ticks])
    if abs(interior_ticks[-1] - vmax) > best_step / 4:
        all_ticks = np.concatenate([interior_ticks, [vmax]])
    all_ticks = np.round(np.unique(all_ticks), 2)

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

    diverging_map = "managua"  # shiftedColorMap(plt.cm.coolwarm, 0, 1 - vmax / (vmax + abs(vmin)), 1)
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
    y_ticks = np.int16(np.linspace(0, len(Tpeak)-1, 4, endpoint=True))
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
            y_component = state_df.xs((key, component), level=["strength", "component"]).to_numpy().flatten()
            y_grid[key] = np.array([y_component[idx].mean()
                                    for idx in neighbors])
            print(f"{component},{key}:{np.mean(y_component)}")
        impact_component = y_grid[1.0] - y_grid[0.0]
        impact_df = pd.DataFrame({"value": impact_component},
                               index=pd.MultiIndex.from_arrays(X_plot.T, names=OVERSHOOT_PROPERTIES))
        cfgs[component] = prepare_impact_plot(impact_df)

    cfgs["titles"] = [f"{chr(97+i)}) {cfgs["totals"]["titles"][i]}" for i in range(3)]  # Look ma, I took a C course!
    fig, axes = imshow_grid(
        **cfgs["totals"],
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
    divnorm = colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    for i, (ax, component) in enumerate(zip(axes[3:], )):
        im = ax.imshow(cfgs[component]["matrices"][element_Tlim_index], origin="lower", aspect="auto",
                   cmap=diverging_map, norm=divnorm)
        cfgs["titles"].append(rf"{chr(100+i)}) {component}")

    for ax, title in zip(axes, cfgs["titles"]):
        ax.set_title(title, loc='left', fontsize='medium')

    colorbar = fig.colorbar(im, ax=axes[3:])
    colorbar.set_label("Impact on element", fontsize="small")
    colorbar.ax.set_ylim(vmin - 0.03, vmax)  # cant go above b/c colorspace ends there
    y_ticks = [np.round(vmin, 1), *np.arange(0, 0.6, 0.5), np.round(vmax, 1)]
    colorbar.ax.set_yticks(y_ticks, y_ticks)  # get_yticks is useless as ever, so this needs to be hardcoded

    plot_overshoot_inset(axes[-1])
    plt.show()


def plot_overshoot_inset(ax: Axes):
    overshoot_ax = ax.inset_axes((0.49, 0.15, 0.5, 0.5))
    overshoot_ax.patch.set_alpha(0.0)
    inset_indicator = ax.indicate_inset((2.5, 3.5, 1, 1), inset_ax=overshoot_ax, edgecolor='k', alpha=1, lw=0.5,
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
    neighbors = return_neighbors(X_plot, X_temp, radius=0.06)  # avoids taking points at other Tlims
    return X_plot, neighbors


def cascade_analysis(df, X_temp):
    tipping_properties = pd.read_csv(f"{FOLDER}/tipping_properties.csv", index_col=0)
    components = df.index.get_level_values("component").unique()
    component_series = {component: df.xs(component, level="component").reindex(X_temp, fill_value=np.inf)
                        for component in components}
    cascades = {}
    # names used in the LHS are so different from the ones used in data storage that this is the easiest way to get at them
    translator = {"GIS":'gis_time', "AMOC":'thc_time', "WAIS":'wais_time', "Amazonas":'amaz_time', "NINO":'nino_time'}
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


def intervention_analysis(state_df, timing_df):
    # X_plot, neighbors = lay_plot_grid(state_df)

    state_series = state_df[50000]
    X_plot = state_series.index.to_frame(index=False)[OVERSHOOT_PROPERTIES].drop_duplicates().to_numpy()
    for intervention in state_series.index.get_level_values("interventions").unique():
        cfgs = {}
        for component in state_series.index.get_level_values("component").unique():
            if component in [intervention,
                             "NINO"]:  # uninteresting, cuz component then isnt dynamic/not tipping element
                continue
            # WAIS and GIS tip too slowly to show much effect on each other after 1ka
            no_tip = state_series.xs((intervention, -1, component),
                                     level=["interventions", "intervention_states", "component"])
            tip = state_series.xs((intervention, 1, component),
                                  level=["interventions", "intervention_states", "component"])
            free_run = state_series.xs((intervention, 0, component),
                                       level=["interventions", "intervention_states", "component"])
            if component == "total":  # remove the effect of the intervention on the total
                tip -= state_series.xs((intervention, 1, intervention),
                                       level=["interventions", "intervention_states", "component"]) > 0
            p_intervention = ((state_df[50000].xs((intervention, 0, intervention),
                                                  level=["interventions", "intervention_states", "component"]) > 1)
                              .groupby(level=[OVERSHOOT_PROPERTIES]).mean())
            y_component = tip - no_tip
            ate_on_component = y_component.groupby(
                level=[OVERSHOOT_PROPERTIES]).mean()  # np.array([y_component[idx].mean() for idx in neighbors])

            print(
                f"ATE of {intervention} on {component}:{np.mean(ate_on_component):.2f}+-{np.std(ate_on_component):.2f}")
            impact_df = pd.DataFrame({"value": ate_on_component},
                                     index=pd.MultiIndex.from_arrays(X_plot.T, names=OVERSHOOT_PROPERTIES))
            cfgs[component] = prepare_impact_plot(impact_df)

        # Create combined configuration for selected components at index 2
        selected_components = ["AMOC", "Amazonas", "GIS", "WAIS"]
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


def plot_pf_calibration():
    data = pd.read_csv("interaction_calibration.csv", header=[0, 1], index_col=0)
    plt.plot(data['gis_to_thc']['pf'], data['gis_to_thc']['interaction_fac'])
    plt.xlabel("PF")
    plt.ylabel(r"linear coupling/$\tau_\mathrm{GIS}$")
    plt.title("Calibration curve of GIS to AMOC")
    plt.show()

def main():
    timeframes = {#"close": 100,
                "medium":1000,
                "far":50000}
    # long_df = read_files(keyword, timeframe)
    timing_df = load_longform_df(fr"{FOLDER}\timing_dataframe.csv")
    if "nino_state" in timing_df.columns:
        timing_df.drop(columns = "nino_state", inplace=True) # preliminary
    state_df = load_longform_df(fr"{FOLDER}\dataframe.csv")
    for keyword, timeframe in timeframes.items():
        intervention_analysis(state_df, timing_df)
        snapshot_df = state_df[timeframe]
        snapshot_df.name = "value"
        # snapshot_df = pd.concat([snapshot_df, max_chain_df]).sort_index()
        state_plot(snapshot_df)

OVERSHOOT_PROPERTIES = ["T_lim", "T_peak", "t_conv"]
FOLDER = r"C:\Users\lukas\Documents\PhD\numerical_data\results\intervention\2026-05-15_2"
if __name__ == "__main__":
    main()


# No connection THC-to-AMAZ, indirect influence ATE: -0.0589 (0.27 tipping chance, 0.15 if thc tipped before. Oddly, additional
# negative correlation from THC tipping? Probably due to the messy cycles in that region of the model. Might also be with
# AMOC tipping slower than AMAZ, but exerting influence already)
# Full connection ATE: -0.1875 (0.22 tipping chance, 0.088 if thc tipped before)
# Read: if I set THC to tipped, AMAZ would not tip in an additional 18% of scenarios
# TODO: influence in flat trajectories, additional elements from bara