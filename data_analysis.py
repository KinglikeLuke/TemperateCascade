import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import statsmodels.api as sm
from scipy.integrate import quad
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colorbar as cbar
import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "sans-serif",      # Use sans serif font family
    "font.serif": ["Computer Modern"],# Match default LaTeX font
    "font.sans-serif": ["Latin Modern"],
    "axes.labelsize": 10,             # Font size for axis labels
    "font.size": 10,                  # General font size
    "legend.fontsize": 8,             # Legend font size
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
        "ytick.direction": "in",
    "xtick.direction": "in",
    'figure.constrained_layout.use': True,
    "legend.frameon":    False,
    "figure.dpi": 300
})

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
    vmin=None,
    vmax=None,):
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    axes = [axes] if isinstance(axes, Axes) else list(axes.flat)

    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="inferno",
        #    interpolation="bilinear"
        )
        ax.set_title(title)

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

    cbar = fig.colorbar(im, ax=axes[:len(axes)//2], shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)

    return fig, axes

def prepare_impact_plot(impact_df, Tlim=None):
    # avg_impact = impact_df.xs(0.5, level="strength").droplevel("integral")
    Tlim_options = impact_df.index.get_level_values("Tlim").unique()
    if not Tlim:
        Tlim = Tlim_options
    tconv = (
        impact_df.index
        .get_level_values("tconv")
        .unique()
        .sort_values()
    )

    matrices = []
    titles = []

    for tlim in Tlim:
        df_slice = impact_df.xs(tlim, level="Tlim")
        mat = (
            df_slice
            .unstack("tconv")
            .sort_index()
            .values
        )
        matrices.append(mat)
        decimal = int(tlim-int(tlim)!=0)
        titles.append(fr"$T_{{\mathrm{{lim}}}}={tlim:.{decimal}f}^\circ C$")

    Tpeak = (
        df_slice.index
        .get_level_values("Tpeak")
        .unique()
        .sort_values()
    )

    return {
        "matrices": matrices,
        "titles": titles,
        "x_ticks": np.arange(len(tconv))[::3],
        "x_ticklabels": tconv.round(2)[::3],
        "y_ticks": np.arange(len(Tpeak)),
        "y_ticklabels": Tpeak.round(2),
        "vmin": impact_df.min(),
        "vmax": impact_df.max(),
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
    filenames = glob.glob(f"results/no_feedbacks/network_1.0_1.0_1.0/*{suffix}.npy")
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

def loess(X_np, y_np, span=0.2):
    X_norm = normalize(X_np, norm="max", axis=0)
    strength_coefs = np.zeros(y_np.shape)
    for i, x_star in enumerate(X_norm):
        distance = np.sqrt(np.sum((X_norm-x_star)**2, axis=1))
        mask = distance<span
        X_loc, y_loc = X_np[mask], y_np[mask]
        X_loc = np.column_stack([np.ones(len(X_loc)), X_loc])
        coef, *_ = np.linalg.lstsq(X_loc, y_loc, rcond=None)
        strength_coefs[i] = coef[-1]
    return strength_coefs

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

        cax, kwargs = cbar.make_axes(ax, aspect=1, pad=0)
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

def dataframe_ify(long_df:pd.DataFrame, new_data):
    new_df = long_df.copy().xs("total", level="component")
    new_df[:] = new_data

    new_df = (
        new_df
        .groupby(level=["Tlim", "Tpeak", "tconv"])
        .mean()
    )
    return new_df

traj_results = glob.glob("results/*total_tipped.npy")
temp_df = pd.read_csv(r"temp_input\Tpeak_tconv_values\temp_input_values.txt", dtype=float, delimiter=" ", comment="#")
#temp_df = pd.DataFrame(temperature_trajs, columns=["Tpeak", "Tlim", "tconv", "R", "mu0", "mu1"])
temp_idx_df = temp_df.set_index(["T_lim", "T_peak", "t_conv"])
timeframes = {#"close": 100, 
              "medium":1000, 
              "far":1000}

for keyword, timeframe in timeframes.items():

    long_df = read_files(keyword, timeframe)
    long_df = long_df.drop(2.0, level="Tpeak")
    feature_df = long_df.xs("total", level="component")

    X = feature_df.reset_index(name="value")[["Tlim", "Tpeak", "tconv", "strength"]].to_numpy()
    y_total = feature_df.values
    y_gis = long_df.xs("GIS", level="component").values
    y_wais = long_df.xs("WAIS", level="component").values
    y_amoc = long_df.xs("AMOC", level="component").values
    y_ar = long_df.xs("Amazonas", level="component").values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # # Forest gets absolutely overfitted to regular Temperature and strength features
    # regr = ensemble.RandomForestRegressor(n_estimators=5, max_depth=4)
    # regr.fit(X_train, y_train)
    # r2=1-np.sum((regr.predict(X_test)-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)
    # r2_adj = 1-(1-r2)*(y_train.size-1)/(y_train.size-1-sum(tree.tree_.node_count for tree in regr.estimators_)*5)


    GIS_diff = loess(X, y_gis)
    WAIS_diff = loess(X, y_wais)
    AMOC_diff = loess(X, y_amoc)
    AR_diff = loess(X, y_ar)
    # total_diff = loess(X, y_total, 0.2)
    total_diff_df = (feature_df.xs(1.0, level="strength") - feature_df.xs(0.0, level="strength")).droplevel("integral") 
    # total_diff_df /= feature_df.xs(0.0, level="strength").droplevel("integral")
    cfg_gis = prepare_impact_plot(dataframe_ify(long_df, GIS_diff))
    cfg_impact = prepare_impact_plot(impact_df=total_diff_df, Tlim=[0.,1.,2.])
    cfg_wais = prepare_impact_plot(dataframe_ify(long_df, WAIS_diff))
    cfg_amoc = prepare_impact_plot(dataframe_ify(long_df, AMOC_diff))
    cfg_ar = prepare_impact_plot(dataframe_ify(long_df, AR_diff))

    fig, axes = imshow_grid(
        **cfg_impact,
        nrows=2,
        ncols=3,
        figsize=set_plot_size("article"),
        xlabel="Convergence time / a",
        ylabel=r"Peak temperature / $^\circ C$",
        cbar_label="Total impact",
    )
    element_tlim_index = 2
    # fig, axes = plt.subplots(1, 3, figsize=set_plot_size("article", subplots=(1, 4)), sharey=True, sharex=True)
    # axes[0].imshow(cfg_impact["matrices"][0], origin="lower", aspect="auto", vmin=cfg_impact["vmin"], vmax=cfg_impact["vmax"],cmap="inferno")
    # axes[0].set_title(cfg_impact["titles"][0])

    # im = axes[1].imshow(cfg_impact["matrices"][element_tlim_index], origin="lower", aspect="auto", vmin=cfg_impact["vmin"], vmax=cfg_impact["vmax"], cmap="inferno")
    # axes[1].set_title(cfg_impact["titles"][element_tlim_index])
    # fig.supxlabel("Convergence time / a")
    # fig.supylabel("Peak temperature / ^\circ C")
    # axes[0].set_yticks(cfg_impact["y_ticks"])
    # axes[0].set_yticklabels(cfg_impact["y_ticklabels"])
    # for ax in axes:
    #    ax.set_xticks(cfg_impact["x_ticks"])
    #    ax.set_xticklabels(cfg_impact["x_ticklabels"])
    rgb_matrix = np.array([
        cfg_ar["matrices"][element_tlim_index],
        cfg_amoc["matrices"][element_tlim_index], 
        cfg_gis["matrices"][element_tlim_index], 
        ]).transpose((1,2,0))
    # rgb_matrix /= np.sum(rgb_matrix, axis=-1)[:,:, None]
    # axes[-1].imshow(rgb_matrix, origin="lower", aspect="auto")
    # axes[-1].set_title(rf"{cfg_impact["titles"][element_tlim_index]}, per Element")
    # cax=plot_legend(axes)
    axes[3].imshow(cfg_amoc["matrices"][element_tlim_index], origin="lower", aspect="auto", vmin=rgb_matrix.min(), vmax=rgb_matrix.max())
    axes[3].set_title(f"{cfg_impact["titles"][element_tlim_index]}, AMOC")
    axes[4].imshow(cfg_gis["matrices"][element_tlim_index], origin="lower", aspect="auto", vmin=rgb_matrix.min(), vmax=rgb_matrix.max())
    axes[4].set_title(f"{cfg_impact["titles"][element_tlim_index]}, GIS")
    im = axes[5].imshow(cfg_ar["matrices"][element_tlim_index], origin="lower", aspect="auto", vmin=rgb_matrix.min(), vmax=rgb_matrix.max())
    axes[5].set_title(f"{cfg_impact["titles"][element_tlim_index]}, AR")

    colorbar = fig.colorbar(im, ax=axes[3:])
    colorbar.set_label("Element impact")

    components = ["AMOC", "GIS", "Amazonas", "total"]
    cfg_tipping = prepare_tipping_plot(long_df, components)
    nrows = 2 if len(components) > 1 else 1
    ncols = int(np.ceil(len(components) / nrows))
    # imshow_grid(
    #     **cfg_tipping,
    #     nrows=nrows,
    #     ncols=ncols,
    #     figsize=(4 * nrows, 3 * ncols),
    #     xlabel="Integrated Temperature index",
    #     ylabel="Interaction strength",
    #     cbar_label="value",
    # )
    plt.show()


