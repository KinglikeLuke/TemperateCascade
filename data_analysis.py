from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from plotly import graph_objects as go

from temp_input.overshoot_trajectory import overshoot_trajectory


OVERSHOOT_PROPERTIES = ["T_lim", "T_peak", "t_conv"]
FOLDER = Path(
    r"C:\Users\lukas\Documents\PhD\Paper 1\numerical_data\results\intervention\2026-09-22_temperature_HL_1"
)
TEMP_INPUT_PATH = Path(r"temp_input\Tpeak_tconv_values\temp_input_values.txt")
INFLUENCE_OUTPUT_DIR = Path(
    r"C:\Users\lukas\Documents\PhD\Paper 1\numerical_data\analysis_results\Causal Effect"
)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern"],
    "font.sans-serif": ["Latin Modern"],
    "axes.labelsize": 14,
    "font.size": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "ytick.direction": "out",
    "xtick.direction": "out",
    "figure.constrained_layout.use": True,
    "legend.frameon": False,
    "figure.dpi": 200,
})

temp_df = pd.read_csv(TEMP_INPUT_PATH, dtype=float, delimiter=" ", comment="#")
temp_idx_df = temp_df.set_index(OVERSHOOT_PROPERTIES)


@dataclass(frozen=True)
class FigureSize:
    width_type: str = "article"
    fraction: float = 1
    subplots: tuple[int, int] = (1, 1)

    def inches(self) -> tuple[float, float]:
        widths_pt = {
            "article": 418.25,
            "thesis": 398.33864,
            "report": 345.0,
            "prb": (3 + 3 / 8) * 72.27,
        }
        try:
            width = widths_pt[self.width_type]
        except KeyError as exc:
            raise ValueError(f"Unknown width type: {self.width_type}") from exc

        fig_width = width * self.fraction / 72.27
        golden_ratio = (5 ** 0.5 - 1) / 2
        fig_height = fig_width * golden_ratio * np.sqrt(self.subplots[0] / self.subplots[1])
        return fig_width, fig_height


@dataclass(frozen=True)
class MatrixGrid:
    matrices: list[np.ndarray]
    titles: list[str]
    x_ticks: np.ndarray | None = None
    x_ticklabels: np.ndarray | None = None
    y_ticks: np.ndarray | None = None
    y_ticklabels: np.ndarray | None = None

    @classmethod
    def from_impact_frame(
        cls,
        impact_df: pd.DataFrame,
        t_limits: Iterable[float] | None = None,
    ) -> "MatrixGrid":
        tlim_options = impact_df.index.get_level_values("T_lim").unique().sort_values()
        if t_limits is None:
            t_limits = tlim_options

        tconv = impact_df.index.get_level_values("t_conv").unique().sort_values()
        matrices = []
        titles = []

        for tlim in t_limits:
            df_slice = impact_df.xs(tlim, level="T_lim")
            matrices.append(df_slice.unstack("t_conv").sort_index().values)
            decimal = int(tlim - int(tlim) != 0)
            titles.append(fr"$T_{{\mathrm{{lim}}}}={tlim:.{decimal}f}^\circ C$")

        tpeak = df_slice.index.get_level_values("T_peak").unique().sort_values()
        x_ticks = np.int16(np.linspace(0, len(tconv) - 1, 4, endpoint=True))
        y_ticks = np.int16(np.linspace(0, len(tpeak) - 1, 5, endpoint=True))
        return cls(
            matrices=matrices,
            titles=titles,
            x_ticks=x_ticks,
            x_ticklabels=np.int16(tconv.round(0))[x_ticks],
            y_ticks=y_ticks,
            y_ticklabels=tpeak.round(1)[y_ticks],
        )

    def with_titles(self, titles: list[str]) -> "MatrixGrid":
        return MatrixGrid(
            matrices=self.matrices,
            titles=titles,
            x_ticks=self.x_ticks,
            x_ticklabels=self.x_ticklabels,
            y_ticks=self.y_ticks,
            y_ticklabels=self.y_ticklabels,
        )


@dataclass(frozen=True)
class GridPlotConfig:
    nrows: int
    ncols: int
    size: FigureSize
    xlabel: str | None = None
    ylabel: str | None = None
    cbar_label: str | None = None
    cbar_axis: int | None = None
    cmap: str = "inferno"


@dataclass(frozen=True)
class InfluenceResult:
    influences: dict[str, dict[str, pd.DataFrame]]
    matrix: np.ndarray
    temperatures: np.ndarray


@dataclass(frozen=True)
class InfluenceMatrixPlotConfig:
    mode: str = "ATE"
    plotname: str | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class NetworkTransition:
    intervention: str
    component: str


@dataclass(frozen=True)
class NetworkEffectsConfig:
    time: int = 50_000
    combinations: tuple[NetworkTransition, ...] = (
        NetworkTransition("GIS", "AMOC"),
        NetworkTransition("WAIS", "AMOC"),
        NetworkTransition("AMOC", "WAIS"),
    )
    sankey_transition: NetworkTransition = NetworkTransition("WAIS", "AMOC")
    sankey_strength: float = 1.0


def set_plot_size(width_type: str, fraction: float = 1, subplots: tuple[int, int] = (1, 1)) -> tuple[float, float]:
    return FigureSize(width_type, fraction, subplots).inches()


def calculate_clean_ticks(vmin: float, vmax: float, target_ticks: int = 5) -> np.ndarray:
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
    ticks = np.round(np.array([vmin, *np.arange(start, vmax, preferred_interval), vmax]), 2)
    ticks = np.unique(ticks[np.isfinite(ticks)])
    if len(ticks) > 2 and ticks[1] - vmin < preferred_interval / 4:
        ticks = np.delete(ticks, 1)
    if len(ticks) > 2 and vmax - ticks[-2] < preferred_interval / 4:
        ticks = np.delete(ticks, -2)
    return ticks


def load_longform_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    index_value_demarcation = df.columns.get_loc("component") + 1
    long_idx = pd.MultiIndex.from_frame(df.drop(columns=df.columns[index_value_demarcation:]))
    long_df = pd.DataFrame(
        {key: df[key].to_numpy() for key in df.columns[index_value_demarcation:]},
        index=long_idx,
    )
    long_df = long_df.loc[~long_df.index.duplicated(), :]

    try:
        long_df.columns = np.int32(np.float32(long_df.columns))
    except ValueError:
        if "year" in long_idx.names:
            long_df = long_df.unstack("year")
            long_df.columns = long_df.columns.get_level_values(1)
    return long_df


def hatch_nan_cells(ax: Axes, values: np.ndarray, hatch: str = "////", edgecolor: str = "white") -> None:
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


def prepare_impact_plot(impact_df: pd.DataFrame, t_limits: Iterable[float] | None = None) -> MatrixGrid:
    return MatrixGrid.from_impact_frame(impact_df, t_limits)


def temperature_points(df: pd.DataFrame | pd.Series) -> np.ndarray:
    return df.index.to_frame(index=False)[OVERSHOOT_PROPERTIES].drop_duplicates().to_numpy()


def overshoot_plot(ax: Axes) -> None:
    vals = {"T_lim": 1.0, "t_conv": 300, "T_peak": 4.0}
    row = temp_idx_df.loc[tuple(vals[key] for key in temp_idx_df.index.names)]
    forcing = lambda t: overshoot_trajectory(t, 1.0, row["T_lim"], row["R"], row["mu_0"], row["mu_1"])
    time = np.linspace(0, 500, 100)
    inner_color = "w"
    arrowprops = dict(arrowstyle="-", color=inner_color, shrinkA=0, shrinkB=1, lw=0.5)

    ax.plot(time, forcing(time), c=inner_color)
    ax.vlines(vals["t_conv"], ymin=0, ymax=forcing(vals["t_conv"]), color=inner_color, ls=":")
    ax.hlines(vals["T_peak"], xmin=0, xmax=time[np.argmax(forcing(time))], color=inner_color, ls=":")
    ax.axhline(vals["T_lim"], ls=":", c=inner_color)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 4.6)
    ax.annotate(
        r"$T_\mathrm{lim}$",
        (500, vals["T_lim"]),
        (440, vals["T_lim"] + 0.55),
        arrowprops=arrowprops,
        fontsize="x-small",
        ha="right",
        color=inner_color,
    )
    ax.annotate(
        r"$T_\mathrm{peak}$",
        (time[np.argmax(forcing(time))], vals["T_peak"]),
        (time[np.argmax(forcing(time))] + 90, vals["T_peak"] - 0.1),
        arrowprops=arrowprops,
        fontsize="x-small",
        color=inner_color,
    )
    ax.annotate(
        r"$t_\mathrm{conv}$",
        (vals["t_conv"], 0),
        (vals["t_conv"] - 80, 0.22),
        arrowprops=arrowprops,
        fontsize="x-small",
        ha="right",
        color=inner_color,
    )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Overshoot", fontsize="x-small", c="w", pad=0)
    ax.set_xlabel(r"$t$", fontsize="x-small", labelpad=1, color=inner_color)
    ax.set_ylabel(r"$\Delta$GMT", fontsize="x-small", labelpad=-0.1, color=inner_color)
    for spine in ax.spines.values():
        spine.set_color(inner_color)


def extract_influences(
    state_series: pd.Series,
    interventions: pd.Index,
    components: pd.Index,
    mode: str = "ATE",
) -> InfluenceResult:
    mode = mode.upper()
    if mode not in {"ATE", "PF"}:
        raise ValueError(f"Unknown mode {mode}")

    x_plot = temperature_points(state_series)
    influence_depth = x_plot.shape[0] if x_plot.shape[0] < 5 else 1
    if influence_depth == 1 and x_plot.shape[0] >= 5:
        x_plot = x_plot[[0]]

    influence_matrix = np.full((len(interventions), len(components), influence_depth), np.nan)
    influences: dict[str, dict[str, pd.DataFrame]] = {}

    for intervention in interventions:
        influences[intervention] = {}
        for component in components:
            if component == intervention:
                continue

            if component == "total":
                manual_total = (
                    state_series.xs(intervention, level="intervention")
                    .drop("total", level="component", errors="ignore")
                    .drop(intervention, level="component", errors="ignore")
                    > 0
                )
                no_tip = manual_total.xs(-1, level="state").groupby(level=["lhc", *OVERSHOOT_PROPERTIES]).sum()
                tip = manual_total.xs(0, level="state").groupby(level=["lhc", *OVERSHOOT_PROPERTIES]).sum()
            else:
                no_tip = (
                    state_series.xs((intervention, -1, component), level=["intervention", "state", "component"])
                    > 0
                ).astype(int)
                tip = (
                    state_series.xs((intervention, 0, component), level=["intervention", "state", "component"])
                    > 0
                ).astype(int)

            tip_mean = tip.groupby(level=OVERSHOOT_PROPERTIES).mean()
            no_tip_mean = no_tip.groupby(level=OVERSHOOT_PROPERTIES).mean()
            if influence_depth == 1:
                tip_mean = pd.Series([tip_mean.mean()], index=pd.MultiIndex.from_arrays(x_plot.T, names=OVERSHOOT_PROPERTIES))
                no_tip_mean = pd.Series(
                    [no_tip_mean.mean()],
                    index=pd.MultiIndex.from_arrays(x_plot.T, names=OVERSHOOT_PROPERTIES),
                )

            if mode == "PF":
                effect = pd.Series(
                    np.divide(
                        tip_mean.to_numpy(),
                        no_tip_mean.to_numpy(),
                        out=np.full(tip_mean.shape, np.nan, dtype=float),
                        where=no_tip_mean.to_numpy() != 0,
                    ),
                    index=tip_mean.index,
                )
                effect[(tip_mean == 0) & (no_tip_mean == 0)] = 1
            else:
                effect = tip_mean - no_tip_mean

            print(f"{mode} of {intervention} on {component}: {effect.mean():.2f}+-{effect.std():.2f}")
            impact_df = pd.DataFrame({"value": effect}, index=effect.index)
            influences[intervention][component] = impact_df
            influence_matrix[interventions.get_loc(intervention), components.get_loc(component), :] = (
                impact_df.to_numpy().flatten()
            )

    return InfluenceResult(influences, influence_matrix, x_plot)


class AnalysisPlotter:
    def imshow_grid(self, grid: MatrixGrid, config: GridPlotConfig) -> tuple[Figure, list[Axes]]:
        fig, axes = plt.subplots(
            nrows=config.nrows,
            ncols=config.ncols,
            figsize=config.size.inches(),
            sharex=True,
            sharey=True,
        )
        axes_list = [axes] if isinstance(axes, Axes) else list(axes.flat)
        matrices = grid.matrices[: config.nrows * config.ncols]
        finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        limit = max(abs(vmin), abs(vmax), 0.05)
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)

        image = None
        for ax, matrix, title in zip(axes_list, matrices, grid.titles):
            image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=config.cmap, norm=norm)
            ax.set_title(title, loc="left", fontsize="medium")

        if config.xlabel:
            fig.supxlabel(config.xlabel)
        if config.ylabel:
            fig.supylabel(config.ylabel)

        if grid.x_ticks is not None:
            for ax in axes_list:
                ax.set_xticks(grid.x_ticks)
                ax.set_xticklabels(grid.x_ticklabels)

        if grid.y_ticks is not None:
            axes_list[0].set_yticks(grid.y_ticks)
            axes_list[0].set_yticklabels(grid.y_ticklabels)

        if image is not None:
            if config.cbar_axis:
                cbar_axes = axes_list[:config.cbar_axis]
            elif len(axes_list) > 1:
                cbar_axes = axes_list[:len(axes_list) // 2]
            else:
                cbar_axes = axes_list
            cbar = fig.colorbar(image, ax=cbar_axes)
            ticks = calculate_clean_ticks(vmin, vmax)
            cbar.ax.set_yticks(ticks)
            cbar.ax.set_ylim(vmin, vmax)
            cbar.ax.set_yticklabels([f"{tick:.2g}" for tick in ticks])
            if config.cbar_label:
                cbar.set_label(config.cbar_label, fontsize="small")

        return fig, axes_list

    def plot_overshoot_inset(self, ax: Axes) -> None:
        overshoot_ax = ax.inset_axes((0.49, 0.15, 0.5, 0.5))
        overshoot_ax.patch.set_alpha(0.0)
        inset_indicator = ax.indicate_inset(
            (1.5, 1.5, 1, 1),
            inset_ax=overshoot_ax,
            edgecolor="k",
            alpha=1,
            lw=0.5,
            transform=ax.transData,
        )
        for connector in inset_indicator.connectors:
            connector.set(color="w")
        inset_indicator.connectors[0].set(visible=True)
        inset_indicator.connectors[1].set(visible=True)
        inset_indicator.connectors[3].set(visible=False)
        inset_indicator.rectangle.set(edgecolor="w")
        overshoot_plot(overshoot_ax)

    def state_plot(self, state_df: pd.DataFrame | pd.Series, show: bool = True) -> tuple[Figure, list[Axes]]:
        x_plot = temperature_points(state_df)
        cfgs: dict[str, MatrixGrid] = {}
        strengths = state_df.index.get_level_values("strength").unique()

        for component in state_df.index.get_level_values("component").unique():
            y_grid = {}
            for strength in strengths:
                component_series = state_df.xs((strength, component), level=["strength", "component"])
                if component == "total":
                    y_grid[strength] = component_series.groupby(OVERSHOOT_PROPERTIES).mean()
                else:
                    y_grid[strength] = (component_series > 0).groupby(OVERSHOOT_PROPERTIES).mean()
                print(f"{component},{strength}: {np.mean(y_grid[strength]):.2f}")

            impact_component = y_grid[1.0] - y_grid[0.0]
            impact_df = pd.DataFrame(
                {"value": impact_component},
                index=pd.MultiIndex.from_arrays(x_plot.T, names=OVERSHOOT_PROPERTIES),
            )
            cfgs[component] = prepare_impact_plot(impact_df)

        total_titles = [f"{chr(97 + i)}) {title}" for i, title in enumerate(cfgs["total"].titles)]
        total_grid = cfgs["total"].with_titles(total_titles)
        fig, axes = self.imshow_grid(
            total_grid,
            GridPlotConfig(
                nrows=2,
                ncols=3,
                size=FigureSize("article"),
                xlabel="Convergence time / a",
                ylabel=r"Peak temperature / $^\circ C$",
                cbar_label=r"Impact on \# tipped",
            ),
        )

        elements = ["AMOC", "Amazonas", "GIS"]
        tlim_index = 2
        state_plots = np.array([cfgs[element].matrices for element in elements])
        finite = state_plots[np.isfinite(state_plots)]
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        limit = max(float(np.nanmax(np.abs(finite))), 0.05)
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)

        image = None
        for i, (ax, component) in enumerate(zip(axes[3:], elements)):
            image = ax.imshow(
                cfgs[component].matrices[tlim_index],
                origin="lower",
                aspect="auto",
                cmap="managua",
                norm=norm,
            )
            ax.set_title(f"{chr(100 + i)}) {component}", loc="left", fontsize="medium")

        if image is not None:
            colorbar = fig.colorbar(image, ax=axes[3:])
            colorbar.set_label("Impact on element", fontsize="small")
            colorbar.ax.set_ylim(vmin - 0.03, vmax)
            ticks = calculate_clean_ticks(vmin, vmax)
            colorbar.ax.set_yticks(ticks[np.isfinite(ticks)])

        self.plot_overshoot_inset(axes[-1])
        if show:
            plt.show()
        return fig, axes

    def plot_influence_matrix(
        self,
        influence_matrix: np.ndarray,
        components: pd.Index,
        interventions: pd.Index,
        config: InfluenceMatrixPlotConfig,
    ) -> list[Figure]:
        figures = []
        plot_matrix = np.array(influence_matrix, dtype=float, copy=True)
        if config.mode.upper() == "PF":
            plot_matrix = np.log2(plot_matrix)

        for i in range(plot_matrix.shape[-1]):
            fig, (comp_ax, comp_cmap_ax, total_ax, total_cmap_ax) = plt.subplots(
                1,
                4,
                width_ratios=(plot_matrix.shape[-2] - 1, 0.7, 1, 0.7),
                figsize=FigureSize("article").inches(),
                layout="constrained",
            )
            figures.append(fig)

            component_values = plot_matrix[:, :-1, i]
            component_heatmap = self._heatmap_with_hatching(comp_ax, component_values, components[:-1], interventions)

            total_values = plot_matrix[:, [-1], i]
            total_heatmap = self._heatmap_with_hatching(total_ax, total_values, ["total"], interventions, cmap="bwr")
            plt.setp(total_ax.get_yticklabels(), visible=False)

            self._configure_influence_colorbar(fig, component_heatmap, comp_cmap_ax, component_values, config.mode, 4)
            self._configure_influence_colorbar(fig, total_heatmap, total_cmap_ax, total_values, config.mode, 3)

            if config.plotname:
                fig.suptitle(f"{config.mode} on {config.plotname}", fontsize="medium")
                if config.output_dir:
                    config.output_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(config.output_dir / f"{config.mode},{config.plotname}.png")

        return figures

    def network_effects(
        self,
        state_df: pd.DataFrame,
        config: NetworkEffectsConfig = NetworkEffectsConfig(),
        show: bool = True,
    ) -> tuple[Figure, go.Figure]:
        state_series = state_df[config.time].drop(labels="NINO", level="component", errors="ignore")
        tip_series = state_series > 0
        fig, axes = plt.subplots(
            nrows=len(config.combinations),
            ncols=2,
            sharex=True,
            sharey=True,
            figsize=FigureSize("article", subplots=(len(config.combinations), 2)).inches(),
        )
        axes = np.atleast_2d(axes)
        rows = []

        for row_index, transition in enumerate(config.combinations):
            for strength in [0.0, 1.0]:
                ax = axes[row_index, int(strength)]
                matrix = transition_matrix(tip_series, transition, strength)
                ax.imshow(matrix, cmap="plasma", vmin=0, vmax=1)
                ax.set_xticks(
                    np.arange(2),
                    labels=["B stable\nif A tips", "B tips\nif A tips"],
                    rotation=90,
                    fontsize="small",
                )
                ax.set_yticks(
                    np.arange(2),
                    labels=["B stable\nif A stable", "B tips\nif A stable"],
                    rotation=0,
                    fontsize="small",
                )
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, fr"{100 * matrix[i, j]:.0f}\%", ha="center", va="center")
            rows.append(f"{transition.intervention} on {transition.component}")

        self._label_network_axes(axes, rows)
        plt.tight_layout(w_pad=1.1)
        fig.subplots_adjust(wspace=0.1)
        sankey_fig = self.sankey(tip_series, config.sankey_transition, config.sankey_strength)
        if show:
            plt.show()
            sankey_fig.show()
        return fig, sankey_fig

    def sankey(self, tip_series: pd.Series, transition: NetworkTransition, strength: float) -> go.Figure:
        matrix = transition_matrix(tip_series, transition, strength)
        fig = go.Figure(
            go.Sankey(
                node=dict(
                    pad=30,
                    thickness=20,
                    label=[
                        f"{transition.component} stable when {transition.intervention} stable (Y_0=0)",
                        f"{transition.component} tips when {transition.intervention} stable (Y_0=1)",
                        f"{transition.component} stable when {transition.intervention} tips (Y_1=0)",
                        f"{transition.component} tips when {transition.intervention} tips (Y_1=1)",
                    ],
                ),
                link=dict(
                    source=[0, 0, 1, 1],
                    target=[2, 3, 2, 3],
                    value=[matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]],
                ),
            )
        )
        fig.update_layout(title="Intervention transition structure", font_size=14)
        return fig

    def _heatmap_with_hatching(
        self,
        ax: Axes,
        values: np.ndarray,
        x_labels: Iterable[str],
        y_labels: Iterable[str],
        cmap: str = "seismic",
    ):
        x_labels = list(x_labels)
        y_labels = list(y_labels)
        finite = values[np.isfinite(values)]
        limit = np.nanmax(np.abs(finite)) + 0.05 if finite.size else 0.05
        colormap = matplotlib.colormaps.get_cmap(cmap).copy()
        colormap.set_bad(color="k")
        heatmap = ax.imshow(
            values,
            origin="upper",
            aspect="equal",
            cmap=colormap,
            interpolation="none",
            norm=colors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
        )
        hatch_nan_cells(ax, values)
        ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(y_labels)), labels=y_labels)
        return heatmap

    def _configure_influence_colorbar(
        self,
        fig: Figure,
        heatmap,
        cax: Axes,
        values: np.ndarray,
        mode: str,
        target_ticks: int,
    ) -> None:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return

        value_min = float(np.nanmin(finite))
        value_max = float(np.nanmax(finite))
        cbar = fig.colorbar(heatmap, cax=cax)
        cbar.ax.set_ylim(value_min - 0.01, value_max + 0.01)
        ticks = calculate_clean_ticks(value_min, value_max, target_ticks)
        if mode.upper() == "PF":
            cbar.ax.set_yticks(ticks, labels=np.round(2 ** ticks, 2))
        else:
            cbar.ax.set_yticks(ticks, labels=np.round(ticks, 2))

    def _label_network_axes(self, axes: np.ndarray, rows: list[str]) -> None:
        pad = 5
        cols = ["Pairwise Effect", "Network Effect"]
        for ax, col in zip(axes[0], cols):
            ax.annotate(
                col,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size="small",
                ha="center",
                va="baseline",
            )

        for ax, row in zip(axes[:, 0], rows):
            ax.annotate(
                row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad + pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size="small",
                ha="right",
                va="center",
                rotation="vertical",
            )
        for ax in axes[:, 0]:
            ax.set_anchor("E")
        for ax in axes[:, 1]:
            ax.set_anchor("W")


def transition_matrix(tip_series: pd.Series, transition: NetworkTransition, strength: float) -> np.ndarray:
    component_series = tip_series.xs(
        (transition.intervention, transition.component, strength),
        level=["intervention", "component", "strength"],
    )
    no_treatment = component_series.xs(-1, level="state")
    treatment = component_series.xs(1, level="state")
    return np.array([
        [np.mean(~no_treatment & ~treatment), np.mean(~no_treatment & treatment)],
        [np.mean(no_treatment & ~treatment), np.mean(no_treatment & treatment)],
    ])


def plot_overshoot_inset(ax: Axes) -> None:
    AnalysisPlotter().plot_overshoot_inset(ax)


def plot_influence_matrix(
    influence_matrix: np.ndarray,
    components: pd.Index,
    interventions: pd.Index,
    mode: str = "ATE",
    plotname: str | None = None,
) -> list[Figure]:
    config = InfluenceMatrixPlotConfig(mode=mode, plotname=plotname, output_dir=INFLUENCE_OUTPUT_DIR if plotname else None)
    return AnalysisPlotter().plot_influence_matrix(influence_matrix, components, interventions, config)


def state_plot(state_df: pd.DataFrame | pd.Series, show: bool = True) -> tuple[Figure, list[Axes]]:
    return AnalysisPlotter().state_plot(state_df, show=show)


def intervention_matrix(
    state_df: pd.DataFrame,
    time: int = 50_000,
    mode: str = "PF",
    show: bool = True,
) -> list[Figure]:
    state_df = (
        state_df
        .drop(labels="REEF", level="component", errors="ignore")
        .drop(labels="REEF", level="intervention", errors="ignore")
    )
    state_series = state_df[time].xs(1.0, level="strength")
    components = state_series.index.get_level_values("component").unique()
    if "total" in components:
        components = components.drop("total").append(pd.Index(["total"]))
    interventions = state_series.index.get_level_values("intervention").unique()
    order = {component: i for i, component in enumerate(components)}
    interventions = pd.Index(sorted(interventions, key=lambda intervention: order.get(intervention, len(order))))

    result = extract_influences(state_series, interventions, components, mode=mode)

    figures = AnalysisPlotter().plot_influence_matrix(
        result.matrix,
        components,
        interventions,
        InfluenceMatrixPlotConfig(mode=mode),
    )
    if show:
        plt.show()
    return figures


def network_effects(
    state_df: pd.DataFrame,
    config: NetworkEffectsConfig = NetworkEffectsConfig(),
    show: bool = True,
) -> tuple[Figure, go.Figure]:
    return AnalysisPlotter().network_effects(state_df, config=config, show=show)


def sankey(
    tip_series: pd.Series,
    transition: NetworkTransition = NetworkTransition("WAIS", "AMOC"),
    strength: float = 1.0,
    show: bool = True,
) -> go.Figure:
    fig = AnalysisPlotter().sankey(tip_series, transition, strength)
    if show:
        fig.show()
    return fig


def main() -> None:
    state_df = load_longform_df(FOLDER / "dataframe.csv")
    intervention_matrix(state_df)
    network_effects(state_df)


if __name__ == "__main__":
    main()
