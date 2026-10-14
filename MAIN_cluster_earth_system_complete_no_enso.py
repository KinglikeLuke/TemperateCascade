# Add modules directory to path
import os
import sys
import re
import json
import glob
from pathlib import Path
import cProfile
from dataclasses import dataclass
sys.path.append('')

# global imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set(font_scale=1.)
import itertools
import datetime
# from PyPDF2 import PdfMerger # dont really know what I would want with the merged PDFs
from scipy.integrate import solve_ivp, quad
from tqdm import tqdm
import pandas as pd
from pydoe import lhs
# private imports from sys.path

#private imports for earth system
from earth_sys.timing_no_enso import individual_timescales
from earth_sys.earth_no_enso import earth_network, earth_elements, intervene_in_network

from temp_input.overshoot_trajectory import (
    overshoot_trajectory,
    fit_parameters,
    load_projection_forcings,
)
# TODO ask llm to makes this less ass
from start_ensemble.latin_probability_distribution import update_limits
update_limits()
#measure time
#start = time.time()
#############################GLOBAL SWITCHES#########################################
time_scale = True            # time scale of tipping is incorporated
plus_minus_include = True    # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
######################################################################
DURATION = 50000 #actual real simulation years
N_STEPS = 1000
T_0 = 1.0
N_OVERSHOOTS = 200
TEMPERATURE_TRAJECTORY_KIND = "forcing" # os.environ.get("TEMPERATURE_TRAJECTORY_KIND", "overshoot").lower()
FORCING_PROJECTION_PATH = r"temp_input\trajectories\temperature_HL.csv"#os.environ.get("FORCING_PROJECTION_PATH", r"temp_input\trajectories")
FORCING_TIME_COLUMN = "timebounds" #os.environ.get("FORCING_TIME_COLUMN") or None
FORCING_TEMPERATURE_COLUMN = "*" # os.environ.get("FORCING_TEMPERATURE_COLUMN") or None
FORCING_TIME_ZERO = 2020 # os.environ.get("FORCING_TIME_ZERO")
FORCING_TIME_ZERO = float(FORCING_TIME_ZERO) if FORCING_TIME_ZERO is not None else None
FORCING_EXTRAPOLATION = os.environ.get("FORCING_EXTRAPOLATION", "edge").lower()
#Names to create the respective directories
long_save_name = "../numerical_data/results"
limit_filename = r"start_ensemble\limits.json"
with open(limit_filename, "r") as file:
    LIMITS = json.load(file)
#######################GLOBAL VARIABLES##############################
#drive coupling strength
#np.linspace(0.0, 1.0, 2, endpoint=True)
#temperature input (forced with generated overshoot inputs)
# temperature_trajs = np.loadtxt(r"temp_input\Tpeak_tconv_values\temp_input_values.txt", skiprows=1) #T_peak T_lim t_conv R mu_0 mu_1

def forcing_function(T_0, mu_0, mu_1, T_lim, R):
    """Returns the overshoot trajectory for given parameters as a function of t
    """
    return lambda t: overshoot_trajectory(t, T_0, T_lim, R, mu_0, mu_1)


def scale_forcing(forcing, conv_fac_gis):
    """Scale model time to forcing time."""
    return lambda t: forcing(t * conv_fac_gis)


COMPONENTS = ["GIS", "AMOC", "WAIS", "Amazonas", "REEF", "AWSI", "PERM", "WAM", "NINO"] # tipping elements need to be gathered at the start
########################Declaration of variables from passed values#######################
input_file = pd.read_csv(r"start_ensemble\latin_prob.txt", delimiter=",")
startdate = str(datetime.datetime.now().date())

#include uncertain "+-" links:
if plus_minus_include == False:
    plus_minus_links = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)))

    #in the NO_ENSO case (i.e., the second link must be 0.0)
    plus_minus_data = []
    for pm in plus_minus_links:
        if pm[1] == 0.0:
            plus_minus_data.append(pm)
    plus_minus_links = np.array(plus_minus_data)

else:
    plus_minus_links = [np.array([1., 1., 1.])]

def set_colormap(ax, data_length):
    """Sets the colormap for a new observation to the new colormap in the list
    copied from Plasway code

    Args:
        ax (_type_): plot area that uses the new colormap rule
        index (_type_): index of the observation that gets the new colormap
    """
    index = 0
    # Reset color-cycling to fresh scale
    gradient = np.linspace(0.01,1,data_length)
    ax.set_prop_cycle(plt.cycler("color", plt.cm.magma(gradient)))
    ax_colormap = inset_axes(ax, width="30%", height="2%", loc=("lower center"), borderpad=1.6+index*1.1)   # Inset that shows the cmaps for the cycle data. Must be located in a center coordinate so that the border hack works
    ax_colormap.imshow(np.vstack((gradient, gradient)), aspect="auto", cmap=plt.cm.magma, extent=[0,1,0,1])    
    ax_colormap.text(0.5, 1, "Interaction strength", va='bottom', ha='center', fontsize=10, transform=ax_colormap.transAxes)
    if index == 0:
        ax_colormap.axes.get_yaxis().set_visible(False) # only the lowest bar gets the bottom axis
        ax_colormap.grid(False)
    else: ax_colormap.set_axis_off()  

################################# MAIN LOOP #################################
@dataclass(frozen=True)
class SimulationCase:
    index_values: tuple
    strength: float = 1.0
    intervention: str | None = None
    intervention_state: float | None = None
    isolated_intervention: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    temperature_mode: str
    case_index_names: tuple[str, ...]
    cases: tuple[SimulationCase, ...]

    @property
    def index_names(self):
        return ["lhc", "T_peak", "T_lim", "t_conv", *self.case_index_names, "component"]


def model_strengths():
    strength_cases = tuple(
        SimulationCase(index_values=(strength,), strength=strength)
        for strength in (0, 1)
    )
    run_model_experiment(
        ExperimentSpec(
            name="strengths",
            temperature_mode="grid",
            case_index_names=("strength",),
            cases=strength_cases,
        )
    )


def model_interventions():
    """
    Performs do-interventions on the network with every tipping element,
    once tipped, once untipped, and once at the threshold.
    """
    intervention_cases = tuple(
        SimulationCase(
            index_values=(element, state, connection_strength),
            intervention=element,
            intervention_state=state,
            isolated_intervention=connection_strength == 0,
        )
        for element, state, connection_strength in itertools.product(COMPONENTS, [-1, 0, 1], [0, 1])
    )
    run_model_experiment(
        ExperimentSpec(
            name="intervention",
            temperature_mode="flat",
            case_index_names=("intervention", "state", "strength"),
            cases=intervention_cases,
        )
    )


def run_model_experiment(spec: ExperimentSpec):
    folder = prepare_folder(spec.name)
    tipping_df, timing_df = empty_result_frames(spec.index_names)
    temperature_inputs = prepare_temperature_inputs(mode=spec.temperature_mode)

    for lhc_index, sys_var in enumerate(tqdm(input_file.iterrows(), total=input_file.shape[0])):
        conv_fac_gis, base_earth_params = prepare_earth_params(sys_var)
        state_output = {}
        timing_output = {}

        for temperature_input, case in itertools.product(temperature_inputs, spec.cases):
            run_single_case(
                lhc_index,
                temperature_input,
                case,
                base_earth_params,
                conv_fac_gis,
                state_output,
                timing_output,
            )

        tipping_df = state_results_to_df(state_output, tipping_df)
        timing_df = timing_results_to_df(timing_output, timing_df)
        save_progress(folder, tipping_df, timing_df)

    input_file.to_csv(f"{folder}/tipping_properties.csv")
    print("Finish")


def empty_result_frames(index_names):
    index = pd.MultiIndex.from_arrays([[] for _ in index_names], names=index_names)
    return pd.DataFrame({}, index=index), pd.DataFrame({}, index=index)


def run_single_case(
    lhc_index,
    temperature_input,
    case: SimulationCase,
    base_earth_params,
    conv_fac_gis,
    state_output,
    timing_output,
):
    forcing = scale_forcing(temperature_input["forcing"], conv_fac_gis)
    earth_params = earth_params_for_case(base_earth_params, case)
    net, node_dict = earth_network(earth_params, forcing, case.strength, 1, 1, 1)
    initial_state = -1 * np.ones(len(net.nodes))

    if case.intervention is not None:
        net, initial_state = intervene_in_network(net, case.intervention, case.intervention_state, node_dict)

    state_results, tip_times = simulate_network(net, initial_state, conv_fac_gis)
    if np.any(np.isnan(state_results)):
        raise RuntimeError("NaN in solution")

    base_index = temperature_case_index(lhc_index, temperature_input, case)
    record_case_results(base_index, node_dict, state_results, tip_times, state_output, timing_output)


def earth_params_for_case(base_earth_params, case: SimulationCase):
    earth_params = base_earth_params.copy()
    if case.isolated_intervention:
        # Isolated interaction: probability factors are neutral except outgoing links from the intervention.
        for key in LIMITS.keys():
            if key.startswith("pf_") and not key.startswith(f"pf_{case.intervention}_"):
                earth_params[key] = 1
    return earth_params


def temperature_case_index(lhc_index, temperature_input, case: SimulationCase):
    return (
        lhc_index,
        temperature_input["T_peak"],
        temperature_input["T_lim"],
        temperature_input["t_conv"],
        *case.index_values,
    )


def record_case_results(base_index, node_dict, state_results, tip_times, state_output, timing_output):
    for component_index, component in enumerate(node_dict.keys()):
        result_index = (*base_index, component)
        state_output[result_index] = state_results[component_index]
        timing_output[result_index] = tip_times[component_index]
    state_output[(*base_index, "total")] = state_results[-1]


def save_progress(folder, tipping_df, timing_df):
    if DEBUGGING_MODE:
        return
    if not os.path.isdir(folder):
        os.makedirs(folder)
    tipping_df.to_csv(f"{folder}/dataframe.csv")
    timing_df.to_csv(f"{folder}/timing_dataframe.csv")


def simulate_network(net, initial_state, conv_fac_gis):
    # t_end given in years; also possible to use equilibrate method
    t_end = DURATION / conv_fac_gis  # simulation length in "real" years
    t_span = (0, t_end)
    t_eval = np.linspace(*t_span, N_STEPS + 1)
    # global TODO: arguments of f and jac have been swapped
    solution = solve_ivp(net.f, t_span, initial_state, t_eval=t_eval, jac=net.jac, method='LSODA',
                         events=[lambda t, x, i=i: x[i] for i in range(len(net.nodes))])
    sol = solution.y.T  # solve_ivp transposes everything
    t = solution.t
    # Big Savefile - around 100 MB. Too large for a csv, a pkl will make much less hassle
    tip_times = [t_event[0] * conv_fac_gis if len(t_event) > 0 else np.nan for t_event in solution.t_events]
    # tip_nino = [y_event[-1][0] if len(y_event)>0 else np.nan for y_event in solution.y_events]
    # dont use the total tipped anyway
    total_tipped = np.zeros(sol.shape[0])#np.array([net.get_number_tipped(timeseries) for timeseries in sol])
    state_results = np.concatenate((sol, total_tipped[:, np.newaxis]), axis=1).T
    return state_results, tip_times


def prepare_temperature_inputs(mode="random"):
    if TEMPERATURE_TRAJECTORY_KIND == "overshoot":
        return prepare_overshoot_inputs(mode)
    if TEMPERATURE_TRAJECTORY_KIND == "forcing":
        return prepare_projection_inputs()
    raise ValueError(
        "TEMPERATURE_TRAJECTORY_KIND must be 'overshoot' or 'forcing', "
        f"not '{TEMPERATURE_TRAJECTORY_KIND}'."
    )


def prepare_overshoot_inputs(mode="random"):
    T_0s, T_lims, T_peaks, t_convs = prepare_overshoots(mode=mode)
    temperature_inputs = []
    for T_0_iter, T_peak, T_lim, t_conv in zip(T_0s, T_peaks, T_lims, t_convs):
        try:
            R, mu_0, mu_1 = fit_parameters(T_0_iter, T_peak, T_lim, t_conv)
        except RuntimeError as error:
            print(f"{error}: Parameters T_0:{T_0_iter}, T_peak:{T_peak}, T_lim:{T_lim}, t_conv:{t_conv}")
            continue
        temperature_inputs.append({
            "T_peak": T_peak,
            "T_lim": T_lim,
            "t_conv": t_conv,
            "forcing": forcing_function(T_0_iter, mu_0, mu_1, T_lim, R),
        })
    return temperature_inputs


def prepare_projection_inputs():
    paths = projection_paths(FORCING_PROJECTION_PATH)
    temperature_inputs = []
    for path in paths:
        projections = load_projection_forcings(
            path,
            time_column=FORCING_TIME_COLUMN,
            temperature_columns=FORCING_TEMPERATURE_COLUMN,
            time_zero=FORCING_TIME_ZERO,
            extrapolation=FORCING_EXTRAPOLATION,
        )
        for projection in projections:
            temperature_inputs.append({
                "T_peak": projection["T_peak"],
                "T_lim": projection["T_lim"],
                "t_conv": projection["t_conv"],
                "forcing": projection["forcing"],
            })
    return temperature_inputs


def projection_paths(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        paths = []
        for pattern in ("*.csv", "*.txt", "*.tsv", "*.dat"):
            paths.extend(glob.glob(os.path.join(path, pattern)))
        if paths:
            return sorted(paths)
    raise FileNotFoundError(
        f"No forcing projection files found at '{path}'. Set FORCING_PROJECTION_PATH "
        "to a file or a directory containing csv/txt/tsv/dat projections."
    )


def prepare_overshoots(mode="random"):
    """
    Prepares parameters for the overshoots.
    Args:
        mode: str, "flat" or "random", fallback: "grid"

    Returns:

    """
    if mode == "flat":
        # Scenario with T_lims and T_peaks at 1.5, 2, 2.5 and t_convs at 0
        # T_0 varies in lockstep with T_lim and T_peak for flat trajectories
        T_0s = np.array([1.5, 2.0, 2.5])
        T_peaks = np.array([1.5, 2.0, 2.5])
        T_lims = np.array([1.5, 2.0, 2.5])
        t_convs = np.array([0.0, 0.0, 0.0])
    elif mode == "random":
        lhc_distr = np.array(lhs(3, samples=N_OVERSHOOTS))
        T_peaks = np.round(4 * lhc_distr[:, 0] + 2, 2)
        # T_lims = np.round(2*lhc_distr[:,1], 2)
        T_lims = np.round(2 * lhc_distr[:, 1], 2)  # for plotting in the T_lim plane
        t_convs = np.round(900 * lhc_distr[:, 2] + 100, 0)
        T_0s = np.full_like(T_peaks, T_0)  # Use global T_0 for non-flat scenarios
    else:
        T_peaks_grid = np.arange(2., 6.1, 0.5)
        T_lims_grid = np.arange(0, 2.1, 0.5)
        t_convs_grid = np.arange(100, 1001, 100)
        T_peaks_mesh, T_lims_mesh, t_convs_mesh = np.meshgrid(T_peaks_grid, T_lims_grid, t_convs_grid, indexing='ij')
        T_peaks = T_peaks_mesh.flatten()
        T_lims = T_lims_mesh.flatten()
        t_convs = t_convs_mesh.flatten()
        T_0s = np.full_like(T_peaks, T_0)  # Use global T_0 for non-flat scenarios
    return T_0s, T_lims, T_peaks, t_convs


def prepare_earth_params(sys_var):
    earth_params_raw = sys_var[1].to_dict()

    # Time scale
    if time_scale:
        # print("compute calibration timescale")
        # function call for absolute timing and time conversion
        earth_params = individual_timescales(earth_params_raw)
        conv_fac_gis = 1.0
    else:
        # no time scales included
        earth_params = earth_params_raw
        earth_params["GIS_time"], earth_params["Amazonas_time"], earth_params["NINO_time"], earth_params["AMOC_time"], \
            earth_params["WAIS_time"] = 1.0, 1.0, 1.0, 1.0, 1.0
        conv_fac_gis = 1.0
    return conv_fac_gis, earth_params


def prepare_folder(experiment_name):
    """
    Create the name of a new folder listing the date and the progressive index of the experiment
    Args:
        experiment_name:

    Returns:

    """
    folder = f"{long_save_name}/{experiment_name}/{startdate}_1"
    if TEMPERATURE_TRAJECTORY_KIND == "forcing":
        folder = f"{long_save_name}/{experiment_name}/{startdate}_{Path(FORCING_PROJECTION_PATH).stem}_1"
    while os.path.isdir(f"{folder}"):
        match = re.search(r"_(\d+)$", folder)
        pos = match.span()
        new_suffix = int(match.group(1)) + 1
        folder = folder[:pos[0]] + "_" + str(new_suffix)
    return folder


def timing_results_to_df(timing_output, timing_df):
    new_timing_df = pd.DataFrame.from_dict(
        timing_output,
        orient="index",
        columns=["values"]
    ).dropna()
    new_timing_df.index = pd.MultiIndex.from_tuples(
        new_timing_df.index,
        names=timing_df.index.names
    )
    timing_df = pd.concat([timing_df, new_timing_df])
    return timing_df


def state_results_to_df(state_output, tipping_df):
    new_tipping_df = pd.DataFrame.from_dict(
        state_output,
        orient="index",
        columns=np.linspace(0, DURATION, N_STEPS + 1)
    )
    new_tipping_df = new_tipping_df[[1000, DURATION]]
    new_tipping_df.index = pd.MultiIndex.from_tuples(
        new_tipping_df.index,
        names=tipping_df.index.names
    )
    tipping_df = pd.concat([tipping_df, new_tipping_df])
    return tipping_df

def make_state_index(i, T_lims, T_peaks,  t_convs, tipping_df, components, **index_levels):
    """
    index_levels: arbitrary keyword arguments where each key is an index level name
                  and each value is the list/array of components for that level.
                  e.g. coupling_strength=coupling_strengths, component=["GIS", "AMOC", ...]
    """
    temperature_index = pd.MultiIndex.from_arrays([T_peaks, T_lims, t_convs], names=tipping_df.index.names[1:4])
    temp_df = temperature_index.to_frame(index=False)

    merged = (
        pd.DataFrame({tipping_df.index.names[0]: [i]})
        .merge(temp_df, how="cross")
    )
    for level_name, level_values in index_levels.items():
        merged = merged.merge(pd.DataFrame({level_name: level_values}), how="cross")
    merged = merged.merge(pd.DataFrame({"component": components}), how="cross")
    state_index = pd.MultiIndex.from_frame(merged)
    return state_index

def rename_characteristic(temperature_trajs,):
    filenames = glob.glob(f"results/no_feedbacks/network_1.0_1.0_1.0/*.npy")
    temp_df = pd.DataFrame(temperature_trajs, columns=["Tpeak", "Tlim", "tconv", "R", "mu0", "mu1"])
    temp_idx_df = temp_df.set_index(["Tlim", "Tpeak", "tconv"])
    for filename in filenames:
        m = re.search(
            r'Tlim(?P<Tlim>[0-9]+(?:\.[0-9]+)?)_'
            r'Tpeak(?P<Tpeak>[0-9]+(?:\.[0-9]+)?)_'
            r'tconv(?P<tconv>[0-9]+(?:\.[0-9]+)?)',
            filename
        )
        vals = {k: float(v) for k, v in m.groupdict().items()}
        row = temp_idx_df.loc[tuple(vals[k] for k in temp_idx_df.index.names)]
        forcing = forcing_function(1.0, row["mu0"], row["mu1"], vals["Tlim"], row["R"])
        characteristic = np.round((quad(forcing, 0, 500)[0]), 2)
        new_filename = re.sub(r'([0-9]+\.[0-9]+)(?!.*[0-9]+\.[0-9]+)', str(characteristic), filename)

        os.rename(filename, new_filename)


DEBUGGING_MODE = sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None
if __name__ == "__main__":
    #cProfile.run("model_interventions()", "new_profile.prof", sort="cumtime")

    model_interventions()
