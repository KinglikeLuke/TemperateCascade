# Add modules directory to path
import os
import sys
import re
import json

sys.path.append('')

# global imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set(font_scale=1.)
import itertools
import datetime
import glob
# from PyPDF2 import PdfMerger # dont really know what I would want with the merged PDFs 
from scipy.integrate import solve_ivp, quad
from tqdm import tqdm
import pandas as pd
from pydoe import lhs

# private imports from sys.path

#private imports for earth system
from earth_sys.timing_no_enso import individual_timescales
from earth_sys.earth_no_enso import earth_network, intervene_in_network

from temp_input.overshoot_trajectory import overshoot_trajectory, fit_parameters

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
def model_strengths():
    # Saving, folder creation only once first run goes through
    coupling_strengths = [0, 1]
    directory_name = prepare_folder("strengths")

    index = pd.MultiIndex.from_arrays(
        [[], [], [], [], [], []],
        names=["lhc", "T_peak", "T_lim", "t_conv", "strength", "component"]
    )
    tipping_df = pd.DataFrame({}, index=index)
    timing_df = pd.DataFrame({"tip_time": pd.Series(dtype="float64")}, index=index)

    for i, sys_var in enumerate(tqdm(input_file.iterrows(), total=input_file.shape[0])):
        conv_fac_gis, earth_params = prepare_earth_params(sys_var)
        T_0s, T_lims, T_peaks, t_convs = prepare_overshoots(mode = "grid")
        state_output = {}
        timing_output = {}
        T_index = 0
        for T_0_iter, T_peak, T_lim, t_conv in zip(T_0s, T_peaks, T_lims, t_convs):
            try:
                R, mu_0, mu_1 = fit_parameters(T_0_iter, T_peak, T_lim, t_conv)
            except RuntimeError as error:
                print(f"{error}: Parameters T_0:{T_0_iter}, T_peak:{T_peak}, T_lim:{T_lim}, t_conv:{t_conv}")
                T_0s = np.delete(T_0s, T_index)
                T_peaks = np.delete(T_peaks, T_index)
                T_lims = np.delete(T_lims, T_index)
                t_convs = np.delete(t_convs, T_index)
                continue
            T_index += 1
            for j, strength in enumerate(coupling_strengths):
                # How many points are to be calculated. odeint's precision is mostly independent of this, taking adaptive steps
                # scale the temperature properly
                forcing = lambda t: forcing_function(T_0_iter, mu_0, mu_1, T_lim, R)(t * conv_fac_gis)
                net, node_dict = earth_network(earth_params, forcing, strength, 1, 1, 1) # here be ks
                initial_state = -1*np.ones(len(net.nodes)) #initial state

                state_results, tip_times = simulate_network(net, initial_state, conv_fac_gis)
                # saving structure: configuration x features x time
                if np.any(np.isnan(state_results)):
                    raise RuntimeError("NaN in solution")
                for l, component in enumerate(COMPONENTS):
                    state_output[i, T_peak, T_lim, t_conv, strength, component] = state_results[l]
                    timing_output[i, T_peak, T_lim, t_conv, strength, component] = tip_times[l]
                state_output[i, T_peak, T_lim, t_conv, strength, "total"] = state_results[-1]
        tipping_df = state_results_to_df(state_output, tipping_df)
        timing_df = timing_results_to_df(timing_output, timing_df)

        if not DEBUGGING_MODE: # to stop it from cluttering my workspace with folders
            if not os.path.isdir(directory_name): # if this is the first time we get here - whether the name fits is decided before the run
                os.makedirs(directory_name)
            tipping_df.to_csv(f"{directory_name}/dataframe.csv")
            timing_df.to_csv(f"{directory_name}/timing_dataframe.csv")

    input_file.to_csv(f"{directory_name}/tipping_properties.csv")

    # current_dir = os.getcwd()
    # os.chdir("{}/network_{}_{}_{}/".format(long_save_name, kk[0], kk[1], kk[2]))
    # pdfs = np.array(np.sort(glob.glob("feedbacks_*.pdf"), axis=0))
    # if len(pdfs) != 0.:
    #     merger = PdfMerger()
    #     for pdf in pdfs:
    #         merger.append(pdf)
    #     merger.write("feedbacks_complete.pdf")
    #     merger.close()
    #     for filename in glob.glob("time_d*.pdf"):
    #         os.remove(filename)
    #     print("Complete PDFs merged")
    # os.chdir(current_dir)

    # idk what this does. seems like a worse version of the upper saveloop
    # for keyword in ["close", "medium", "far"]:
    #     filenames = glob.glob(f"{path}/*{keyword}.npy")
    #     n_temps = len(filenames)
    #     results_array = np.zeros((n_temps, *np.load(filenames[0]).shape))
    #     characteristics =  np.zeros(n_temps)
    #     for i, filename in enumerate(filenames):
    #         data = np.load(filename)
    #         match = re.search(r'([0-9]+\.[0-9]+)(?!.*[0-9]+\.[0-9]+)', filename)
    #         results_array[i] = data
    #         characteristic = match.group(1)
    #         characteristics[i] = float(characteristic)
    #     strengths = np.arange(0.0, 1.01, 0.1)
    #     components = ["time", "gis", "thc", "wais", "amaz", "nino", "total"]
    #     index = pd.MultiIndex.from_product([characteristics, strengths, components], names=["characteristic", "strength", "component"])
    #     results_df = pd.DataFrame({"value": results_array.reshape(-1)}, index=index)
    #     results_df = results_df.drop(index="time", level="component")
    #     results_df.to_csv(f"{keyword}.csv")

    print("Finish")

def model_interventions():
    """
    Performs do-interventions on the network with every tipping element, once tipped, once untipped.
    Returns:

    """
    interventions = COMPONENTS[:-1] # Amazonas has no outgoing connections. Therefore, it cannot have a causal effect
    intervention_states = [-1, 0, 1]
    folder = prepare_folder("intervention")

    index = pd.MultiIndex.from_arrays(
        [[], [], [], [], [], [], [], []],
        names=["lhc", "T_peak", "T_lim", "t_conv", "intervention", "state", "strength", "component"]
    )
    tipping_df = pd.DataFrame({}, index=index)
    timing_df = pd.DataFrame({}, index=index)

    T_0s, T_lims, T_peaks, t_convs = prepare_overshoots(mode = "flat")
    for i, sys_var in enumerate(tqdm(input_file.iterrows(), total=input_file.shape[0])):
        conv_fac_gis, earth_params = prepare_earth_params(sys_var)

        state_output = {}
        timing_output = {}
        T_index = 0
        for T_0_iter, T_peak, T_lim, t_conv in zip(T_0s, T_peaks, T_lims, t_convs):
            try:
                R, mu_0, mu_1 = fit_parameters(T_0_iter, T_peak, T_lim, t_conv)
            except RuntimeError as error:
                print(f"{error} Parameters T_0:{T_0_iter}, T_peak:{T_peak}, T_lim:{T_lim}, t_conv:{t_conv}")
                T_0s = np.delete(T_0s, T_index)
                T_peaks = np.delete(T_peaks, T_index)
                T_lims = np.delete(T_lims, T_index)
                t_convs = np.delete(t_convs, T_index)
                continue
            T_index += 1
            for j, intv_element in enumerate(interventions):
                for k, intv_state in enumerate(intervention_states):
                    for intv_con_strength in (0, 1):
                        earth_params_original = earth_params.copy()
                        for key in LIMITS.keys():
                            if key.startswith(f"pf_") and not key.startswith(f"pf_{intv_element}"):
                                # 0 means isolated interaction (PF of 1 everywhere but intervention),
                                # 1 means normal (maximally strong) interaction
                                if intv_con_strength == 0:
                                    earth_params[key] = 1
                                # else
                                #   earth_params[key] = LIMITS[key][np.argmax(np.abs(np.array(LIMITS[key]) - 1))]
                        # scale the temperature properly
                        forcing = lambda t: forcing_function(T_0_iter, mu_0, mu_1, T_lim, R)(t * conv_fac_gis)
                        net, node_dict = earth_network(earth_params, forcing, strength=1, kk0=1, kk1=1, kk2=1)
                        # TODO TEST!!!
                        net, initial_state = intervene_in_network(net, intv_element, intv_state, node_dict)

                        state_results, tip_times = simulate_network(net, initial_state, conv_fac_gis)
                        #saving structure: configuration x features x time
                        if np.any(np.isnan(state_results)):
                            raise RuntimeError("NaN in solution")
                        # TODO TEST!!!
                        for l, component in enumerate(COMPONENTS):
                            state_output[i, T_peak, T_lim, t_conv, intv_element, intv_state, intv_con_strength, component] \
                                = state_results[l]
                            timing_output[i, T_peak, T_lim, t_conv, intv_element, intv_state, intv_con_strength, component] \
                                = tip_times[l]
                        state_output[i, T_peak, T_lim, t_conv, intv_element, intv_state, intv_con_strength, "total"] \
                            = state_results[-1]
                        earth_params = earth_params_original

            # state_output.append(state_record) # shape: temp_dim x interventions x components x time
            # timing_output.append(timing_record) # shape: temp_dim x interventions x components

        tipping_df = state_results_to_df(state_output, tipping_df) # these modify the df in place
        timing_df = timing_results_to_df(timing_output, timing_df)
        # TODO somethings fucked with the Amazon intervetion
        if not DEBUGGING_MODE: # to stop it from cluttering my workspace with folders
            if not os.path.isdir(folder): # if this is the first time we get here - whether the name fits is decided before the run
                os.makedirs(folder)
            tipping_df.to_csv(f"{folder}/dataframe.csv")
            timing_df.to_csv(f"{folder}/timing_dataframe.csv")

    input_file.to_csv(f"{folder}/tipping_properties.csv")

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
    total_tipped = np.array([net.get_number_tipped(timeseries) for timeseries in sol])
    state_results = np.concatenate((sol, total_tipped[:, np.newaxis]), axis=1).T
    return state_results, tip_times


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
    new_tipping_df = new_tipping_df[[1000, 50000]]
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
    model_interventions()
# Good lord
# The original Code steps in 0.1 (absolute? idk) year steps through the solver (because the stepsize is far greater than the calibrated(?) t_end)
# However, it takes its Temperature curve as if it made 1 year steps (every step a new year)
# So the Temperature changes at 10x rate and the x-label is 10x too large. N.B. the concrete conversion factor is irrelevant in the original code as long as >10
# My own code is consitent between timescales and shows a tipped GIS after only 200 years of 2C warming (which is a bit sketchy) - probably because one has to
# re-norm the Temperature curve
