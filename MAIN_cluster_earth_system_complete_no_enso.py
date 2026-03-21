# Add modules directory to path
import os
import sys
import re

sys.path.append('')

# global imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set(font_scale=1.)
import itertools
import datetime
import glob
# from PyPDF2 import PdfMerger # dont really know what I would want with the merged PDFs 
from netCDF4 import Dataset
import cProfile
from scipy.integrate import solve_ivp, quad
from tqdm import tqdm
import pandas as pd
from pydoe import lhs

# private imports from sys.path
from core.evolve import evolve

#private imports for earth system
from earth_sys.timing_no_enso import timing
from earth_sys.functions_earth_system_no_enso import global_functions
from earth_sys.earth_no_enso import earth_network

from overshoot_trajectory import overshoot_trajectory, fit_parameters

#measure time
#start = time.time()
#############################GLOBAL SWITCHES#########################################
time_scale = True            # time scale of tipping is incorporated
plus_minus_include = True    # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
######################################################################
duration = 50000 #actual real simulation years
#Names to create the respective directories
long_save_name = "../numerical_data/results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
coupling_strengths = np.linspace(0.0, 1.0, 2, endpoint=True)
#temperature input (forced with generated overshoot inputs)
temperature_trajs = np.loadtxt(r"temp_input\Tpeak_tconv_values\temp_input_values.txt", skiprows=1) #T_peak T_lim t_conv R mu_0 mu_1

def forcing_function(T_0, mu_0, mu_1, T_lim, R):
    """Returns the overshoot trajectory for given parameters as a function of t
    """
    return lambda t: overshoot_trajectory(t, T_0, T_lim, R, mu_0, mu_1)

# Tipping ranges from distribution
KEYS = [
    'limits_gis','limits_thc','limits_wais','limits_amaz','limits_nino', 
    'pf_wais_to_gis','pf_thc_to_gis',
    'pf_gis_to_thc','pf_nino_to_thc','pf_wais_to_thc', 
    'pf_nino_to_wais','pf_thc_to_wais','pf_gis_to_wais',
    'pf_nino_to_amaz', 'pf_thc_to_amaz',
    'pf_thc_to_nino',
    'gis_time','thc_time','wais_time','nino_time','amaz_time'
]
########################Declaration of variables from passed values#######################
input_file = np.loadtxt(r"start_ensemble\latin_prob.txt", delimiter=" ")
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
def main():    
    # Saving, folder creation only once first run goes through
    folder = f"{long_save_name}/network_full_range/{startdate}_1"
    while os.path.isdir(f"{folder}"):
        match = re.search(r"(?:_)(\d+)$", folder)
        pos = match.span()
        new_suffix = int(match.group(1)) + 1
        folder = folder[:pos[0]] + "_" + str(new_suffix)
    
    # try:
    #     os.stat("{}/network_{}_{}_{}".format(long_save_name, kk[0], kk[1], kk[2]))
    # except:
    #     os.mkdir("{}/network_{}_{}_{}".format(long_save_name, kk[0], kk[1], kk[2]))

    # try:
    #     os.stat("{}/{}_feedbacks/network_{}_{}_{}/{}".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4) ))
    # except:
    #     os.mkdir("{}/{}_feedbacks/network_{}_{}_{}/{}".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4) ))

    #save starting conditions
    # np.savetxt("{}/{}_feedbacks/network_{}_{}_{}/{}/empirical_values.txt".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4)), sys_var, delimiter=" ", fmt="%s")
    
    index = pd.MultiIndex.from_arrays(
        [[], [], [], [], []],
        names=["lhc", "year", "temp_props", "strength", "component"]
    )
    tipping_df = pd.DataFrame({"value": pd.Series(dtype="float64")}, index=index)
    timing_df = pd.DataFrame({"value": pd.Series(dtype="float64")}, index=index.droplevel('year'))
    T_0 = 1.0
    n_overshoots = 400
    for i, sys_var in enumerate(tqdm(input_file)):
        values = list(map(float, sys_var)) # -1 is the mc_dir
        if len(KEYS) != len(values):
            raise KeyError("KEYS and LHS seed dont match!")
        earth_params_raw = dict(zip(KEYS, values))

        # Time scale
        if time_scale == True:
            # print("compute calibration timescale")
            # function call for absolute timing and time conversion
            time_props = timing(earth_params_raw)
            earth_params = time_props.timescales()
            conv_fac_gis = time_props.conversion()
        else:
            #no time scales included
            earth_params = earth_params_raw
            earth_params["gis_time"], earth_params["amaz_time"], earth_params["nino_time"], earth_params["thc_time"], \
                earth_params["wais_time"] = 1.0, 1.0, 1.0, 1.0, 1.0
            conv_fac_gis = 1.0
        
        lhc_distr = np.array(lhs(3, samples=n_overshoots))
        T_peaks = np.round(4*lhc_distr[:,0] + 2, 2)
        T_lims = np.round(2*lhc_distr[:,1], 2)
        t_convs = np.round(900*lhc_distr[:,2] + 100, 0)
        state_output = []
        timing_output = []

        for T_peak, T_lim, t_conv in zip(T_peaks, T_lims, t_convs):
            R, mu_0, mu_1 = fit_parameters(T_0, T_peak, T_lim, t_conv)         
            n_steps = 1000
            output = np.zeros((len(coupling_strengths), 2 + 5, n_steps)) # 2 + number of elements (difficult to get to at this stage)
            tipping_record = np.zeros((len(coupling_strengths), 5))

            for j, strength in enumerate(coupling_strengths):
                # print("Coupling strength: {}".format(strength))
                # How many points are to be calculated. odeint's precision is mostly independent of this, taking adaptive steps

                # scale the temperature properly
                forcing = lambda t: forcing_function(T_0, mu_0, mu_1, T_lim, R)(t*conv_fac_gis)
                net = earth_network(earth_params, forcing, strength, 1, 1, 1) # here be ks
                
                # initialize state
                initial_state = -1*np.ones(len(net.nodes)) #initial state
                # plotter.network(net)

                #t_end given in years; also possible to use equilibrate method
                t_end = duration/conv_fac_gis # simulation length in "real" years
                t_span = (0, t_end)
                t_eval = np.linspace(*t_span, n_steps)
                # global TODO: arguments of f and jac have been swapped
                solution = solve_ivp(net.f, t_span, initial_state, t_eval=t_eval, jac=net.jac, method='LSODA', events = [lambda t, x, i=i: x[i] for i in range(len(net.nodes))])
                sol = solution.y.T # solve_ivp transposes everything
                t = solution.t
                # Big Savefile - around 100 MB. Too large for a csv, a pkl will make much less hassle
                tip_times = [t_event[0] if len(t_event)>0 else np.nan for t_event in solution.t_events]
                tipping_record[j] = tip_times
                total_tipped = np.array([net.get_number_tipped(timeseries) for timeseries in sol])
                #saving structure: configuration x features x time
                if np.any(np.isnan(sol)):
                    raise RuntimeError("NaN in solution")
                output[j] = np.concatenate((conv_fac_gis*t[:,np.newaxis], sol, total_tipped[:,np.newaxis]), axis=1).T

            
            # ensemble_avg = np.mean(output, axis=0)
            # ensemble_std = np.std(output, axis=0, ddof=1)
            # std_output.append(ensemble_std)
            state_output.append(output) # shape: temp_dim x 2 x components x time
            timing_output.append(tipping_record) # shape: 2 x temp_dim x components

        # characteristic = int(quad(forcing, 0, 1000/conv_fac_gis)[0]*conv_fac_gis) # renorm to account for different conv facs between parametrisations
        output = [] # mostly so it doesnt annoy me in debugger
        # structure: strength x features x time
        data = np.array(state_output)
        t_grid = data[0,0,0]
        data_no_time = data[:,:,1:,:]
        components = ["GIS", "AMOC", "WAIS", "Amazonas", "nino", "total"]
        years = [100, 1000, 50000]
        temperature_index = pd.MultiIndex.from_arrays([T_lims, T_peaks, t_convs])
        tipping_index = pd.MultiIndex.from_product(
            [[i],
            years,
            temperature_index,
            coupling_strengths,
            components],
            names=tipping_df.index.names
        )
        timing_index = pd.MultiIndex.from_product(
            [[i],
            temperature_index,
            coupling_strengths,
            components[:-1]],
            names=timing_df.index.names
        )
        # I want to see how many elements tipped after 100, 1000, 50000 years
        close_index = np.argmin(np.abs(t_grid-100))
        medium_index = np.argmin(np.abs(t_grid-1000))
        far_index = -1
        new_tipping_df = pd.DataFrame(
            {"value": np.round(np.array([data_no_time[:,:,:,close_index],data_no_time[:,:,:,medium_index],data_no_time[:,:,:,far_index]]).flatten(), 2)},
            index=tipping_index,
        )
        tipping_df = pd.concat([tipping_df, new_tipping_df])
        new_timing_df = pd.DataFrame(
            {"value": np.round(np.array(timing_output).flatten(), 3)},
            index=timing_index,
        )
        new_timing_df.dropna(inplace=True)
        timing_df = pd.concat([timing_df, new_timing_df])
        if not DEBUGGING_MODE: # to stop it from cluttering my workspace with folders
            if not os.path.isdir(folder): # if this is the first time we get here - whether the name fits is decided before the run
                os.makedirs(folder)
            tipping_df.to_csv(f"{folder}/dataframe.csv")
            timing_df.to_csv(f"{folder}/timing_dataframe.csv")
            # np.save(f"{path}_{characteristic}_total_tipped", data[:,-1])
            # fig, ax = plt.subplots()
            # set_colormap(ax, data.shape[0])
            # for temperature_traj, data_strength in enumerate(data):
            #     ax.plot(t_grid, data_strength[-1], label=f"Interactions: 0.{temperature_traj}")
            # ax.set_title(f"Temperature properties: {key}")
            # ax.set_xlabel("Time [yr]")
            # ax.set_ylabel("Tipped elements")
            # # ax.legend(loc='best')  # , ncol=5)
            # fig.savefig(f"{path}.pdf")
            # #plt.show()
            # plt.clf()
            # plt.close()



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

def rename_characteristic():
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
main()
# Good lord
# The original Code steps in 0.1 (absolute? idk) year steps through the solver (because the stepsize is far greater than the calibrated(?) t_end)
# However, it takes its Temperature curve as if it made 1 year steps (every step a new year)
# So the Temperature changes at 10x rate and the x-label is 10x too large. N.B. the concrete conversion factor is irrelevant in the original code as long as >10
# My own code is consitent between timescales and shows a tipped GIS after only 200 years of 2C warming (which is a bit sketchy) - probably because one has to
# re-norm the Temperature curve