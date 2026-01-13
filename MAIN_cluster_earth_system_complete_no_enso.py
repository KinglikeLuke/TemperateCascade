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
import time
import glob
from PyPDF2 import PdfMerger
from netCDF4 import Dataset
import cProfile
from scipy.integrate import odeint, quad
from tqdm import tqdm

# private imports from sys.path
from core.evolve import evolve

#private imports for earth system
from earth_sys.timing_no_enso import timing
from earth_sys.functions_earth_system_no_enso import global_functions
from earth_sys.earth_no_enso import EarthParams, earth_network

#measure time
#start = time.time()
#############################GLOBAL SWITCHES#########################################
time_scale = True            # time scale of tipping is incorporated
plus_minus_include = True    # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
######################################################################
duration = 50000 #actual real simulation years

params = [0.9183029815394349, 2.3885370105887995, 1.992801071295943, 3.9887964479687965, 4.480642921936935, 3., 
          0.17987908345005993, 0.8406125344797735, 
          0.37982673798028066, 0.19622466304037592, 0.21995682709593467, 0.2, 
          0.2507240489292405, 0.12155858668779652, 0.17717806892447752, 
          0.15796027793553247, 0.1346368579815631, 
          0.41619629099617483, 0.38748518389882447, 
          0.2, 
          7757.23023703856, 72.19192428443324, 6501.990656966239, 96.92296797423056, 182.25839860429673, 100., 0000]


#Names to create the respective directories
namefile = "no"
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
coupling_strength = np.linspace(0.0, 1.0, 11, endpoint=True)
#temperature input (forced with generated overshoot inputs)
temperature_trajs = np.loadtxt(r"temp_input\Tpeak_tconv_values\temp_input_values.txt", comments=['#']) #T_peak T_lim t_conv R mu_0 mu_1

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

# Tipping ranges from distribution
keys = [
    'limits_gis','limits_thc','limits_wais','limits_amaz','limits_nino', 'limits_assi',
    'pf_wais_to_gis','pf_thc_to_gis',
    'pf_gis_to_thc','pf_nino_to_thc','pf_wais_to_thc', 'pf_assi_to_thc',
    'pf_nino_to_wais','pf_thc_to_wais','pf_gis_to_wais',
    'pf_thc_to_nino',
    'pf_nino_to_amaz', 'pf_thc_to_amaz',
    'pf_thc_to_assi',
    'gis_time','thc_time','wais_time','nino_time','amaz_time', 'assi_time'
]
########################Declaration of variables from passed values#######################
sys_var = params # np.array(sys.argv[2:], dtype=str) #low sample -3, intermediate sample: -2, high sample: -1
input_file = np.loadtxt(r"start_ensemble\latin_prob.txt",
                        delimiter=" ")



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
    final_results = {}
    for kk in plus_minus_links:
        print("Wais to Thc:{}".format(kk[0]))
        print("Amaz to Nino:{}".format(kk[1]))
        print("Thc to Amaz:{}".format(kk[2]))
        try:
            os.stat("{}".format(long_save_name))
        except:
            os.makedirs("{}".format(long_save_name))

        try:
            os.stat("{}/{}_feedbacks".format(long_save_name, namefile))
        except:
            os.mkdir("{}/{}_feedbacks".format(long_save_name, namefile))

        try:
            os.stat("{}/{}_feedbacks/network_{}_{}_{}".format(long_save_name, namefile, kk[0], kk[1], kk[2]))
        except:
            os.mkdir("{}/{}_feedbacks/network_{}_{}_{}".format(long_save_name, namefile, kk[0], kk[1], kk[2]))

        # try:
        #     os.stat("{}/{}_feedbacks/network_{}_{}_{}/{}".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4) ))
        # except:
        #     os.mkdir("{}/{}_feedbacks/network_{}_{}_{}/{}".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4) ))

        #save starting conditions
        # np.savetxt("{}/{}_feedbacks/network_{}_{}_{}/{}/empirical_values.txt".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4)), sys_var, delimiter=" ", fmt="%s")

        for i in temperature_trajs[1:2]:
            T_0 = 1.0
            T_peak_det = i[0]
            T_lim = i[1]
            t_conv_det = i[2]
            R = i[3]
            mu_0 = i[4]
            mu_1 = i[5]
            key = f"T_peak:{T_peak_det},T_lim:{T_lim},t_conv:{t_conv_det}"
            print("T_lim: {}°C".format(T_lim))
            print("T_peak: {}°C".format(T_peak_det))
            print("t_conv: {}yrs".format(t_conv_det))
            
            n_steps = 500
            avg_output = []
            std_output = []
            
            for strength in coupling_strength:
                print("Coupling strength: {}".format(strength))
                # How many points are to be calculated. odeint's precision is mostly independent of this, taking adaptive steps
                output = np.zeros((input_file.shape[0], 8, n_steps))

                for i, sys_var in enumerate(tqdm(input_file)):
                    values = list(map(float, sys_var)) # -1 is the mc_dir
                    params_dict = dict(zip(keys, values))
                    earth_params_raw = EarthParams(**params_dict)
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
                        earth_params.gis_time, earth_params.amaz_time, earth_params.assi_time, earth_params.nino_time, earth_params.thc_time, \
                            earth_params.wais_time = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                        conv_fac_gis = 1.0
                    
                    
                    path = "{}/{}_feedbacks/network_{}_{}_{}/feedbacks_Tlim{}_Tpeak{}_tconv{}_{:.2f}".format(long_save_name, 
                        namefile, kk[0], kk[1], kk[2], T_lim, T_peak_det, t_conv_det, strength)
                    if os.path.isfile(path) == True:
                        abs_path = os.path.abspath(path)
                        print(abs_path)
                        print("File already computed")
                        #break
                    #For feedback computations

                    # scale the tempearture properly
                    forcing = lambda t: forcing_function(T_0, mu_0, mu_1, T_lim, R)(t*conv_fac_gis)
                    net = earth_network(earth_params, forcing, strength, kk[0], kk[1], kk[2])
                    # initialize state
                    initial_state = [-1, -1, -1, -1, -1, -1] #initial state
                    # plotter.network(net)

                    #t_end given in years; also possible to use equilibrate method
                    t_end = duration/conv_fac_gis # simulation length in "real" years
                    t_span = np.linspace(0, t_end, n_steps)
                    sol = odeint( net.f , initial_state, t_span, Dfun=net.jac )
                    total_tipped = np.array([net.get_number_tipped(timeseries) for timeseries in sol])
                    #saving structure
                    output[i] = np.concatenate((conv_fac_gis*t_span[:,np.newaxis], sol, total_tipped[:,np.newaxis]), axis=1).T

                
                ensemble_avg = np.mean(output, axis=0)
                ensemble_std = np.std(output, axis=0, ddof=1)
                avg_output.append(ensemble_avg)
                std_output.append(ensemble_std)

                
            #necessary for break condition
            if len(avg_output):
                characteristic = quad(forcing, 0, t_end)
                output = [] # mostly so it doesnt annoy me in debugger
                data = np.array(avg_output)
                t_grid = data[0,0]
                # What do I want to see? Avgs and stds
                # I want to see how many elements tipped after 100, 1000, 50000 years
                close_index = np.argmin(np.abs(t_grid-100))
                medium_index = np.argmin(np.abs(t_grid-1000))
                far_index = -1
                # How this differs between different strengths
                # Eventually disregard non-tipping elements
                np.save(f"{path}_close", data[:,:,close_index])
                np.save(f"{path}_medium", data[:,:,medium_index])
                np.save(f"{path}_far", data[:,:,far_index])
                np.save(f"{path}_total_tipped", data[:,7])
                fig, ax = plt.subplots()
                set_colormap(ax, data.shape[0])
                for i, data_strength in enumerate(data):
                    ax.plot(t_grid, data_strength[7], label=f"Interactions: 0.{i}")
                ax.set_title(f"Temperature properties: {key}")
                ax.set_xlabel("Time [yr]")
                ax.set_ylabel("Tipped elements")
                # ax.legend(loc='best')  # , ncol=5)
                fig.savefig("{}/{}_feedbacks/network_{}_{}_{}/feedbacks_Tlim{}_Tpeak{}_tconv{}.pdf".format(long_save_name, namefile, 
                    kk[0], kk[1], kk[2], T_lim, T_peak_det, t_conv_det))
                #plt.show()
                plt.clf()
                plt.close()



        current_dir = os.getcwd()
        os.chdir("{}/{}_feedbacks/network_{}_{}_{}/".format(long_save_name, namefile, kk[0], kk[1], kk[2]))
        pdfs = np.array(np.sort(glob.glob("feedbacks_*.pdf"), axis=0))
        if len(pdfs) != 0.:
            merger = PdfMerger()
            for pdf in pdfs:
                merger.append(pdf)
            merger.write("feedbacks_complete.pdf")
            merger.close()
            for filename in glob.glob("time_d*.pdf"):
                os.remove(filename)
            print("Complete PDFs merged")
        os.chdir(current_dir)

    print("Finish")
main()
# Good lord
# The original Code steps in 0.1 (absolute? idk) year steps through the solver (because the stepsize is far greater than the calibrated(?) t_end)
# However, it takes its Temperature curve as if it made 1 year steps (every step a new year)
# So the Temperature changes at 10x rate and the x-label is 10x too large. N.B. the concrete conversion factor is irrelevant in the original code as long as >10
# My own code is consitent between timescales and shows a tipped GIS after only 200 years of 2C warming (which is a bit sketchy) - probably because one has to
# re-norm the Temperature curve