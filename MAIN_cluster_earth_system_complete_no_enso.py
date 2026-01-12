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
import seaborn as sns
sns.set(font_scale=1.)
import itertools
import time
import glob
from PyPDF2 import PdfMerger
from netCDF4 import Dataset
import cProfile
from scipy.integrate import odeint
from tqdm import tqdm

# private imports from sys.path
from core.evolve import evolve

#private imports for earth system
from earth_sys.timing_no_enso import timing
from earth_sys.functions_earth_system_no_enso import global_functions
from earth_sys.earth_no_enso import EarthParams, earth_network

#for cluster computations
os.chdir("../")

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
temperature_trajs = np.loadtxt(r"D:\Dokumente\Uni\PhD application\Climate\figshare_overshoots_paper\temp_input\Tpeak_tconv_values\temp_input_values.txt", comments=['#']) #T_peak T_lim t_conv R mu_0 mu_1

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
input_file = np.loadtxt(r"D:\Dokumente\Uni\PhD application\Climate\figshare_overshoots_paper\start_ensemble\latin_prob.txt",
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
            
            lh_output = []
            
            for strength in coupling_strength:
                print("Coupling strength: {}".format(strength))
                output = []

                for sys_var in tqdm(input_file):
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

                    # get back the network of the Earth system
                    forcing = lambda t: forcing_function(T_0, mu_0, mu_1, T_lim, R)(t*conv_fac_gis)
                    net = earth_network(earth_params, forcing, strength, kk[0], kk[1], kk[2])
                    # initialize state
                    initial_state = [-1, -1, -1, -1, -1, -1] #initial state
                    ev = evolve(net, initial_state)
                    # plotter.network(net)

                    # How many points are to be calculated. odeint's precision is mostly independent of this, taking adaptive steps
                    n_steps = 1000

                    #t_end given in years; also possible to use equilibrate method
                    t_end = duration/conv_fac_gis #simulation length in "real" years
                    t_span = np.linspace(0, t_end, n_steps)
                    sol = odeint( net.f , initial_state, t_span, Dfun=net.jac )
                    ev._times = list(t_span*conv_fac_gis)
                    ev._states = list(sol)


                    #saving structure
                    output.append([ev.get_timeseries()[0],
                                ev.get_timeseries()[1][:, 0],
                                ev.get_timeseries()[1][:, 1],
                                ev.get_timeseries()[1][:, 2],
                                ev.get_timeseries()[1][:, 3],
                                ev.get_timeseries()[1][:, 4],
                                ev.get_timeseries()[1][:, 5],
                                [net.get_number_tipped(timeseries) for timeseries in ev.get_timeseries()[1]],
                                [[net.get_tip_states(timeseries)[0]].count(True) for timeseries in ev.get_timeseries()[1]],
                                [[net.get_tip_states(timeseries)[1]].count(True) for timeseries in ev.get_timeseries()[1]],
                                [[net.get_tip_states(timeseries)[2]].count(True) for timeseries in ev.get_timeseries()[1]],
                                [[net.get_tip_states(timeseries)[3]].count(True) for timeseries in ev.get_timeseries()[1]],
                                [[net.get_tip_states(timeseries)[4]].count(True) for timeseries in ev.get_timeseries()[1]],
                                [[net.get_tip_states(timeseries)[5]].count(True) for timeseries in ev.get_timeseries()[1]],
                    ])
                    
                data = np.array(output)
                ensemble_avg = np.mean(data, axis=0)
                lh_output.append(ensemble_avg)

                
            #necessary for break condition
            if len(lh_output):
                #saving structure
                data = np.array(lh_output)
                np.savetxt(f"{path}.txt", data[:,6])
                no_int = data[0]
                int_avg = np.mean(data[1:], axis=0)
                int_std = np.std(data[1:], ddof=1, axis=0)
                t_grid = int_avg[0] # arbitrary whether no_int or int_avg
                diff_gis = int_avg[1] - no_int[1]
                diff_thc = int_avg[2] - no_int[2]
                diff_wais = int_avg[3] - no_int[3]
                diff_amaz = int_avg[4] - no_int[4]
                diff_nino = int_avg[5] - no_int[5]
                diff_assi = int_avg[6] - no_int[6]
                int_n_tipped = int_avg[7]
                no_int_n_tipped = no_int[7]
                final_results[key] = int_n_tipped - no_int_n_tipped
                #plotting structure
                fig = plt.figure()
                plt.grid(True)
                # plt.title("Coupling strength: {}\n  Wais to Thc:{}  Amaz to Nino:{} Thc to Amaz:{} \n Tlim={}°C Tpeak={}°C tconv={}yr".format(
                #     np.round(strength, 2), kk[0], kk[1], kk[2], T_lim, T_peak_det, t_conv_det))
                # plt.plot(time, state_gis, label="GIS", color='c')
                # plt.plot(time, state_thc, label="THC", color='b')
                # plt.plot(time, state_wais, label="WAIS", color='k')
                # plt.plot(time, state_amaz, label="AMAZ", color='g')
                # plt.plot(time, state_nino, label="NINO", color='y')
                for i, data_strength in enumerate(data[1:]):
                    plt.plot(t_grid, data_strength[7], label=f"Interactions: 0.{i+1}")
                plt.plot(t_grid, no_int_n_tipped, label="No interactions", color='y')
                plt.title(f"Temperature properties: {key}")
                plt.xlabel("Time [yr]")
                plt.ylabel("Tipped elements")
                plt.legend(loc='best')  # , ncol=5)
                plt.tight_layout()
                plt.savefig("{}/{}_feedbacks/network_{}_{}_{}/feedbacks_Tlim{}_Tpeak{}_tconv{}.pdf".format(long_save_name, namefile, 
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