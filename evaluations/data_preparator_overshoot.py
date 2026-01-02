import numpy as np
import sys
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set(font_scale=1.)
import re
import glob
import os


#needs to be mounted for cluster run
os.chdir(os.getcwd() + "evaluations")


sys_var = np.array(sys.argv[2:], dtype=str)
num = int(sys_var[0])
print(sys_var)

# read all files
networks = [np.array(np.sort(glob.glob("../results/no_feedbacks/*0_0.0_*")))[num]]
print(networks)

for network in networks:
    print(network)
    net_splitter = re.split("network", network)[-1] #used for the saving structure


    #open and read file
    output = []

    # read all files
    folders = np.array(np.sort(glob.glob(network + "/0*")))
    for folder in folders:
        print(folder)
        subfolders = np.array(np.sort(glob.glob(folder + "/feedbacks_*.txt")))
        emp_vals = np.loadtxt(folder + "/empirical_values.txt")

        for kk in subfolders:
            #find coupling
            print(kk)
            resplit = re.split("Tlim|Tpeak|tconv|_|.txt", kk)

            Tlim = float(resplit[-7])/10.0
            Tpeak = float(resplit[-5])/10.0
            tconv = np.round(float(resplit[-3]),-2)
            coupling = float(resplit[-2])


            #compute overshooting strength
            ov_data = np.loadtxt("../temp_input/timeseries_final/Tlim{}_Tpeak{}_tconv{}.txt".format(int(10*Tlim), int(10*Tpeak), int(tconv))).T[-1]


            #GIS
            gis_ov = np.subtract(ov_data, emp_vals[0])
            gis_ov[gis_ov<0.0] = 0.0
            gis_time_ov = np.argwhere((gis_ov>0.0))
            if len(gis_time_ov) != 0:
                gis_time_ov = np.amax(gis_time_ov) - np.amin(gis_time_ov) #Time of Overshoot in years
            else:
                gis_time_ov = 0.0
            gis_max_ov = np.amax(gis_ov) #Maximum Overshoot in °C
            gis_ov = np.sum(gis_ov) #Overshoot strength in yr*°C

            #THC
            thc_ov = np.subtract(ov_data, emp_vals[1])
            thc_ov[thc_ov<0.0] = 0.0
            thc_time_ov = np.argwhere((thc_ov>0.0))
            if len(thc_time_ov) != 0:
                thc_time_ov = np.amax(thc_time_ov) - np.amin(thc_time_ov) #Time of Overshoot in years
            else:
                thc_time_ov = 0.0
            thc_max_ov = np.amax(thc_ov) #Maximum Overshoot in °C
            thc_ov = np.sum(thc_ov) #Overshoot strength in yr*°C

            #WAIS
            wais_ov = np.subtract(ov_data, emp_vals[2])
            wais_ov[wais_ov<0.0] = 0.0
            wais_time_ov = np.argwhere((wais_ov>0.0))
            if len(wais_time_ov) != 0:
                wais_time_ov = np.amax(wais_time_ov) - np.amin(wais_time_ov) #Time of Overshoot in years
            else:
                wais_time_ov = 0.0
            wais_max_ov = np.amax(wais_ov) #Maximum Overshoot in °C
            wais_ov = np.sum(wais_ov) #Overshoot strength in yr*°C

            #AMAZ
            amaz_ov = np.subtract(ov_data, emp_vals[3])
            amaz_ov[amaz_ov<0.0] = 0.0
            amaz_time_ov = np.argwhere((amaz_ov>0.0))
            if len(amaz_time_ov) != 0:
                amaz_time_ov = np.amax(amaz_time_ov) - np.amin(amaz_time_ov) #Time of Overshoot in years
            else:
                amaz_time_ov = 0.0
            amaz_max_ov = np.amax(amaz_ov) #Maximum Overshoot in °C
            amaz_ov = np.sum(amaz_ov) #Overshoot strength in yr*°C

            #############
            tip_info = np.loadtxt(kk)[5:] 
            tip_info = np.insert(tip_info, 0, np.array([Tlim, Tpeak, tconv, coupling]))
            tip_info = np.concatenate((tip_info, np.array([gis_ov, gis_time_ov, gis_max_ov, thc_ov, thc_time_ov, thc_max_ov, wais_ov, wais_time_ov, wais_max_ov, amaz_ov, amaz_time_ov, amaz_max_ov])))
            output.append([float(i) for i in tip_info])

    data = np.array(output)

    #Saving and plotting structure
    np.savetxt("latin/network{}/data_preparator_overshoot.txt".format(net_splitter), data)
