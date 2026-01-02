import numpy as np
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set(font_scale=1.5)
import re
import glob
import os


def data_plotter(jj, kk): #jj=overshoot_data_index; kk=tipping_risk_index
    density = 50 #density=number of samples on the x-axis
    min_num_sam = 5 # minimum acceptable number of samples in interval

    plot_data = []
    plot_vals = np.linspace(np.amin(mean.T[jj].flatten()), np.amax(mean.T[jj].flatten()), density+1)
    for i in range(1, len(plot_vals)):
        plot_arg = np.argwhere((mean.T[jj].flatten() >= plot_vals[i-1]) & (mean.T[jj].flatten() <= plot_vals[i]))
        if len(plot_arg) >= min_num_sam:
            plot_aim_y = mean.T[kk].flatten()[plot_arg]
            plot_data.append([plot_vals[i-1]+0.5*(plot_vals[i]-plot_vals[i-1]), np.mean(plot_aim_y), np.std(plot_aim_y)]) #pretty larger errors - if standard error is used, this problem is solved: divide np.std(plot_aim_y)/np.sqrt(len(plot_arg))
    
    return np.array(plot_data)



def data_plotter_all(all_data, tip_data): #data=overshoot data; tip_data=tipping_data
    density = 50 #density=number of samples on the x-axis
    min_num_sam = 5 # minimum acceptable number of samples in interval

    plot_data = []
    plot_vals = np.linspace(np.amin(all_data), np.amax(all_data), density+1)
    for i in range(1, len(plot_vals)):
        plot_arg = np.argwhere((all_data >= plot_vals[i-1]) & (all_data <= plot_vals[i]))
        if len(plot_arg) >= min_num_sam:
            plot_aim_y = tip_data[plot_arg]
            plot_data.append([plot_vals[i-1]+0.5*(plot_vals[i]-plot_vals[i-1]), np.mean(plot_aim_y), np.std(plot_aim_y)]) 

    return np.array(plot_data)



# read all files
networks = np.sort(glob.glob("../latin/*0_0.0_*"))
print(networks)


#find all possible values to loop over them
data = np.loadtxt("{}/data_preparator_overshoot.txt".format(networks[0]))
##rounding errors in T_peak must be handled appropriately
data.T[1] = np.where(data.T[1]==1.9, 2.0, data.T[1])
data.T[1] = np.where(data.T[1]==2.9, 3.0, data.T[1])
data.T[1] = np.where(data.T[1]==3.9, 4.0, data.T[1])
data.T[1] = np.where(data.T[1]==4.9, 5.0, data.T[1])
data.T[1] = np.where(data.T[1]==5.9, 6.0, data.T[1])
##


Tlim_vals = np.unique(data.T[0])
Tpeak_vals = np.unique(data.T[1])
tconv_vals = np.unique(data.T[2])
coupling_vals = np.unique(data.T[3])


for Tpeak in Tpeak_vals:
    print(Tpeak)
    for Tlim in Tlim_vals:
        print(Tlim)
        output = []
        for network in networks:
            print(network)
            data = np.loadtxt("{}/data_preparator_overshoot.txt".format(network))
            """
            new tip info - entries are defined as follows:
            0 --> Tlim
            1 --> Tpeak
            2 --> tconv
            3 --> coupling
            4 --> total nr. of tipping elements
            5 --> final state gis
            6 --> final state thc
            7 --> final state wais
            8 --> final state amaz
            9-11 --> GIS: Overshoot strength, Overshoot time, Max Overshoot 
            12-14 --> THC: Overshoot strength, Overshoot time, Max Overshoot 
            15-17 --> WAIS: Overshoot strength, Overshoot time, Max Overshoot 
            18-20 --> AMAZ: Overshoot strength, Overshoot time, Max Overshoot 
            """

            ##rounding errors in T_peak must be handled appropriately
            data.T[1] = np.where(data.T[1]==1.9, 2.0, data.T[1])
            data.T[1] = np.where(data.T[1]==2.9, 3.0, data.T[1])
            data.T[1] = np.where(data.T[1]==3.9, 4.0, data.T[1])
            data.T[1] = np.where(data.T[1]==4.9, 5.0, data.T[1])
            data.T[1] = np.where(data.T[1]==5.9, 6.0, data.T[1])
            ##
            data_sampled = np.argwhere((data.T[0]==Tlim) & (data.T[1]==Tpeak))
            data_sampled = data[data_sampled]

            for i in data_sampled:
                output.append([j for j in i[0]])

        output = np.array(output)
        np.savetxt("extra_evaluation_data/overshoot_extra_Tpeak{}_Tlim{}.txt".format(Tpeak, Tlim), output)






#global variales
t_max = 1000.0 #maximal overshoot time

files = np.sort(glob.glob("extra_evaluation_data/overshoot_extra_Tpeak*_Tlim*.txt"))
print(files)


output = []
for file in files:
    print(file)
    data = np.loadtxt(file)

    #scenario
    Tlim = data.T[0][0]
    Tpeak = data.T[1][0]

    """
    new tip info - entries are defined as follows:
    0 --> Tlim
    1 --> Tpeak
    2 --> tconv
    3 --> coupling
    4 --> total nr. of tipping elements
    5 --> final state gis
    6 --> final state thc
    7 --> final state wais
    8 --> final state amaz
    9-11 --> GIS: Overshoot strength, Overshoot time, Max Overshoot 
    12-14 --> THC: Overshoot strength, Overshoot time, Max Overshoot 
    15-17 --> WAIS: Overshoot strength, Overshoot time, Max Overshoot 
    18-20 --> AMAZ: Overshoot strength, Overshoot time, Max Overshoot 
    """

    #all cascades are sampled
    data_tipped = np.argwhere((data.T[4]>0.0)).flatten()

    nr_sample = len(data)
    nr_tipped = len(data_tipped) #number of tipped samples

    #go through all tipping events/cascades and decide whether that cascade is triggered by a basline scenario, an overshoot scenario, or an interaction scenario
    baseline = 0
    overshoot = 0
    interaction = 0
    for casc in data_tipped:
        cascade = data[casc]
        if cascade[10]>t_max or cascade[13]>t_max or cascade[16]>t_max or cascade[19]>t_max: # if any critical temperature is higher than the final temperature ==> baseline cascade
            baseline += 1
        else: #if every critical temperature is below the final temperature
            if cascade[9]>0.0 or cascade[12]>0.0 or cascade[15]>0.0 or cascade[18]>0.0: # there is a temporal overshoot in some tipping element ==> overshoot cascade
                overshoot += 1
            else: #no overshoot is reached ==> the critical temperature is not surpassed in any time step. Therefore this is a pure interaction cascade [should not happen too often]
                interaction += 1

    output.append([Tlim, Tpeak, nr_sample, nr_tipped, baseline, overshoot, interaction])

output = np.array(output)
np.savetxt("extra_evaluation_data/baseline_overshoot_interaction_cascades.txt", output)


print("Finish")