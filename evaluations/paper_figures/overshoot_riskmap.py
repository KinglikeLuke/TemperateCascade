import numpy as np
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
import re
import glob
import os


import seaborn as sns
sns.set(font_scale=2.0)
sns.set_style("ticks")

#sns.set_style("ticks")
##sns.despine()


##PREPARE COLORMAP
#more colormaps from Fabio Crameri (perceptually uniform)
cmap_data = np.loadtxt("../ScientificColourMaps4/lajolla/lajolla.txt")
cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)

cmaplist = [cmap(i) for i in range(cmap.N)]
colors = np.linspace(75, 255, 6)
colors = np.array([int(x) for x in colors])
############


# read all files
networks = np.sort(glob.glob("../latin/*0_0.0_*"))
#print(networks)


#find all possible values to loop over them
Tpeak_vals = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

#Do all averging for all four investigated variables
output_Tpeak = []
###separated into different Tlim values
output_Tpeak00 = []
output_Tpeak05 = []
output_Tpeak10 = []
output_Tpeak15 = []
output_Tpeak20 = []



#####
for Tpeak in Tpeak_vals:
    print(Tpeak)
    output = []

    #separated into T_lim_values
    output_Tlim00 = []
    output_Tlim05 = []
    output_Tlim10 = []
    output_Tlim15 = []
    output_Tlim20 = []


    for network in networks:
        print(network)
        file = np.loadtxt("{}/overshoot_histlong_Tpeak_{}.txt".format(network, Tpeak))

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

        for i in range(0, len(file)):
            output.append([j for j in file[i]])

            if file[i][0] == 0.0:
                output_Tlim00.append([j for j in file[i]])
            elif file[i][0] == 0.5:
                output_Tlim05.append([j for j in file[i]])
            elif file[i][0] == 1.0:
                output_Tlim10.append([j for j in file[i]])
            elif file[i][0] == 1.5:
                output_Tlim15.append([j for j in file[i]])
            elif file[i][0] == 2.0:
                output_Tlim20.append([j for j in file[i]])
            else:
                die
    
    #complete output    
    output = np.array(output)
    tconv_vals = np.unique(output.T[2])


    #output for specific Tlim values
    output_Tlim00 = np.array(output_Tlim00)
    tconv_vals00 = np.unique(output_Tlim00.T[2])

    output_Tlim05 = np.array(output_Tlim05)
    tconv_vals05 = np.unique(output_Tlim05.T[2])

    output_Tlim10 = np.array(output_Tlim10)
    tconv_vals10 = np.unique(output_Tlim10.T[2])

    output_Tlim15 = np.array(output_Tlim15)
    tconv_vals15 = np.unique(output_Tlim15.T[2])

    output_Tlim20 = np.array(output_Tlim20)
    tconv_vals20 = np.unique(output_Tlim20.T[2])



    #sort for Tpeak, tconv and nr_tipped elements
    for tconv in tconv_vals:
        tconv_args = np.argwhere((output.T[2]==tconv))    #Convergence time args
        nr_tipped = np.mean(output.T[4][tconv_args])      #Number of tipped tipping elements
        nr_tipped_std = np.std(output.T[4][tconv_args])   #Std. of tipped elements
        tip_risk = 1.0 - np.count_nonzero(output.T[4][tconv_args]==0.)/len(output.T[4][tconv_args]) #Risk that at least 1 TE is tipped

        output_Tpeak.append([Tpeak, tconv, nr_tipped, nr_tipped_std, tip_risk])



    #sort for Tpeak, tconv and nr_tipped elements
    for tconv in tconv_vals00:
        tconv_args = np.argwhere((output_Tlim00.T[2]==tconv))    #Convergence time args
        nr_tipped = np.mean(output_Tlim00.T[4][tconv_args])      #Number of tipped tipping elements
        nr_tipped_std = np.std(output_Tlim00.T[4][tconv_args])   #Std. of tipped elements
        tip_risk = 1.0 - np.count_nonzero(output_Tlim00.T[4][tconv_args]==0.)/len(output_Tlim00.T[4][tconv_args]) #Risk that at least 1 TE is tipped

        output_Tpeak00.append([Tpeak, tconv, nr_tipped, nr_tipped_std, tip_risk])



    #sort for Tpeak, tconv and nr_tipped elements
    for tconv in tconv_vals05:
        tconv_args = np.argwhere((output_Tlim05.T[2]==tconv))    #Convergence time args
        nr_tipped = np.mean(output_Tlim05.T[4][tconv_args])      #Number of tipped tipping elements
        nr_tipped_std = np.std(output_Tlim05.T[4][tconv_args])   #Std. of tipped elements
        tip_risk = 1.0 - np.count_nonzero(output_Tlim05.T[4][tconv_args]==0.)/len(output_Tlim05.T[4][tconv_args]) #Risk that at least 1 TE is tipped

        output_Tpeak05.append([Tpeak, tconv, nr_tipped, nr_tipped_std, tip_risk])



    #sort for Tpeak, tconv and nr_tipped elements
    for tconv in tconv_vals10:
        tconv_args = np.argwhere((output_Tlim10.T[2]==tconv))    #Convergence time args
        nr_tipped = np.mean(output_Tlim10.T[4][tconv_args])      #Number of tipped tipping elements
        nr_tipped_std = np.std(output_Tlim10.T[4][tconv_args])   #Std. of tipped elements
        tip_risk = 1.0 - np.count_nonzero(output_Tlim10.T[4][tconv_args]==0.)/len(output_Tlim10.T[4][tconv_args]) #Risk that at least 1 TE is tipped

        output_Tpeak10.append([Tpeak, tconv, nr_tipped, nr_tipped_std, tip_risk])


    #sort for Tpeak, tconv and nr_tipped elements
    for tconv in tconv_vals15:
        tconv_args = np.argwhere((output_Tlim15.T[2]==tconv))    #Convergence time args
        nr_tipped = np.mean(output_Tlim15.T[4][tconv_args])      #Number of tipped tipping elements
        nr_tipped_std = np.std(output_Tlim15.T[4][tconv_args])   #Std. of tipped elements
        tip_risk = 1.0 - np.count_nonzero(output_Tlim15.T[4][tconv_args]==0.)/len(output_Tlim15.T[4][tconv_args]) #Risk that at least 1 TE is tipped

        output_Tpeak15.append([Tpeak, tconv, nr_tipped, nr_tipped_std, tip_risk])



    #sort for Tpeak, tconv and nr_tipped elements
    for tconv in tconv_vals20:
        tconv_args = np.argwhere((output_Tlim20.T[2]==tconv))    #Convergence time args
        nr_tipped = np.mean(output_Tlim20.T[4][tconv_args])      #Number of tipped tipping elements
        nr_tipped_std = np.std(output_Tlim20.T[4][tconv_args])   #Std. of tipped elements
        tip_risk = 1.0 - np.count_nonzero(output_Tlim20.T[4][tconv_args]==0.)/len(output_Tlim20.T[4][tconv_args]) #Risk that at least 1 TE is tipped

        output_Tpeak20.append([Tpeak, tconv, nr_tipped, nr_tipped_std, tip_risk])


output_Tpeak = np.array(output_Tpeak)
np.savetxt("figs/riskmap/riskmap.txt", output_Tpeak)



output_Tpeak00 = np.array(output_Tpeak00)
np.savetxt("figs/riskmap/riskmap_Tlim00.txt", output_Tpeak00)

output_Tpeak05 = np.array(output_Tpeak05)
np.savetxt("figs/riskmap/riskmap_Tlim05.txt", output_Tpeak05)

output_Tpeak10 = np.array(output_Tpeak10)
np.savetxt("figs/riskmap/riskmap_Tlim10.txt", output_Tpeak10)

output_Tpeak15 = np.array(output_Tpeak15)
np.savetxt("figs/riskmap/riskmap_Tlim15.txt", output_Tpeak15)

output_Tpeak20 = np.array(output_Tpeak20)
np.savetxt("figs/riskmap/riskmap_Tlim20.txt", output_Tpeak20)


print("Finish")