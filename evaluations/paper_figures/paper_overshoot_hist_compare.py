import numpy as np
from scipy.stats import iqr
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


rng_lower = 25
rng_upper = 75


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
print(networks)


print("t_conv")
#find all possible values to loop over them
tconv_vals = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

for tconv in tconv_vals:
    print(tconv)
    #################### Equilibrium ####################
    #print("Equilibrium")
    output = []
    for network in networks:
        print(network)
        file = np.loadtxt("{}/overshoot_hist_tconv_{}.0.txt".format(network, tconv))
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

        zero_tipped  = np.count_nonzero(file.T[4]==0.)/len(file.T[4])
        one_tipped   = np.count_nonzero(file.T[4]==1.)/len(file.T[4])
        two_tipped   = np.count_nonzero(file.T[4]==2.)/len(file.T[4])
        three_tipped = np.count_nonzero(file.T[4]==3.)/len(file.T[4])
        four_tipped  = np.count_nonzero(file.T[4]==4.)/len(file.T[4])

        gis_tipped  = np.count_nonzero(file.T[5]==1.)/len(file.T[5])
        thc_tipped  = np.count_nonzero(file.T[6]==1.)/len(file.T[6])
        wais_tipped = np.count_nonzero(file.T[7]==1.)/len(file.T[7])
        amaz_tipped = np.count_nonzero(file.T[8]==1.)/len(file.T[8])


        output.append([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped, gis_tipped, thc_tipped, wais_tipped, amaz_tipped])
    output = np.array(output)

    zero_tipped_mean_eq = np.mean(output.T[0]) 
    zero_tipped_std_eq = np.std(output.T[0])

    #zero_tipped_mean_eq = np.median(output.T[0])
    #zero_tipped_std_eq = iqr(output.T[0], rng=(rng_lower, rng_upper))

    one_tipped_mean_eq = np.mean(output.T[1])
    one_tipped_std_eq = np.std(output.T[1])

    two_tipped_mean_eq = np.mean(output.T[2])
    two_tipped_std_eq = np.std(output.T[2])

    three_tipped_mean_eq = np.mean(output.T[3])
    three_tipped_std_eq = np.std(output.T[3])

    four_tipped_mean_eq = np.mean(output.T[4])
    four_tipped_std_eq = np.std(output.T[4])


    gis_tipped_mean_eq = np.mean(output.T[5])
    gis_tipped_std_eq = np.std(output.T[5])

    thc_tipped_mean_eq = np.mean(output.T[6])
    thc_tipped_std_eq = np.std(output.T[6])

    wais_tipped_mean_eq = np.mean(output.T[7])
    wais_tipped_std_eq = np.std(output.T[7])

    amaz_tipped_mean_eq = np.mean(output.T[8])
    amaz_tipped_std_eq = np.std(output.T[8])



    ###############################################
    #Plotting
    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    #ax0.grid(True)
    ax0.bar(np.arange(5), 100*np.array([zero_tipped_mean_eq, one_tipped_mean_eq, two_tipped_mean_eq, three_tipped_mean_eq, four_tipped_mean_eq]),
        yerr=100*np.array([zero_tipped_std_eq, one_tipped_std_eq, two_tipped_std_eq, three_tipped_std_eq, four_tipped_std_eq]),
        color=[cmaplist[i] for i in colors], 
        align='center', alpha=0.85, 
        ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax0.set_xticks(np.arange(5))
    ax0.set_xticklabels(["0", "1", "2", "3", "4"])
    ax0.set_xlabel("Elements tipped")
    ax0.set_ylabel("Tipping risk [%]")
    ax0.set_ylim([0, 103])
    ax0.set_yticks([0, 20, 40, 60, 80, 100])


    #ax1.grid(True)
    ax1.bar(np.arange(5), 100*np.array([gis_tipped_mean_eq, thc_tipped_mean_eq, wais_tipped_mean_eq, amaz_tipped_mean_eq, 0]),
        yerr=100*np.array([gis_tipped_std_eq, thc_tipped_std_eq, wais_tipped_std_eq, amaz_tipped_std_eq, 0]),
        color=["c", "b", "#9760F1", "g", "r"], 
        align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels(["GIS", "AMOC", "WAIS", "AMAZ", "D"])
    ax1.set_xlabel("Tipped")
    ax1.set_ylabel("Tipping risk [%]")
    ax1.set_ylim([0, 103])
    ax1.set_yticks([0, 20, 40, 60, 80, 100])

    #sns.despine(bottom=True, left=True) #no right and upper border lines
    fig.tight_layout()
    fig.savefig("figs/overshoot_hist_long_tconv_{}.png".format(tconv))
    fig.savefig("figs/overshoot_hist_long_tconv_{}.pdf".format(tconv))
    # fig.show()
    fig.clf()
    plt.close()
    #########################END##########################



#find all possible values to loop over them
print("T_peak")
Tpeak_vals = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

for Tpeak in Tpeak_vals:
    print(Tpeak)
    #################### Equilibrium ####################
    print("Equilibrium")
    output = []
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

        zero_tipped  = np.count_nonzero(file.T[4]==0.)/len(file.T[4])
        one_tipped   = np.count_nonzero(file.T[4]==1.)/len(file.T[4])
        two_tipped   = np.count_nonzero(file.T[4]==2.)/len(file.T[4])
        three_tipped = np.count_nonzero(file.T[4]==3.)/len(file.T[4])
        four_tipped  = np.count_nonzero(file.T[4]==4.)/len(file.T[4])

        gis_tipped  = np.count_nonzero(file.T[5]==1.)/len(file.T[5])
        thc_tipped  = np.count_nonzero(file.T[6]==1.)/len(file.T[6])
        wais_tipped = np.count_nonzero(file.T[7]==1.)/len(file.T[7])
        amaz_tipped = np.count_nonzero(file.T[8]==1.)/len(file.T[8])


        output.append([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped, gis_tipped, thc_tipped, wais_tipped, amaz_tipped])
    output = np.array(output)


    zero_tipped_mean_eq = np.mean(output.T[0])
    zero_tipped_std_eq = np.std(output.T[0])

    one_tipped_mean_eq = np.mean(output.T[1])
    one_tipped_std_eq = np.std(output.T[1])

    two_tipped_mean_eq = np.mean(output.T[2])
    two_tipped_std_eq = np.std(output.T[2])

    three_tipped_mean_eq = np.mean(output.T[3])
    three_tipped_std_eq = np.std(output.T[3])

    four_tipped_mean_eq = np.mean(output.T[4])
    four_tipped_std_eq = np.std(output.T[4])


    gis_tipped_mean_eq = np.mean(output.T[5])
    gis_tipped_std_eq = np.std(output.T[5])

    thc_tipped_mean_eq = np.mean(output.T[6])
    thc_tipped_std_eq = np.std(output.T[6])

    wais_tipped_mean_eq = np.mean(output.T[7])
    wais_tipped_std_eq = np.std(output.T[7])

    amaz_tipped_mean_eq = np.mean(output.T[8])
    amaz_tipped_std_eq = np.std(output.T[8])




    #################### 1000years ####################
    output = []
    print("1000 years")
    for network in networks:
        print(network)
        file = np.loadtxt("../../evaluations_1000/paper_figures/{}/overshoot_histlong_Tpeak_{}.txt".format(network, Tpeak))
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

        zero_tipped  = np.count_nonzero(file.T[4]==0.)/len(file.T[4])
        one_tipped   = np.count_nonzero(file.T[4]==1.)/len(file.T[4])
        two_tipped   = np.count_nonzero(file.T[4]==2.)/len(file.T[4])
        three_tipped = np.count_nonzero(file.T[4]==3.)/len(file.T[4])
        four_tipped  = np.count_nonzero(file.T[4]==4.)/len(file.T[4])

        gis_tipped  = np.count_nonzero(file.T[5]==1.)/len(file.T[5])
        thc_tipped  = np.count_nonzero(file.T[6]==1.)/len(file.T[6])
        wais_tipped = np.count_nonzero(file.T[7]==1.)/len(file.T[7])
        amaz_tipped = np.count_nonzero(file.T[8]==1.)/len(file.T[8])


        output.append([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped, gis_tipped, thc_tipped, wais_tipped, amaz_tipped])
    output = np.array(output)


    zero_tipped_mean_1000 = np.mean(output.T[0])
    zero_tipped_std_1000 = np.std(output.T[0])

    one_tipped_mean_1000 = np.mean(output.T[1])
    one_tipped_std_1000 = np.std(output.T[1])

    two_tipped_mean_1000 = np.mean(output.T[2])
    two_tipped_std_1000 = np.std(output.T[2])

    three_tipped_mean_1000 = np.mean(output.T[3])
    three_tipped_std_1000 = np.std(output.T[3])

    four_tipped_mean_1000 = np.mean(output.T[4])
    four_tipped_std_1000 = np.std(output.T[4])


    gis_tipped_mean_1000 = np.mean(output.T[5])
    gis_tipped_std_1000 = np.std(output.T[5])

    thc_tipped_mean_1000 = np.mean(output.T[6])
    thc_tipped_std_1000 = np.std(output.T[6])

    wais_tipped_mean_1000 = np.mean(output.T[7])
    wais_tipped_std_1000 = np.std(output.T[7])

    amaz_tipped_mean_1000 = np.mean(output.T[8])
    amaz_tipped_std_1000 = np.std(output.T[8])




    #################### 100years ####################
    output = []
    print("100 years")
    for network in networks:
        print(network)
        file = np.loadtxt("../../evaluations_100/paper_figures/{}/overshoot_histlong_Tpeak_{}.txt".format(network, Tpeak))
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

        zero_tipped  = np.count_nonzero(file.T[4]==0.)/len(file.T[4])
        one_tipped   = np.count_nonzero(file.T[4]==1.)/len(file.T[4])
        two_tipped   = np.count_nonzero(file.T[4]==2.)/len(file.T[4])
        three_tipped = np.count_nonzero(file.T[4]==3.)/len(file.T[4])
        four_tipped  = np.count_nonzero(file.T[4]==4.)/len(file.T[4])

        gis_tipped  = np.count_nonzero(file.T[5]==1.)/len(file.T[5])
        thc_tipped  = np.count_nonzero(file.T[6]==1.)/len(file.T[6])
        wais_tipped = np.count_nonzero(file.T[7]==1.)/len(file.T[7])
        amaz_tipped = np.count_nonzero(file.T[8]==1.)/len(file.T[8])


        output.append([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped, gis_tipped, thc_tipped, wais_tipped, amaz_tipped])
    output = np.array(output)


    zero_tipped_mean_100 = np.mean(output.T[0])
    zero_tipped_std_100 = np.std(output.T[0])

    one_tipped_mean_100 = np.mean(output.T[1])
    one_tipped_std_100 = np.std(output.T[1])

    two_tipped_mean_100 = np.mean(output.T[2])
    two_tipped_std_100 = np.std(output.T[2])

    three_tipped_mean_100 = np.mean(output.T[3])
    three_tipped_std_100 = np.std(output.T[3])

    four_tipped_mean_100 = np.mean(output.T[4])
    four_tipped_std_100 = np.std(output.T[4])


    gis_tipped_mean_100 = np.mean(output.T[5])
    gis_tipped_std_100 = np.std(output.T[5])

    thc_tipped_mean_100 = np.mean(output.T[6])
    thc_tipped_std_100 = np.std(output.T[6])

    wais_tipped_mean_100 = np.mean(output.T[7])
    wais_tipped_std_100 = np.std(output.T[7])

    amaz_tipped_mean_100 = np.mean(output.T[8])
    amaz_tipped_std_100 = np.std(output.T[8])




    ###############################################
    #Plotting
    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    #ax0.grid(True)
    width = 0.25
    ax0.bar(np.arange(5) - width, 100*np.array([zero_tipped_mean_eq, one_tipped_mean_eq, two_tipped_mean_eq, three_tipped_mean_eq, four_tipped_mean_eq]), width,
        yerr=100*np.array([zero_tipped_std_eq, one_tipped_std_eq, two_tipped_std_eq, three_tipped_std_eq, four_tipped_std_eq]),
        color=[cmaplist[i] for i in colors], 
        align='center', alpha=0.85, 
        ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax0.bar(np.arange(5), 100*np.array([zero_tipped_mean_1000, one_tipped_mean_1000, two_tipped_mean_1000, three_tipped_mean_1000, four_tipped_mean_1000]), width,
        yerr=100*np.array([zero_tipped_std_1000, one_tipped_std_1000, two_tipped_std_1000, three_tipped_std_1000, four_tipped_std_1000]),
        color=[cmaplist[i] for i in colors], 
        align='center', alpha=0.85, 
        ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax0.bar(np.arange(5) + width, 100*np.array([zero_tipped_mean_100, one_tipped_mean_100, two_tipped_mean_100, three_tipped_mean_100, four_tipped_mean_100]), width,
        yerr=100*np.array([zero_tipped_std_100, one_tipped_std_100, two_tipped_std_100, three_tipped_std_100, four_tipped_std_100]),
        color=[cmaplist[i] for i in colors], 
        align='center', alpha=0.85, 
        ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax0.set_xticks(np.arange(5))
    ax0.set_xticklabels(["0", "1", "2", "3", "4"])
    ax0.set_xlabel("Elements tipped")
    ax0.set_ylabel("Tipping risk [%]")
    ax0.set_ylim([0, 103])
    ax0.set_yticks([0, 20, 40, 60, 80, 100])


    #ax1.grid(True)
    ax1.bar(np.arange(5) - width, 100*np.array([gis_tipped_mean_eq, thc_tipped_mean_eq, wais_tipped_mean_eq, amaz_tipped_mean_eq, 0]), width,
        yerr=100*np.array([gis_tipped_std_eq, thc_tipped_std_eq, wais_tipped_std_eq, amaz_tipped_std_eq, 0]),
        color=["c", "b", "#9760F1", "g", "r"], 
        align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    print(100*np.array([gis_tipped_mean_eq, thc_tipped_mean_eq, wais_tipped_mean_eq, amaz_tipped_mean_eq, 0]))
    print(100*np.array([gis_tipped_std_eq, thc_tipped_std_eq, wais_tipped_std_eq, amaz_tipped_std_eq, 0]))
    ax1.bar(np.arange(5), 100*np.array([gis_tipped_mean_1000, thc_tipped_mean_1000, wais_tipped_mean_1000, amaz_tipped_mean_1000, 0]), width,
        yerr=100*np.array([gis_tipped_std_1000, thc_tipped_std_1000, wais_tipped_std_1000, amaz_tipped_std_1000, 0]),
        color=["c", "b", "#9760F1", "g", "r"], 
        align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax1.bar(np.arange(5) + width, 100*np.array([gis_tipped_mean_100, thc_tipped_mean_100, wais_tipped_mean_100, amaz_tipped_mean_100, 0]), width,
        yerr=100*np.array([gis_tipped_std_100, thc_tipped_std_100, wais_tipped_std_100, amaz_tipped_std_100, 0]),
        color=["c", "b", "#9760F1", "g", "r"], 
        align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels(["GIS", "AMOC", "WAIS", "AMAZ", "D"])
    ax1.set_xlabel("Tipped")
    ax1.set_ylabel("Tipping risk [%]")
    ax1.set_ylim([0, 103])
    ax1.set_yticks([0, 20, 40, 60, 80, 100])

    #sns.despine(bottom=True, left=True) #no right and upper border lines
    fig.tight_layout()
    fig.savefig("figs/overshoot_hist_long_Tpeak_eq_1000_100_{}.png".format(Tpeak))
    fig.savefig("figs/overshoot_hist_long_Tpeak_eq_1000_100_{}.pdf".format(Tpeak))
    # fig.show()
    fig.clf()
    plt.close()
    #########################END##########################
    
print("Finish")