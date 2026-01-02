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


##PREPARE COLORMAP
#more colormaps from Fabio Crameri (perceptually uniform)
cmap_data = np.loadtxt("../ScientificColourMaps4/lajolla/lajolla.txt")
cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)

cmaplist = [cmap(i) for i in range(cmap.N)]
colors = np.linspace(75, 255, 6)
colors = np.array([int(x) for x in colors])
############

rng_lower = 25
rng_upper = 75


# read all files
networks = np.sort(glob.glob("../latin/*0_0.0_*"))
print(networks)



#find all possible values to loop over them
Tpeak_vals = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])


#Do all averging for all four investigated variables
output_Tpeak = []

for Tpeak in Tpeak_vals:
    print(Tpeak)
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

    zero_tipped_mean = np.mean(output.T[0]) 
    zero_tipped_std = np.std(output.T[0])

    #zero_tipped_mean = np.median(output.T[0])
    #zero_tipped_std = iqr(output.T[0], rng=(rng_lower, rng_upper))

    one_tipped_mean = np.mean(output.T[1])
    one_tipped_std = np.std(output.T[1])

    two_tipped_mean = np.mean(output.T[2])
    two_tipped_std = np.std(output.T[2])

    three_tipped_mean = np.mean(output.T[3])
    three_tipped_std = np.std(output.T[3])

    four_tipped_mean = np.mean(output.T[4])
    four_tipped_std = np.std(output.T[4])


    gis_tipped_mean = np.mean(output.T[5])
    gis_tipped_std = np.std(output.T[5])

    thc_tipped_mean = np.mean(output.T[6])
    thc_tipped_std = np.std(output.T[6])

    wais_tipped_mean = np.mean(output.T[7])
    wais_tipped_std = np.std(output.T[7])

    amaz_tipped_mean = np.mean(output.T[8])
    amaz_tipped_std = np.std(output.T[8])


    #plotting
    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))

    #ax0.grid(True)
    ax0.bar(np.arange(5), 100*np.array([zero_tipped_mean, one_tipped_mean, two_tipped_mean, three_tipped_mean, four_tipped_mean]), 
        yerr=100*np.array([zero_tipped_std, one_tipped_std, two_tipped_std, three_tipped_std, four_tipped_std]),
        color=[cmaplist[i] for i in colors], 
        align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    

    """
    parts = ax0.violinplot([100*output.T[0], 100*output.T[1], 100*output.T[2], 100*output.T[3], 100*output.T[4]], 
        np.arange(5), showextrema=False, showmeans=True, showmedians=False)
    

    #for pc in parts['bodies']:
    #    pc.set_facecolor([cmaplist[i] for i in colors])
    #    pc.set_alpha(1)
    """

    ax0.set_xticks(np.arange(5))
    ax0.set_xticklabels(["0", "1", "2", "3", "4"])
    ax0.set_xlabel("Elements tipped")
    ax0.set_ylabel("Tipping risk [%]")
    ax0.set_ylim([0, 103])
    ax0.set_yticks([0, 20, 40, 60, 80, 100])

    #ax1.grid(True)
    ax1.bar(np.arange(5), 100*np.array([gis_tipped_mean, thc_tipped_mean, wais_tipped_mean, amaz_tipped_mean, 0]), 
        yerr=100*np.array([gis_tipped_std, thc_tipped_std, wais_tipped_std, amaz_tipped_std, 0]),
        color=["c", "b", "#9760F1", "g", "r"], 
        align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels(["GIS", "THC", "WAIS", "AMAZ", "D"])
    ax1.set_xlabel("Tipped")
    ax1.set_ylabel("Tipping risk [%]")
    ax1.set_ylim([0, 103])
    ax1.set_yticks([0, 20, 40, 60, 80, 100])

    #sns.despine(bottom=True, left=True) #no right and upper border lines
    fig.tight_layout()
    fig.savefig("figs/overshoot_hist_long_Tpeak_{}.png".format(Tpeak))
    fig.savefig("figs/overshoot_hist_long_Tpeak_{}.pdf".format(Tpeak))
    # fig.show()
    fig.clf()
    plt.close()

    #########################END##########################
    print(zero_tipped_mean, one_tipped_mean, two_tipped_mean, three_tipped_mean, four_tipped_mean)
    print(zero_tipped_std, one_tipped_std, two_tipped_std, three_tipped_std, four_tipped_std)

    output_Tpeak.append([
        gis_tipped_mean, gis_tipped_std, 
        thc_tipped_mean, thc_tipped_std, 
        wais_tipped_mean, wais_tipped_std, 
        amaz_tipped_mean, amaz_tipped_std
        ])



output_Tpeak = np.array(output_Tpeak)

gis_mean = np.take(output_Tpeak.T[0], [0, 2, 4, 6, 8]) #only taking 2, 3, 4, 5, 6°C
gis_std = np.take(output_Tpeak.T[1], [0, 2, 4, 6, 8])

thc_mean = np.take(output_Tpeak.T[2], [0, 2, 4, 6, 8])
thc_std = np.take(output_Tpeak.T[3], [0, 2, 4, 6, 8])

wais_mean = np.take(output_Tpeak.T[4], [0, 2, 4, 6, 8])
wais_std = np.take(output_Tpeak.T[5], [0, 2, 4, 6, 8])

amaz_mean = np.take(output_Tpeak.T[6], [0, 2, 4, 6, 8])
amaz_std = np.take(output_Tpeak.T[7], [0, 2, 4, 6, 8])


#Plotting
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))

ax0.bar(np.arange(5), 100*gis_mean, yerr=100*gis_std, color=["c", "c", "c", "c", "c"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax0.set_xticks(np.arange(5))
ax0.set_xticklabels(["2°C", "3°C", "4°C", "5°C", "6°C"])
ax0.set_xlabel("Peak temperature")
ax0.set_ylabel("Tipping risk [%]")
ax0.set_ylim([0, 103])
ax0.set_yticks([0, 20, 40, 60, 80, 100])

ax1.bar(np.arange(5), 100*thc_mean, yerr=100*thc_std, color=["b", "b", "b", "b", "b"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax1.set_xticks(np.arange(5))
ax1.set_xticklabels(["2°C", "3°C", "4°C", "5°C", "6°C"])
ax1.set_xlabel("Peak temperature")
ax1.set_ylabel("Tipping risk [%]")
ax1.set_ylim([0, 103])
ax1.set_yticks([0, 20, 40, 60, 80, 100])

ax2.bar(np.arange(5), 100*wais_mean, yerr=100*wais_std, color=["#9760F1", "#9760F1", "#9760F1", "#9760F1", "#9760F1"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax2.set_xticks(np.arange(5))
ax2.set_xticklabels(["2°C", "3°C", "4°C", "5°C", "6°C"])
ax2.set_xlabel("Peak temperature")
ax2.set_ylabel("Tipping risk [%]")
ax2.set_ylim([0, 103])
ax2.set_yticks([0, 20, 40, 60, 80, 100])

ax3.bar(np.arange(5), 100*amaz_mean, yerr=100*amaz_std, color=["g", "g", "g", "g", "g"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax3.set_xticks(np.arange(5))
ax3.set_xticklabels(["2°C", "3°C", "4°C", "5°C", "6°C"])
ax3.set_xlabel("Peak temperature")
ax3.set_ylabel("Tipping risk [%]")
ax3.set_ylim([0, 103])
ax3.set_yticks([0, 20, 40, 60, 80, 100])


#sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("figs/overshoot_hist_long_singleTEs.png")
fig.savefig("figs/overshoot_hist_long_singleTEs.pdf")
# fig.show()
fig.clf()
plt.close()



##############################################
gis_mean = np.take(output_Tpeak.T[0], [0, 1, 2, 3, 4]) #only taking 2.0, 2.5, 3.0, 3.5, 4.0°C
gis_std = np.take(output_Tpeak.T[1], [0, 1, 2, 3, 4])

thc_mean = np.take(output_Tpeak.T[2], [0, 1, 2, 3, 4])
thc_std = np.take(output_Tpeak.T[3], [0, 1, 2, 3, 4])

wais_mean = np.take(output_Tpeak.T[4], [0, 1, 2, 3, 4])
wais_std = np.take(output_Tpeak.T[5], [0, 1, 2, 3, 4])

amaz_mean = np.take(output_Tpeak.T[6], [0, 1, 2, 3, 4])
amaz_std = np.take(output_Tpeak.T[7], [0, 1, 2, 3, 4])


print(gis_mean)
print(gis_std)

print(thc_mean )
print(thc_std )

print(wais_mean)
print(wais_std )

print(amaz_mean)
print(amaz_std )


#Plotting
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))


ax0.bar(np.arange(5), 100*gis_mean, yerr=100*gis_std, color=["c", "c", "c", "c", "c"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax0.set_xticks(np.arange(5))
ax0.set_xticklabels(["2.0°C", "2.5°C", "3.0°C", "3.5°C", "4.0°C"])
ax0.set_xlabel("Peak temperature")
ax0.set_ylabel("Tipping risk [%]")
ax0.set_ylim([0, 103])
ax0.set_yticks([0, 20, 40, 60, 80, 100])

ax1.bar(np.arange(5), 100*thc_mean, yerr=100*thc_std, color=["b", "b", "b", "b", "b"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax1.set_xticks(np.arange(5))
ax1.set_xticklabels(["2.0°C", "2.5°C", "3.0°C", "3.5°C", "4.0°C"])
ax1.set_xlabel("Peak temperature")
ax1.set_ylabel("Tipping risk [%]")
ax1.set_ylim([0, 103])
ax1.set_yticks([0, 20, 40, 60, 80, 100])

ax2.bar(np.arange(5), 100*wais_mean, yerr=100*wais_std, color=["#9760F1", "#9760F1", "#9760F1", "#9760F1", "#9760F1"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax2.set_xticks(np.arange(5))
ax2.set_xticklabels(["2.0°C", "2.5°C", "3.0°C", "3.5°C", "4.0°C"])
ax2.set_xlabel("Peak temperature")
ax2.set_ylabel("Tipping risk [%]")
ax2.set_ylim([0, 103])
ax2.set_yticks([0, 20, 40, 60, 80, 100])

ax3.bar(np.arange(5), 100*amaz_mean, yerr=100*amaz_std, color=["g", "g", "g", "g", "g"], align='center', alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
ax3.set_xticks(np.arange(5))
ax3.set_xticklabels(["2.0°C", "2.5°C", "3.0°C", "3.5°C", "4.0°C"])
ax3.set_xlabel("Peak temperature")
ax3.set_ylabel("Tipping risk [%]")
ax3.set_ylim([0, 103])
ax3.set_yticks([0, 20, 40, 60, 80, 100])


#sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("figs/overshoot_hist_long_singleTEs_lowPeak.png")
fig.savefig("figs/overshoot_hist_long_singleTEs_lowPeak.pdf")
# fig.show()
fig.clf()
plt.close()




print("Finish")