import numpy as np
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import glob
import os

import seaborn as sns
sns.set(font_scale=2.0)
sns.set_style("ticks")


#sns.set_style("ticks")
#sns.despine()



def tipping_difference(name, Tlim_vals, coupling, tconv_vals):
    if name == "empty":
        const_load = np.loadtxt("../../evaluations_no_enso/latin/all/dynamic_classes_mean.txt")
    else:
        const_load = np.loadtxt("../../evaluations_no_enso/latin/all/dynamic_classes_mean_{}.txt".format(name))

    gmt_const = const_load.T[0]
    cpl_const = const_load.T[1]
    all_mean_const = const_load.T[2]
    all_std_const = const_load.T[3]

    #find indices of correct coupling
    Tlim_const = Tlim_vals
    all_mean_const_array = []
    for i in Tlim_const:
        idx_cpl = np.where((cpl_const==coupling) & (gmt_const==i))[0]
        for j in range(0, len(tconv_vals)):
            all_mean_const_array.append(all_mean_const[idx_cpl][0])
    all_mean_const_array = np.array(all_mean_const_array)
    return all_mean_const_array




####MAIN
files = np.sort(glob.glob("../latin/all/dynamic_classes_mean_cpl*_Tpeak???.txt"))

data = []
for file in files:
    for line in np.loadtxt(file):
        data.append(line)
data = np.array(data)

Tlim_vals = np.unique(data.T[0])
Tpeak_vals = np.unique(data.T[1])
tconv_vals = np.unique(data.T[2])
coupling_vals = np.unique(data.T[3])

for Tlim in Tlim_vals:
    print(Tlim)
    #Plotting and Saving procedure
    output_all = np.loadtxt("figs/dynamic_classes_mean_Tlim{}.txt".format(Tlim))

    Tlim_plot = output_all.T[0]
    tconv_plot = output_all.T[2]
    cpl_plot = output_all.T[3]
    #all tipping elements
    tipped_mean = output_all.T[4]
    tipped_std = output_all.T[5]

    #all single tipping elements
    gis_mean = output_all.T[6]
    gis_std = output_all.T[7]

    thc_mean = output_all.T[8]
    thc_std = output_all.T[9]

    wais_mean = output_all.T[10]
    wais_std = output_all.T[11]

    amaz_mean = output_all.T[12]
    amaz_std = output_all.T[13]
    ############


    output_agg = []
    for cpl_agg in coupling_vals:
        data_agg = np.argwhere((cpl_plot>cpl_agg-0.001) & (cpl_plot<cpl_agg+0.001))
        output = []
        for i in data_agg:
            output.append([tipped_mean[i[0]], tipped_std[i[0]]])
        output = np.array(output)
        #for first value - define baseline
        if len(output_agg) == 0:
            baseline = np.mean(output.T[0])
        output_agg.append([
            cpl_agg, 
            np.mean(output.T[0]), 
            np.std(output.T[0])/(np.sqrt(9)),       #divided by np.sqrt(number of networks) 
            np.mean(output.T[0])/baseline-1, 
            (1/(len(output.T[1])))*np.sqrt(np.sum(np.square(output.T[1])))
        ]) #last entry is error progatation (very similar to np.std(output.T[0]))
    output_agg_eq = np.array(output_agg)



    #Plotting and Saving procedure
    output_all = np.loadtxt("../../evaluations_1000/paper_figures/figs/dynamic_classes_mean_Tlim{}.txt".format(Tlim))

    Tlim_plot = output_all.T[0]
    tconv_plot = output_all.T[2]
    cpl_plot = output_all.T[3]
    #all tipping elements
    tipped_mean = output_all.T[4]
    tipped_std = output_all.T[5]

    #all single tipping elements
    gis_mean = output_all.T[6]
    gis_std = output_all.T[7]

    thc_mean = output_all.T[8]
    thc_std = output_all.T[9]

    wais_mean = output_all.T[10]
    wais_std = output_all.T[11]

    amaz_mean = output_all.T[12]
    amaz_std = output_all.T[13]
    ############


    output_agg = []
    for cpl_agg in coupling_vals:
        data_agg = np.argwhere((cpl_plot>cpl_agg-0.001) & (cpl_plot<cpl_agg+0.001))
        output = []
        for i in data_agg:
            output.append([tipped_mean[i[0]], tipped_std[i[0]]])
        output = np.array(output)
        #for first value - define baseline
        if len(output_agg) == 0:
            baseline = np.mean(output.T[0])
        output_agg.append([
            cpl_agg, 
            np.mean(output.T[0]), 
            np.std(output.T[0])/(np.sqrt(9)),       #divided by np.sqrt(number of networks) 
            np.mean(output.T[0])/baseline-1, 
            (1/(len(output.T[1])))*np.sqrt(np.sum(np.square(output.T[1])))
        ]) #last entry is error progatation (very similar to np.std(output.T[0]))
    output_agg_1000yrs = np.array(output_agg)




    #Plotting and Saving procedure
    output_all = np.loadtxt("../../evaluations_100/paper_figures/figs/dynamic_classes_mean_Tlim{}.txt".format(Tlim))

    Tlim_plot = output_all.T[0]
    tconv_plot = output_all.T[2]
    cpl_plot = output_all.T[3]
    #all tipping elements
    tipped_mean = output_all.T[4]
    tipped_std = output_all.T[5]

    #all single tipping elements
    gis_mean = output_all.T[6]
    gis_std = output_all.T[7]

    thc_mean = output_all.T[8]
    thc_std = output_all.T[9]

    wais_mean = output_all.T[10]
    wais_std = output_all.T[11]

    amaz_mean = output_all.T[12]
    amaz_std = output_all.T[13]
    ############


    output_agg = []
    for cpl_agg in coupling_vals:
        data_agg = np.argwhere((cpl_plot>cpl_agg-0.001) & (cpl_plot<cpl_agg+0.001))
        output = []
        for i in data_agg:
            output.append([tipped_mean[i[0]], tipped_std[i[0]]])
        output = np.array(output)
        #for first value - define baseline
        if len(output_agg) == 0:
            baseline = np.mean(output.T[0])
        output_agg.append([
            cpl_agg, 
            np.mean(output.T[0]), 
            np.std(output.T[0])/(np.sqrt(9)),       #divided by np.sqrt(number of networks) 
            np.mean(output.T[0])/baseline-1, 
            (1/(len(output.T[1])))*np.sqrt(np.sum(np.square(output.T[1])))
        ]) #last entry is error progatation (very similar to np.std(output.T[0]))
    output_agg_100yrs = np.array(output_agg)


    #####
    output_agg_eq = output_agg_eq
    output_agg_1000yrs = output_agg_1000yrs
    output_agg_100yrs = output_agg_100yrs

    #print(output_agg_100yrs.T[1])
    print(output_agg_1000yrs.T[1])
    print(output_agg_eq.T[1])
    print(output_agg_eq.T[2])
    #####

    diff_1000_100 = np.subtract(output_agg_1000yrs.T[1], output_agg_100yrs.T[1])
    diff_1000_100[diff_1000_100<0] = 0.0
    diff_eq_1000 = np.subtract(output_agg_eq.T[1], output_agg_1000yrs.T[1])
    diff_eq_1000[diff_eq_1000<0] = 0.0

    
    # plot mean figure
    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

    
    ax0.barh(np.arange(0.0, 1.1, 0.1), output_agg_100yrs.T[1], xerr=output_agg_100yrs.T[2], color=["#993351"],
            height=0.08, align="center", alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))
    
    ax0.barh(np.arange(0.0, 1.1, 0.1), diff_1000_100, xerr=output_agg_1000yrs.T[2], left=output_agg_100yrs.T[1], color=["#E94849"],
            height=0.08, align="center", alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))


    ax0.barh(np.arange(0.0, 1.1, 0.1), diff_eq_1000, xerr=output_agg_eq.T[2], left=output_agg_1000yrs.T[1], color=["#FDA463"], 
            height=0.08, align="center", alpha=0.85, ecolor="k", error_kw=dict(ecolor='k', lw=2, capsize=4, capthick=4, alpha=1.0))


    ax0.set_xlabel("Number of tipped elements")
    ax0.set_ylabel("Interaction strength [a.u.]")
    ax0.set_xlim(0.0, 3.0)
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax0.grid(axis="y", color="gray")

    #sns.despine(bottom=False, left=False) #no right and upper border lines
    #fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    fig.savefig("figs/compare_dynamic_classes_mean_Tlim{}.png".format(Tlim))
    fig.savefig("figs/compare_dynamic_classes_mean_Tlim{}.pdf".format(Tlim))
    # fig.show()
    fig.clf()
    plt.close()



print("Finish")


