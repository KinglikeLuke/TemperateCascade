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
sns.set_style("white")
sns.despine()


data = np.loadtxt("extra_evaluation_data/baseline_overshoot_interaction_cascades.txt")
data_100yrs = np.loadtxt("../../evaluations_100/paper_figures/extra_evaluation_data/baseline_overshoot_interaction_cascades.txt")
data_1000yrs = np.loadtxt("../../evaluations_1000/paper_figures/extra_evaluation_data/baseline_overshoot_interaction_cascades.txt")

Tpeak_vals = np.unique(data.T[1])
"""
Tlim = line[0]
Tpeak = line[1]
nr_sample = line[2]
nr_tipped = line[3]
baseline = line[4]
overshoot = line[5]
interaction = line[6]
"""


output = []

for Tpeak in Tpeak_vals:
    print(Tpeak)
    data_Tpeak_args = np.argwhere((data.T[1] == Tpeak)).flatten()

    #peak temperature
    data_Tpeak_100yrs = data_100yrs[data_Tpeak_args]
    data_Tpeak_1000yrs = data_1000yrs[data_Tpeak_args]
    data_Tpeak = data[data_Tpeak_args]
    

    #compute compartments of pie chart
    perc_frac_100yrs = data_Tpeak_100yrs.T[3]/data_Tpeak_100yrs.T[2]
    perc_frac_1000yrs = data_Tpeak_1000yrs.T[3]/data_Tpeak_1000yrs.T[2]
    perc_frac = data_Tpeak.T[3]/data_Tpeak.T[2]

    perc_mean_100yrs = perc_frac_100yrs
    perc_mean_1000yrs = perc_frac_1000yrs - perc_frac_100yrs
    perc_mean = perc_frac - perc_frac_1000yrs

    perc_av_100yrs = np.mean(perc_mean_100yrs)
    perc_av_1000yrs = np.mean(perc_mean_1000yrs)
    perc_av = np.mean(perc_mean)
    sizes = [perc_av_100yrs, perc_av_1000yrs, perc_av]
    sizes = [perc_av_100yrs, perc_av_1000yrs, perc_av]/sum(sizes)
    output.append(sizes)
    ###


    #fraction of tipped
    data_Tpeak_100yrs = np.mean(data_Tpeak_100yrs, axis=0)
    data_Tpeak_1000yrs = np.mean(data_Tpeak_1000yrs, axis=0)
    data_Tpeak = np.mean(data_Tpeak, axis=0)
    
    nr_tipped_100yrs = data_Tpeak_100yrs[3]
    nr_tipped_1000yrs = data_Tpeak_1000yrs[3]
    nr_tipped = data_Tpeak[3]
    nr_sample = data_Tpeak[2]
    ###


    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    #red-ish colors (use with alpha=0.6)
    colors = ["#800026", "#E31A1C", "#FD8D3C"]
    alpha = 0.8
    labels = ["\n100 years", "\n1000 years", "\nEquilibrium"]
    explode = (0, 0)
    

    ax0.set_title("{}% {}% {}%".format(np.round(100*nr_tipped_100yrs/nr_tipped, 1), np.round(100*nr_tipped_1000yrs/nr_tipped, 1), np.round(100*nr_tipped/nr_sample, 1)))

    wedges, plt_labels, autopct = ax0.pie(sizes, colors=colors, counterclock=False, startangle=90, wedgeprops={'alpha':alpha}, autopct="\n\n%.0f%%", radius=nr_tipped/nr_sample)
    for w in wedges:
        w.set_linewidth(2)
        w.set_edgecolor('k')
    #necessary since overlapping with border of piechart
    #autopct[-1].set_position((-0.5,  1.05))
    #autopct[-1].set_position((1.3,  0.))
    #autopct[-1].set_position((1.3,  0.))
    #ax1.axis('equal')
    circle = plt.Circle((0, 0), 1.0, color="#36454F", fill=False, linewidth=3.0, alpha=0.8)
    ax0.add_patch(circle)
    ax0.text(-0.25, 0.0, "100% tipping", color="#36454F", alpha=1.0)


    #ax1.text(-2, 0.75, "$\\mathbf{b}$")

    fig.tight_layout()
    fig.savefig("figs/pie_charts/time/Tpeak{}.png".format(Tpeak))
    fig.savefig("figs/pie_charts/time/Tpeak{}.pdf".format(Tpeak))
    fig.clf()
    plt.close()


print("Finish")