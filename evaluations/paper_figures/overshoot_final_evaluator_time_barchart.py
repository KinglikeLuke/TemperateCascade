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

for i in range(0, len(data)):
    Tlim = data[i, 0]
    Tpeak = data[i, 1]
    print(Tpeak)
    print(Tlim)


    #compute compartments of pie chart
    perc_frac_100yrs = data_100yrs.T[3, i]/data_100yrs.T[2, i]
    perc_frac_1000yrs = data_1000yrs.T[3, i]/data_1000yrs.T[2, i]
    perc_frac = data.T[3, i]/data.T[2, i]

    perc_mean_100yrs = perc_frac_100yrs
    perc_mean_1000yrs = perc_frac_1000yrs - perc_frac_100yrs
    perc_mean = perc_frac - perc_frac_1000yrs

    sizes = [perc_mean_100yrs, perc_mean_1000yrs, perc_mean]
    for j in range(0, len(sizes)):
        if sizes[j] < 0:
            sizes[j] = 0
        elif sizes[j] > 1.0:
            sizes[j] = 1.0

    #sizes = sizes/sum(sizes)
    print(sizes)
    output.append(sizes)



    ###
    hundred = sizes[0]
    thousand = sizes[1]
    eq = sizes[2]
    not_tipped = 1.0 - sizes[0] - sizes[1] - sizes[2]


    """
    #BARCHART
    ####
    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

    #red-ish colors (use with alpha=0.85)
    colors = ["#800026", "#E31A1C", "#FD8D3C"]

    ax0.bar(np.arange(1), 100*hundred,
        color=["#800026"], width=0.25, align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=3, alpha=1.0), edgecolor=["#800026"], lw=2)
    ax0.bar(np.arange(1), 100*thousand, bottom=100*hundred,
        color=["#E31A1C"], width=0.25, align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=3, alpha=1.0), edgecolor=["#E31A1C"], lw=2) 
    ax0.bar(np.arange(1), 100*eq, bottom=100*thousand+100*hundred,
        color=["#FD8D3C"], width=0.25, align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=3, alpha=1.0), edgecolor=["#FD8D3C"], lw=2) 
    ax0.bar(np.arange(1), 100*not_tipped, bottom=100*thousand+100*hundred+100*eq,
        color="None", width=0.25, align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=3, alpha=1.0), hatch='//', edgecolor=["#36454F"], lw=2) 


    ax0.set_ylabel("Tipping risk [%]")
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_xticks([])
    ax0.set_ylim(0, 102)
    ax0.set_yticks(np.arange(0.0, 110, 10))

    plt.text(-0.25, 20, "{}%".format(int(np.round(100*hundred, 0))), rotation=90, color="#800026")
    plt.text(-0.25, 40, "{}%".format(int(np.round(100*thousand, 0))), rotation=90, color="#E31A1C")
    plt.text(-0.25, 60, "{}%".format(int(np.round(100*eq, 0))), rotation=90, color="#FD8D3C")
    plt.text(-0.25, 80, "{}%".format(int(np.round(100*not_tipped, 0))), rotation=90, color="#36454F")

    #sns.despine(bottom=False, left=False) #no right and upper border lines
    #fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    fig.savefig("figs/pie_charts/time/barchart_Tlim{}_Tpeak{}.png".format(Tlim, Tpeak))
    fig.savefig("figs/pie_charts/time/barchart_Tlim{}_Tpeak{}.pdf".format(Tlim, Tpeak))
    fig.clf()
    plt.close()
    """


    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    #red-ish colors (use with alpha=0.85)
    colors = ["#800026", "#E31A1C", "#FD8D3C"]
    alpha = 0.85
    explode = (0, 0)
    sizes = [hundred, thousand, eq]
    sizes = sizes/sum(sizes)

    ax0.set_title("Tipping risk: {}%".format(np.round(100*(hundred+thousand+eq)/(hundred+thousand+eq+not_tipped), 1)))
    wedges, plt_labels, autopct = ax0.pie(sizes, colors=colors, startangle=0, wedgeprops={'alpha':alpha}, autopct="\n\n%.0f%%", radius=(hundred+thousand+eq)/(hundred+thousand+eq+not_tipped))
    for w in wedges:
        w.set_linewidth(2)
        w.set_edgecolor('k')


    circle = plt.Circle((0, 0), 1.0, color="#36454F", fill=False, linewidth=3.0, alpha=0.8)
    ax0.add_patch(circle)
    ax0.text(-0.25, 0.5, "100% tipping", color="#36454F", alpha=1.0)

    fig.tight_layout()
    fig.savefig("figs/pie_charts/time/barchart_Tlim{}_Tpeak{}.png".format(Tlim, Tpeak))
    fig.savefig("figs/pie_charts/time/barchart_Tlim{}_Tpeak{}.pdf".format(Tlim, Tpeak))
    fig.clf()
    plt.close()
    


output = np.array(output)
print(output)
np.savetxt("figs/pie_charts/time/barchart.txt", output)

print("Finish")