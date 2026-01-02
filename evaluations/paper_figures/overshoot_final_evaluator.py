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


for line in data:
    print(line)
    Tlim = line[0]
    Tpeak = line[1]
    nr_sample = line[2]
    nr_tipped = line[3]
    baseline = line[4]
    overshoot = line[5]
    interaction = line[6]


    """
    #PIE CHARTS
    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    #red-ish colors (use with alpha=0.6)
    colors = ["#B03A2E", "#2874A6"]
    alpha = 0.8
    labels = ["\nOvershoot", "\nBaseline"]
    explode = (0, 0)
    sizes = [overshoot, baseline]


    ax0.set_title("{}% {}% {}%".format(np.round(100*overshoot/(baseline+overshoot), 0), np.round(100*baseline/(baseline+overshoot), 0), np.round(100*nr_tipped/nr_sample, 1)))
    wedges, plt_labels, autopct = ax0.pie(sizes, colors=colors, startangle=0, wedgeprops={'alpha':alpha}, autopct="\n\n%.0f%%", radius=nr_tipped/nr_sample)
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
    fig.savefig("figs/pie_charts/Tlim{}_Tpeak{}.png".format(Tlim, Tpeak))
    fig.savefig("figs/pie_charts/Tlim{}_Tpeak{}.pdf".format(Tlim, Tpeak))
    fig.clf()
    plt.close()
    """



    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

    #red-ish colors (use with alpha=0.85)
    colors = ["#B03A2E", "#2874A6"]
    ax0.set_title("{}% {}% {}%".format(np.round(100*overshoot/(baseline+overshoot), 0), np.round(100*baseline/(baseline+overshoot), 0), np.round(100*nr_tipped/nr_sample, 1)))

    ax0.bar(np.arange(1), 100*overshoot/(baseline+overshoot),
        color=["#B03A2E"], width=0.25, align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=3, alpha=1.0), edgecolor=["#B03A2E"], lw=2)
    ax0.bar(np.arange(1), 100*baseline/(baseline+overshoot), bottom=100*overshoot/(baseline+overshoot),
        color=["#2874A6"], width=0.25, align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=3, alpha=1.0), edgecolor=["#2874A6"], lw=2) 

    ax0.set_ylabel("Tipping risk [%]")
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_xticks([])
    ax0.set_ylim(0, 102)
    ax0.set_yticks(np.arange(0.0, 110, 10))

    plt.text(-0.25, 20, "{}%".format(int(np.round(100*overshoot/(baseline+overshoot), 0))), rotation=90, color="#B03A2E")
    plt.text(-0.25, 40, "{}%".format(int(np.round(100*baseline/(baseline+overshoot), 0))), rotation=90, color="#2874A6")

    #sns.despine(bottom=False, left=False) #no right and upper border lines
    #fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    fig.savefig("figs/pie_charts/Tlim{}_Tpeak{}.png".format(Tlim, Tpeak))
    fig.savefig("figs/pie_charts/Tlim{}_Tpeak{}.pdf".format(Tlim, Tpeak))
    fig.clf()
    plt.close()
    



print("Finish")