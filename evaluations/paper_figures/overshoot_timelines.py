import numpy as np
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
import re
import glob
import os

import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("white")
sns.despine()



#scenario values
Tcrit_gis = 1.1000044240427855
Tcrit_thc = 3.610264578635851
Tcrit_wais = 2.9911259752938916
Tcrit_amaz = 4.30407670121842




###########Overshoot scenarios
#global variables
t_vals = np.arange(0, 50001, 1) 

#Temperature input funciton
def T_input(t, T_0, mu_0, mu_1, T_lim, y):
	temp_input = T_0 + y*t - (1 - np.exp(-(mu_0+mu_1*t)*t))*(y*t - (T_lim - T_0))
	return temp_input

#proportionality factor (based on current warming rates)
def gamma(T_0, mu_0, T_lim, R):
	y = R + mu_0*(T_0-T_lim)
	return y



####################################MAIN####################################
#parameters
T_0 = 1.0
mu_0 = 0.0015 


#Data
data = np.loadtxt("timeline_data/temp_input_values.txt", comments=['#']) #T_peak T_lim t_conv R mu_0 mu_1

output = []
for i in data:
	print(i)
	T_peak_det = i[0]
	T_lim = i[1]
	t_conv_det = i[2]
	R = i[3]
	mu_0 = i[4]
	mu_1 = i[5]


	y = gamma(T_0, mu_0, T_lim, R) #y can also be fixed at 0.02
	
	for t in t_vals:
		temp_input = T_input(t, T_0, mu_0, mu_1, T_lim, y)
		output.append([t, mu_0, mu_1, T_0, T_lim, temp_input])

output = np.array(output)
output = np.array_split(output, 4)

#print(output)




##PREPARE COLORMAP
#more colormaps from Fabio Crameri (perceptually uniform)
cmap_data = np.loadtxt("../ScientificColourMaps4/lajolla/lajolla.txt")
cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)

cmaplist = [cmap(i) for i in range(cmap.N)]
colors = np.array([80, 120, 160, 200]) #np.linspace(65, 200, 4)
colors = np.array([int(x) for x in colors])
############



#Plotting
fig, ax0 = plt.subplots(1, 1, figsize=(16, 6))

ax0.plot(output[0].T[0], output[0].T[-1], color=cmaplist[colors[0]], label="Overshoot scenario $\\mathbf{b}$", lw=3)
ax0.plot(output[1].T[0], output[1].T[-1], color=cmaplist[colors[1]], label="Overshoot scenario $\\mathbf{c}$", lw=3)
ax0.plot(output[2].T[0], output[2].T[-1], color=cmaplist[colors[2]], label="Overshoot scenario $\\mathbf{d}$", lw=3)
ax0.plot(output[3].T[0], output[3].T[-1], color=cmaplist[colors[3]], label="Overshoot scenario $\\mathbf{e}$", lw=3)

ax0.axhline(y=Tcrit_gis, color="c", linestyle="-")
ax0.text(900, 0.83, "T$_{crit, GIS}$", color="c")
ax0.axhline(y=Tcrit_thc+0.07, color="b", linestyle="-")
ax0.text(890, 3.775, "T$_{crit, AMOC}$", color="b")
ax0.axhline(y=Tcrit_wais, color="#9760F1", linestyle="-")
ax0.text(675, 3.1, "T$_{crit, WAIS}$", color="#9760F1")
ax0.axhline(y=Tcrit_amaz, color="g", linestyle="-")
ax0.text(890, 4.4, "T$_{crit, AMAZ}$", color="g")


ax0.set_xlim([-10, 1000.0])
ax0.set_ylim([-0.1, 5.1])
ax0.set_xlabel("Model time [yr]")
ax0.set_ylabel("$\Delta$ GMT [°C]")

ax0.text(-55, 4.9, "$\\mathbf{a}$", fontsize=20)
ax0.legend(loc="center right", bbox_to_anchor=(0.999, 0.57))


fig.tight_layout()
fig.savefig("figs/timeline/overshoot_scenarios.png")
fig.savefig("figs/timeline/overshoot_scenarios.pdf")
#fig.show()
fig.clf()
plt.close()





#######Data for tipping experiments
files = np.sort(glob.glob("timeline_data/timeline*.txt"))
print(files)

#load each of the four files
tl_0 = np.loadtxt(files[0])
tl_1 = np.loadtxt(files[1])
tl_2 = np.loadtxt(files[2])
tl_3 = np.loadtxt(files[3])


##################################################################################################################
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 12)) #figsize is given in width, height

lw = 3
alpha_back = 0.75
color_gis = "c"
color_thc = "b"
color_wais = "#9760F1" #use other color than "k"
color_amaz = "g"



ax0.plot(tl_0.T[0], tl_0.T[1], color=color_gis, label="Greenland", lw=lw)
ax0.plot(tl_0.T[0], tl_0.T[3], color=color_wais, label="West\nAntarctica", lw=lw)
ax0.plot(tl_0.T[0], tl_0.T[2], color=color_thc, label="AMOC", lw=lw)
ax0.plot(tl_0.T[0], tl_0.T[4], color=color_amaz, label="Amazon\nrainforest", lw=lw)
ax0.set_xlabel("Model time (yr)")
ax0.set_ylabel("State of tipping elements")

ax0.set_ylim([-1.3, 1.3])
#ax0.set_xlim([-1000, 11000])
ax0.set_xlim([-250, 3000])
ax0.set_yticks(np.arange(-1., 1.5, 0.5))
#ax0.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
ax0.axhspan(1/np.sqrt(3), 1.5, hatch="\\\\", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax0.axhspan(-1.5, -1/np.sqrt(3), hatch="//", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax0.text(750, 0.9, "Transitioned regime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax0.text(900, -0.8, "Baseline regime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax0.text(800, -0.36, "Overshoot\nscenario:\nT$_{Peak}$=3.0 °C\nT$_{Conv}$=0.0 °C\nt$_{Conv}$=100 yrs", color=cmaplist[colors[0]], bbox=dict(facecolor="white", edgecolor=cmaplist[colors[0]], boxstyle='round,pad=1'))

ax0.legend(loc="best")#, bbox_to_anchor=(1., 1.0), fontsize=20)#, ncol=5, fontsize=35)
ax0.text(-600, 1.2, "$\\mathbf{b}$", fontsize=20)


ax1.plot(tl_1.T[0], tl_1.T[1], color=color_gis, label="Greenland", lw=lw)
ax1.plot(tl_1.T[0], tl_1.T[3], color=color_wais, label="West\nAntarctica", lw=lw)
ax1.plot(tl_1.T[0], tl_1.T[2], color=color_thc, label="AMOC", lw=lw)
ax1.plot(tl_1.T[0], tl_1.T[4], color=color_amaz, label="Amazon\nrainforest", lw=lw)
ax1.set_xlabel("Model time (yr)")
ax1.set_ylabel("State of tipping element")

ax1.set_ylim([-1.3, 1.3])
ax1.set_xlim([-250, 3000])
ax1.set_yticks(np.arange(-1., 1.5, 0.5))
#ax1.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
ax1.axhspan(1/np.sqrt(3), 1.5, hatch="\\\\", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax1.axhspan(-1.5, -1/np.sqrt(3), hatch="//", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax1.text(900, 0.75, "Transitioned regime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax1.text(1100, -0.8, "Baseline regime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax1.text(1965, -0.36, "Overshoot\nscenario:\nT$_{Peak}$=4.5 °C\nT$_{Conv}$=0.0 °C\nt$_{Conv}$=400 yrs", color=cmaplist[colors[1]], bbox=dict(facecolor="white", edgecolor=cmaplist[colors[1]], boxstyle='round,pad=1'))

#ax1.legend(loc="best")#, bbox_to_anchor=(1., 1.0), fontsize=20)#, ncol=5, fontsize=35)
ax1.text(-600, 1.2, "$\\mathbf{c}$", fontsize=20)



ax2.plot(tl_2.T[0], tl_2.T[1], color=color_gis, label="Greenland", lw=lw)
ax2.plot(tl_2.T[0], tl_2.T[3], color=color_wais, label="West\nAntarctica", lw=lw)
ax2.plot(tl_2.T[0], tl_2.T[2], color=color_thc, label="AMOC", lw=lw)
ax2.plot(tl_2.T[0], tl_2.T[4], color=color_amaz, label="Amazon\nrainforest", lw=lw)
ax2.set_xlabel("Model time (yr)")
ax2.set_ylabel("State of tipping element")

ax2.set_ylim([-1.3, 1.3])
ax2.set_xlim([-250, 3000])
ax2.set_yticks(np.arange(-1., 1.5, 0.5))
#ax2.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
ax2.axhspan(1/np.sqrt(3), 1.5, hatch="\\\\", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax2.axhspan(-1.5, -1/np.sqrt(3), hatch="//", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax2.text(50, 0.85, "Transitioned\nregime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax2.text(1000, -1.15, "Baseline regime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax2.text(1965, -0.36, "Overshoot\nscenario:\nT$_{Peak}$=4.0 °C\nT$_{Conv}$=1.5 °C\nt$_{Conv}$=200 yrs", color=cmaplist[colors[2]], bbox=dict(facecolor="white", edgecolor=cmaplist[colors[2]], boxstyle='round,pad=1'))

#ax2.legend(loc="best")#, bbox_to_anchor=(1., 1.0), fontsize=20)#, ncol=5, fontsize=35)
ax2.text(-600, 1.2, "$\\mathbf{d}$", fontsize=20)



ax3.plot(tl_3.T[0], tl_3.T[1], color=color_gis, label="Greenland", lw=lw)
ax3.plot(tl_3.T[0], tl_3.T[3], color=color_wais, label="West\nAntarctica", lw=lw)
ax3.plot(tl_3.T[0], tl_3.T[2], color=color_thc, label="AMOC", lw=lw)
ax3.plot(tl_3.T[0], tl_3.T[4], color=color_amaz, label="Amazon\nrainforest", lw=lw)
ax3.set_xlabel("Model time (yr)")
ax3.set_ylabel("State of tipping element")

ax3.set_ylim([-1.3, 1.3])
ax3.set_xlim([-250, 3000])
ax3.set_yticks(np.arange(-1., 1.5, 0.5))
#ax3.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
ax3.axhspan(1/np.sqrt(3), 1.5, hatch="\\\\", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax3.axhspan(-1.5, -1/np.sqrt(3), hatch="//", edgecolor='gray', facecolor="none", alpha=alpha_back, lw=2)
ax3.text(1300, 0.8, "Transitioned regime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax3.text(1000, -1.0, "Baseline regime", color="k", bbox=dict(facecolor="white", boxstyle='round,pad=0.1'))#, fontsize=20, fontweight='bold')
ax3.text(1965, -0.36, "Overshoot\nscenario:\nT$_{Peak}$=4.0 °C\nT$_{Conv}$=2.0 °C\nt$_{Conv}$=700 yrs", color=cmaplist[colors[3]], bbox=dict(facecolor="white", edgecolor=cmaplist[colors[3]], boxstyle='round,pad=1'))

#ax3.legend(loc="best")#, bbox_to_anchor=(1., 1.0), fontsize=20)#, ncol=5, fontsize=35)
ax3.text(-600, 1.2, "$\\mathbf{e}$", fontsize=20)



fig.tight_layout()
fig.savefig("figs/timeline/timeline.png")
fig.savefig("figs/timeline/timeline.pdf")
fig.clf()
plt.close()


print("Finish")