import numpy as np
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import re
import glob


import seaborn as sns
sns.set(font_scale=1.75)
sns.set_style("ticks")

#sns.set_style("ticks")
##sns.despine()


##PREPARE COLORMAP
#more colormaps from Fabio Crameri (perceptually uniform)
cmap_data = np.loadtxt("../ScientificColourMaps4/lajolla/lajolla.txt")
cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)


#plotting
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))

########################################################################################################################
#Number of tipped elements

########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim10.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax0.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.0, 1.5])#, levels=[1.0, 1.5, 2.0])
tri = ax0.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[0.5])
sc = ax0.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax0, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax0.set_xlabel("Peak temperature [°C]")
ax0.set_ylabel("Convergence time [yr]")
ax0.set_xlim([2.0, 4.0])
ax0.set_xticks(np.arange(2.0, 4.5, 0.5))
ax0.set_yticks(np.arange(100, 1100, 100))
min_val = np.amin(nr_tipped)
ax0.text(2.0, 125, "#$_{tipped, min}$= %.2f" % (min_val))
ax0.text(1.5, 975, "$\\mathbf{a}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim15.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax1.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.5, 1.75, 2.0])
sc = ax1.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax1.set_xlabel("Peak temperature [°C]")
ax1.set_ylabel("Convergence time [yr]")
ax1.set_xlim([2.0, 4.0])
ax1.set_xticks(np.arange(2.0, 4.5, 0.5))
ax1.set_yticks(np.arange(100, 1100, 100))
min_val = np.amin(nr_tipped)
ax1.text(2.0, 125, "#$_{tipped, min}$= %.2f" % (min_val))
ax1.text(1.5, 975, "$\\mathbf{b}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim20.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax2.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[2.0, 2.25])#, levels=[1.0, 1.5, 2.0])
sc = ax2.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax2, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax2.set_xlabel("Peak temperature [°C]")
ax2.set_ylabel("Convergence time [yr]")
ax2.set_xlim([2.0, 4.0])
ax2.set_xticks(np.arange(2.0, 4.5, 0.5))
ax2.set_yticks(np.arange(100, 1100, 100))
min_val = np.amin(nr_tipped)
ax2.text(2.0, 125, "#$_{tipped, min}$= %.2f" % (min_val), color="white")
ax2.text(1.5, 975, "$\\mathbf{c}$")




########################################################################################################################################################
#RISKMAPS

########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim10.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax3.tricontour(tri_refi, tip_risk_refi, levels=[50, 66, 75], linewidths=2.0, colors="white")
tri = ax3.tricontour(tri_refi, tip_risk_refi, levels=[33], linewidths=2.0, colors="#943126")
sc = ax3.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax3, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax3.set_xlabel("Peak temperature [°C]")
ax3.set_ylabel("Convergence time [yr]")
ax3.set_xticks(np.arange(2.0, 4.5, 0.5))
ax3.set_xlim([2.0, 4.0])
ax3.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(np.amin(tip_risk), 2)
ax3.text(2.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
ax3.text(2.0, 500, "Risk", color="#943126")
ax3.text(1.5, 975, "$\\mathbf{d}$")



########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim15.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax4.tricontour(tri_refi, tip_risk_refi, levels=[66, 75, 85], linewidths=2.0, colors="white")
sc = ax4.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax4, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax4.set_xlabel("Peak temperature [°C]")
ax4.set_ylabel("Convergence time [yr]")
ax4.set_xticks(np.arange(2.0, 4.5, 0.5))
ax4.set_xlim([2.0, 4.0])
ax4.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(np.amin(tip_risk), 2)
ax4.text(2.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"), color="white")
ax4.text(1.5, 975, "$\\mathbf{e}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim20.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax5.tricontour(tri_refi, tip_risk_refi, levels=[85, 92], linewidths=2.0, colors="white")
sc = ax5.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax5, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax5.set_xlabel("Peak temperature [°C]")
ax5.set_ylabel("Convergence time [yr]")
ax5.set_xticks(np.arange(2.0, 4.5, 0.5))
ax5.set_xlim([2.0, 4.0])
ax5.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(np.amin(tip_risk), 2)
ax5.text(2.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"), color="white")
ax5.text(1.5, 975, "$\\mathbf{f}$")



#sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("figs/riskmap/Ntipped_riskmap_main.png")
fig.savefig("figs/riskmap/Ntipped_riskmap_main.pdf")
# fig.show()
fig.clf()
plt.close()










#plotting
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

########################################################################################################################
#Number of tipped elements

########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim00.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax0.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="black", levels=[0.5, 1.0])#, levels=[1.0, 1.5, 2.0])
sc = ax0.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax0, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax0.set_xlabel("Peak temperature [°C]")
ax0.set_ylabel("Convergence time [yr]")
ax0.set_xlim([2.0, 4.0])
ax0.set_xticks(np.arange(2.0, 4.5, 0.5))
ax0.set_yticks(np.arange(100, 1100, 100))
min_val = np.amin(nr_tipped)
ax0.text(2.0, 125, "#$_{tipped, min}$= %.2f" % (min_val))
ax0.text(1.5, 975, "$\\mathbf{a}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim05.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax1.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="black", levels=[0.5, 1.0])
sc = ax1.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax1.set_xlabel("Peak temperature [°C]")
ax1.set_ylabel("Convergence time [yr]")
ax1.set_xlim([2.0, 4.0])
ax1.set_xticks(np.arange(2.0, 4.5, 0.5))
ax1.set_yticks(np.arange(100, 1100, 100))
min_val = np.amin(nr_tipped)
ax1.text(2.0, 125, "#$_{tipped, min}$= %.2f" % (min_val))
ax1.text(1.5, 975, "$\\mathbf{b}$")



########################################################################################################################################################
#RISKMAPS

########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim00.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax2.tricontour(tri_refi, tip_risk_refi, levels=[50, 66], linewidths=2.0, colors="white")
tri = ax2.tricontour(tri_refi, tip_risk_refi, levels=[33], linewidths=2.0, colors="#943126")
sc = ax2.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax2, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax2.set_xlabel("Peak temperature [°C]")
ax2.set_ylabel("Convergence time [yr]")
ax2.set_xticks(np.arange(2.0, 4.5, 0.5))
ax2.set_xlim([2.0, 4.0])
ax2.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(np.amin(tip_risk), 2)
ax2.text(2.0, 500, "Risk", color="#943126")
ax2.text(2.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
ax2.text(1.5, 975, "$\\mathbf{c}$")



########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim05.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax3.tricontour(tri_refi, tip_risk_refi, levels=[50, 66], linewidths=2.0, colors="white")
tri = ax3.tricontour(tri_refi, tip_risk_refi, levels=[33], linewidths=2.0, colors="#943126")
sc = ax3.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
ax3.text(3.5, 500, "x", color="black", fontsize=30)
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax3, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax3.set_xlabel("Peak temperature [°C]")
ax3.set_ylabel("Convergence time [yr]")
ax3.set_xticks(np.arange(2.0, 4.5, 0.5))
ax3.set_xlim([2.0, 4.0])
ax3.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(np.amin(tip_risk), 2)
ax3.text(2.0, 500, "Risk", color="#943126")
ax3.text(2.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
ax3.text(1.5, 975, "$\\mathbf{d}$")



#sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("figs/riskmap/Ntipped_riskmap_supplement.png")
fig.savefig("figs/riskmap/Ntipped_riskmap_supplement.pdf")
# fig.show()
fig.clf()
plt.close()













########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################












#####################HIGH-END SCENARIOS
#plotting
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))

########################################################################################################################
#Number of tipped elements

########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim10.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax0.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.0, 1.5, 2.0])
sc = ax0.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax0, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax0.set_xlabel("Peak temperature [°C]")
ax0.set_ylabel("Convergence time [yr]")
ax0.set_xlim([4.0, 6.0])
ax0.set_xticks(np.arange(4.0, 6.5, 0.5))
ax0.set_yticks(np.arange(100, 1100, 100))
min_val = nr_tipped[40]
ax0.text(4.0, 125, "#$_{tipped, min}$= %.2f" % (min_val))
ax0.text(3.5, 975, "$\\mathbf{a}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim15.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax1.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.75, 2.0, 2.5])
sc = ax1.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax1.set_xlabel("Peak temperature [°C]")
ax1.set_ylabel("Convergence time [yr]")
ax1.set_xlim([4.0, 6.0])
ax1.set_xticks(np.arange(4.0, 6.5, 0.5))
ax1.set_yticks(np.arange(100, 1100, 100))
min_val = nr_tipped[40]
ax1.text(4.0, 125, "#$_{tipped, min}$= %.2f" % (min_val), color="white")
ax1.text(3.5, 975, "$\\mathbf{b}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim20.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax2.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[2.5, 2.75])#, levels=[1.0, 1.5, 2.0])
sc = ax2.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax2, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax2.set_xlabel("Peak temperature [°C]")
ax2.set_ylabel("Convergence time [yr]")
ax2.set_xlim([4.0, 6.0])
ax2.set_xticks(np.arange(4.0, 6.5, 0.5))
ax2.set_yticks(np.arange(100, 1100, 100))
min_val = nr_tipped[40]
ax2.text(4.0, 125, "#$_{tipped, min}$= %.2f" % (min_val), color="white")
ax2.text(3.5, 975, "$\\mathbf{c}$")




########################################################################################################################################################
#RISKMAPS

########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim10.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax3.tricontour(tri_refi, tip_risk_refi, levels=[66, 75, 90], linewidths=2.0, colors="white")
sc = ax3.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax3, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax3.set_xlabel("Peak temperature [°C]")
ax3.set_ylabel("Convergence time [yr]")
ax3.set_xticks(np.arange(4.0, 6.5, 0.5))
ax3.set_xlim([4.0, 6.0])
ax3.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(tip_risk[40], 2)
ax3.text(4.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"), color="white")
ax3.text(3.5, 975, "$\\mathbf{d}$")



########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim15.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax4.tricontour(tri_refi, tip_risk_refi, levels=[75, 90, 99], linewidths=2.0, colors="white")
sc = ax4.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax4, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax4.set_xlabel("Peak temperature [°C]")
ax4.set_ylabel("Convergence time [yr]")
ax4.set_xticks(np.arange(4.0, 6.5, 0.5))
ax4.set_xlim([4.0, 6.0])
ax4.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(tip_risk[40], 2)
ax4.text(4.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"), color="white")
ax4.text(3.5, 975, "$\\mathbf{e}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim20.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax5.tricontour(tri_refi, tip_risk_refi, levels=[95, 99, 99.9], linewidths=2.0, colors="white")
sc = ax5.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax5, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax5.set_xlabel("Peak temperature [°C]")
ax5.set_ylabel("Convergence time [yr]")
ax5.set_xticks(np.arange(4.0, 6.5, 0.5))
ax5.set_xlim([4.0, 6.0])
ax5.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(tip_risk[40], 2)
ax5.text(4.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"), color="white")
ax5.text(3.5, 975, "$\\mathbf{f}$")



#sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("figs/riskmap/Ntipped_riskmap_main_highend.png")
fig.savefig("figs/riskmap/Ntipped_riskmap_main_highend.pdf")
# fig.show()
fig.clf()
plt.close()


########################################################################################################################
########################################################################################################################


#plotting
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

########################################################################################################################
#Number of tipped elements

########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim00.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax0.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.0, 1.5, 2.0])
sc = ax0.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax0, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax0.set_xlabel("Peak temperature [°C]")
ax0.set_ylabel("Convergence time [yr]")
ax0.set_xlim([4.0, 6.0])
ax0.set_xticks(np.arange(4.0, 6.5, 0.5))
ax0.set_yticks(np.arange(100, 1100, 100))
min_val = nr_tipped[40]
ax0.text(4.0, 125, "#$_{tipped, min}$= %.2f" % (min_val))
ax0.text(3.5, 975, "$\\mathbf{a}$")


########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim05.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
tri = ax1.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.0, 1.5, 2.0])
sc = ax1.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax1.set_xlabel("Peak temperature [°C]")
ax1.set_ylabel("Convergence time [yr]")
ax1.set_xlim([4.0, 6.0])
ax1.set_xticks(np.arange(4.0, 6.5, 0.5))
ax1.set_yticks(np.arange(100, 1100, 100))
min_val = nr_tipped[40]
ax1.text(4.0, 125, "#$_{tipped, min}$= %.2f" % (min_val))
ax1.text(3.5, 975, "$\\mathbf{b}$")




########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim00.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax2.tricontour(tri_refi, tip_risk_refi, levels=[66, 75, 90], linewidths=2.0, colors="white")
sc = ax2.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax2, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax2.set_xlabel("Peak temperature [°C]")
ax2.set_ylabel("Convergence time [yr]")
ax2.set_xticks(np.arange(4.0, 6.5, 0.5))
ax2.set_xlim([4.0, 6.0])
ax2.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(tip_risk[40], 2)
ax2.text(4.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
ax2.text(3.5, 975, "$\\mathbf{c}$")



########DATA
data = np.loadtxt("figs/riskmap/riskmap_Tlim05.txt")
Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]
######

#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax3.tricontour(tri_refi, tip_risk_refi, levels=[66, 75, 90], linewidths=2.0, colors="white")
sc = ax3.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
ax3.text(5.5, 500, "x", color="black", fontsize=30)
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax3, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax3.set_xlabel("Peak temperature [°C]")
ax3.set_ylabel("Convergence time [yr]")
ax3.set_xticks(np.arange(4.0, 6.5, 0.5))
ax3.set_xlim([4.0, 6.0])
ax3.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(tip_risk[40], 2)
ax3.text(4.0, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
ax3.text(3.5, 975, "$\\mathbf{d}$")



#sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("figs/riskmap/Ntipped_riskmap_supplement_highend.png")
fig.savefig("figs/riskmap/Ntipped_riskmap_supplement_highend.pdf")
# fig.show()
fig.clf()
plt.close()



print("Finish")