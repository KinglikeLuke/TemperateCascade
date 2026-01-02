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

#cmaplist = [cmap(i) for i in range(cmap.N)]
#colors = np.linspace(75, 255, 6)
#colors = np.array([int(x) for x in colors])
############



Tlim_vals = np.array(["00", "05", "10", "15", "20"])

for Tlim in Tlim_vals:
    print(Tlim)
    data = np.loadtxt("figs/riskmap/riskmap_Tlim{}.txt".format(Tlim))
    Tpeak = data.T[0]
    tconv = data.T[1]
    nr_tipped = data.T[2]
    nr_tipped_std = data.T[3]
    tip_risk = data.T[4]

    if Tlim != "20":
        #plotting
        fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

        #tricontour regridder to smoothen all lines
        tri = Triangulation(Tpeak, tconv)
        refiner = UniformTriRefiner(tri)
        tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=5)
        tri = ax0.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.0, 1.5, 2.0])#, levels=[1.0, 1.5, 2.0])
        sc = ax0.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
        #sc.set_clim([0.0, 2.5])

        cbar = fig.colorbar(sc, ax=ax0, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        cbar.set_label(label="Number of tipped elements")
        ax0.set_xlabel("Peak temperature [°C]")
        ax0.set_ylabel("Convergence time [yr]")
        ax0.set_xticks(np.arange(2.5, 6.5, 0.5))
        ax0.set_yticks(np.arange(100, 1100, 100))
        min_val = np.amin(nr_tipped)
        ax0.text(2.6, 125, "Number$_{tipped, min}$= %.2f" % (min_val))
        ax0.text(1.65, 975, "$\\mathbf{a}$")


        #tricontour regridder to smoothen all lines
        tri = Triangulation(Tpeak, tconv)
        refiner = UniformTriRefiner(tri)
        tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
        tri = ax1.tricontour(tri_refi, tip_risk_refi, levels=[50, 75, 90], linewidths=2.0, colors="white")
        sc = ax1.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
        #sc.set_clim([0.0, 2.5])

        cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cbar.set_label(label="Tipping risk [%]")
        ax1.set_xlabel("Peak temperature [°C]")
        ax1.set_ylabel("Convergence time [yr]")
        ax1.set_xticks(np.arange(2.5, 6.5, 0.5))
        ax1.set_yticks(np.arange(100, 1100, 100))
        min_val = 100*np.round(np.amin(tip_risk), 2)
        ax1.text(2.6, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
        ax1.text(1.65, 975, "$\\mathbf{b}$")


        #sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        fig.savefig("figs/riskmap/riskmap_Tlim{}.png".format(Tlim))
        fig.savefig("figs/riskmap/riskmap_Tlim{}.pdf".format(Tlim))
        # fig.show()
        fig.clf()
        plt.close()

    else:
        #plotting
        fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        
        #tricontour regridder to smoothen all lines
        tri = Triangulation(Tpeak, tconv)
        refiner = UniformTriRefiner(tri)
        tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=3)
        tri = ax0.tricontour(tri_refi, nr_tipped_refi, linewidths=2.0, colors="white", levels=[1.5, 2.0, 2.5])#, levels=[1.0, 1.5, 2.0])
        sc = ax0.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
        #sc.set_clim([0.0, 2.5])

        cbar = fig.colorbar(sc, ax=ax0, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        cbar.set_label(label="Number of tipped elements")
        ax0.set_xlabel("Peak temperature [°C]")
        ax0.set_ylabel("Convergence time [yr]")
        ax0.set_xticks(np.arange(2.5, 6.5, 0.5))
        ax0.set_yticks(np.arange(100, 1100, 100))
        min_val = np.amin(nr_tipped)
        ax0.text(2.6, 125, "Number$_{tipped, min}$= %.2f" % (min_val))
        ax0.text(1.65, 975, "$\\mathbf{a}$")


        #tricontour regridder to smoothen all lines
        tri = Triangulation(Tpeak, tconv)
        refiner = UniformTriRefiner(tri)
        tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
        tri = ax1.tricontour(tri_refi, tip_risk_refi, levels=[75, 90, 95], linewidths=2.0, colors="white")
        sc = ax1.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100.1, 100))
        #sc.set_clim([0.0, 2.5])

        cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cbar.set_label(label="Tipping risk [%]")
        ax1.set_xlabel("Peak temperature [°C]")
        ax1.set_ylabel("Convergence time [yr]")
        ax1.set_xticks(np.arange(2.5, 6.5, 0.5))
        ax1.set_yticks(np.arange(100, 1100, 100))
        min_val = 100*np.round(np.amin(tip_risk), 2)
        ax1.text(2.6, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
        ax1.text(1.65, 975, "$\\mathbf{b}$")



        #sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        fig.savefig("figs/riskmap/riskmap_Tlim{}.png".format(Tlim))
        fig.savefig("figs/riskmap/riskmap_Tlim{}.pdf".format(Tlim))
        # fig.show()
        fig.clf()
        plt.close()



#####ALL Average data
print("All")
data = np.loadtxt("figs/riskmap/riskmap.txt") #Tpeak, tconv, nr_tipped, nr_tipped_std, tip_risk

Tpeak = data.T[0]
tconv = data.T[1]
nr_tipped = data.T[2]
nr_tipped_std = data.T[3]
tip_risk = data.T[4]



#plotting
fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))


#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, nr_tipped_refi = refiner.refine_field(nr_tipped, subdiv=3)
tri = ax0.tricontour(tri_refi, nr_tipped_refi, levels=[1.0, 1.5, 2.0], linewidths=2.0, colors="white")
sc = ax0.tricontourf(Tpeak, tconv, nr_tipped, cmap=cmap, levels=np.linspace(0.0, 3.0, 100))
sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax0, orientation="vertical", ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_label(label="Number of tipped elements")
ax0.set_xlabel("Peak temperature [°C]")
ax0.set_ylabel("Convergence time [yr]")
ax0.set_xticks(np.arange(2.5, 6.5, 0.5))
ax0.set_yticks(np.arange(100, 1100, 100))
min_val = np.amin(nr_tipped)
ax0.text(2.6, 125, "Number$_{tipped, min}$= %.2f" % (min_val))
ax0.text(1.65, 975, "$\\mathbf{a}$")



#tricontour regridder to smoothen all lines
tri = Triangulation(Tpeak, tconv)
refiner = UniformTriRefiner(tri)
tri_refi, tip_risk_refi = refiner.refine_field(100*tip_risk, subdiv=3)
tri = ax1.tricontour(tri_refi, tip_risk_refi, 100*tip_risk, levels=[50, 75, 90], linewidths=2.0, colors="white")
sc = ax1.tricontourf(Tpeak, tconv, 100*tip_risk, cmap=cmap, levels=np.linspace(0.0, 100, 100))
#sc.set_clim([0.0, 2.5])

cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.set_label(label="Tipping risk [%]")
ax1.set_xlabel("Peak temperature [°C]")
ax1.set_ylabel("Convergence time [yr]")
ax1.set_xticks(np.arange(2.5, 6.5, 0.5))
ax1.set_yticks(np.arange(100, 1100, 100))
min_val = 100*np.round(np.amin(tip_risk), 2)
ax1.text(2.6, 125, "Risk$_{tipping, min}$= %d%s" % (min_val, "%"))
ax1.text(1.65, 975, "$\\mathbf{b}$")



#sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("figs/riskmap/riskmap.png")
fig.savefig("figs/riskmap/riskmap.pdf")
# fig.show()
fig.clf()
plt.close()
#########################END##########################






print("Finish")