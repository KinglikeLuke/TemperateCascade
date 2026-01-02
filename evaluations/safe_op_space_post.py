import numpy as np
import re
import glob
import os

import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors

import seaborn as sns
sns.set(font_scale=1.5)




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



# read all files
networks = np.sort(glob.glob("latin/*0_0.0_*"))
print(networks)

#find all possible values to loop over them
data = np.loadtxt("{}/data_preparator.txt".format(networks[0]))


##rounding errors in T_peak must be handled appropriately
data.T[1] = np.where(data.T[1]==1.9, 2.0, data.T[1])
data.T[1] = np.where(data.T[1]==2.9, 3.0, data.T[1])
data.T[1] = np.where(data.T[1]==3.9, 4.0, data.T[1])
data.T[1] = np.where(data.T[1]==4.9, 5.0, data.T[1])
data.T[1] = np.where(data.T[1]==5.9, 6.0, data.T[1])
##


Tlim_vals = np.unique(data.T[0])
Tpeak_vals = np.unique(data.T[1])
tconv_vals = np.unique(data.T[2])
coupling_vals = np.unique(data.T[3])



for Tpeak in Tpeak_vals:
    print(Tpeak)
    for coupling in coupling_vals:
        print(coupling)
        output_all = []
        for Tlim in Tlim_vals:
            for tconv in tconv_vals:
                output = []
                for network in networks:
                    data = np.loadtxt("{}/dynamic_classes_mean_cpl{}_Tpeak{}.txt".format(network, coupling, Tpeak))

                    data_sampled = np.argwhere((data.T[0]==Tlim) & (data.T[1]==Tpeak) & (data.T[2]==tconv) & (data.T[3]==coupling))
                    data_sampled = data[data_sampled][0,0]
                    output.append(data_sampled)

                output = np.array(output)
                std = np.std(output, axis=0)
                mean = np.mean(output, axis=0)

                output_all.append([
                    mean[0], mean[1], mean[2], mean[3], #Tlim, Tpeak, tconv, coupling,
                    mean[4], std[4],   #all tipping elements
                    mean[6], std[6],   #gis
                    mean[8], std[8],   #thc
                    mean[10], std[10], #wais
                    mean[12], std[12]  #amaz
                    ])
        output_all = np.array(output_all)
        Tlim = output_all.T[0]
        tconv = output_all.T[2]
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


        #save figure
        np.savetxt("latin/all/dynamic_classes_mean_cpl{}_Tpeak{}.txt".format(coupling, Tpeak), output_all)

        # plot mean figure
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 6.5))

        #more colomaps from Fabio Crameri (perceptually uniform)
        cmap_data = np.loadtxt("ScientificColourMaps4/lajolla/lajolla.txt")
        cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)

        ################################ALL TIPPING ELEMENTS################################
        #ax0.axis([-0.1, 8.1, -0.05, 1.05])
        ax0.set_title("Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax0.scatter(tconv, Tlim, c=tipped_mean, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 4.0])
        cbar = fig.colorbar(sc, ax=ax0, ticks=[0, 1, 2, 3, 4], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '1.0', '2.0', '3.0', '4.0'])
        cbar.set_label(label="Number of tipped elements")#, fontsize=10)
        cbar.set_clim([-0.5, 4.75])
        #ax0.set_title("Mean values")
        ax0.set_xticks(np.arange(100, 1100, 100))
        ax0.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax0.set_xlabel("Convergence time [yr]")
        ax0.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")
        #ax0.add_patch(Rectangle((-0.1, 0.51), 8.2, 0.5, alpha=0.5, facecolor='white', edgecolor="none")) #plt.add_patch(Rectangle((left, bottom), width, height))
        #ax0.text(-2.5, 0.95, "$\\mathbf{a}$", fontsize=25)


        #ax1.axis([-0.1, 8.1, -0.05, 1.05])
        sc = ax1.scatter(tconv, Tlim, c=tipped_std, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax1, ticks=[0, 0.5, 1.0, 1.5, 2.0], pad=0.1)
        sc.set_clim([0.0, 2.0])
        cbar.ax.set_yticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'])
        cbar.set_label(label="Standard deviation")#, fontsize=10)
        cbar.set_clim([-0.25, 2.5])
        #ax1.set_title("Standard deviation")
        ax1.set_xticks(np.arange(100, 1100, 100))
        ax1.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax1.set_xlabel("Convergence time [yr]")
        ax1.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")
        #ax1.add_patch(Rectangle((-0.1, 0.51), 8.2, 0.5, alpha=0.5, facecolor='white', edgecolor="none")) #plt.add_patch(Rectangle((left, bottom), width, height))
        #ax1.text(-2.5, 0.95, "$\\mathbf{b}$", fontsize=25)

        sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        fig.savefig("latin/all/dynamic_classes_mean_cpl{}_Tpeak{}.png".format(coupling, Tpeak))
        fig.savefig("latin/all/dynamic_classes_mean_cpl{}_Tpeak{}.pdf".format(coupling, Tpeak))
        # fig.show()
        fig.clf()
        plt.close()


        ##############COMPUTE DIFFERENCE TO CONSTANT TEMPERATURE [I.E. THE ACTUAL EFFECT OF THE OVERSHOOT]##############
        #load constant temperature profiles for the respective scenario
        all_mean_const_array = tipping_difference("empty", Tlim_vals, coupling, tconv_vals)
        all_mean_const_array = np.subtract(tipped_mean, all_mean_const_array)


        #save figure
        np.savetxt("latin/all/dynamic_classes_mean_cpl{}_Tpeak{}_additional.txt".format(coupling, Tpeak), output_all)

        # plot mean figure
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 6.5))

        #more colomaps from Fabio Crameri (perceptually uniform)
        cmap_data = np.loadtxt("ScientificColourMaps4/lajolla/lajolla.txt")
        cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)

        ################################ALL TIPPING ELEMENTS################################
        ax0.set_title("Additional tipping Cpl. d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax0.scatter(tconv, Tlim, c=all_mean_const_array, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 4.0])
        cbar = fig.colorbar(sc, ax=ax0, pad=0.1, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(['0.0', '1.0', '2.0', '3.0', '4.0'])
        cbar.set_label(label="Number of tipped elements")
        cbar.set_clim([-0.5, 4.75])

        ax0.set_xticks(np.arange(100, 1100, 100))
        ax0.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax0.set_xlabel("Convergence time [yr]")
        ax0.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")


        ax1.set_title("Additional tipping Cpl. d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax1.scatter(tconv, Tlim, c=all_mean_const_array, cmap=cmap, marker="s", s=500, edgecolors="k")
        #sc.set_clim([0.0, 4.0])
        cbar = fig.colorbar(sc, ax=ax1, pad=0.1)#, ticks=[0, 1, 2, 3, 4])
        #cbar.ax.set_yticklabels(['0.0', '1.0', '2.0', '3.0', '4.0'])
        cbar.set_label(label="Number of tipped elements")
        #cbar.set_clim([-0.5, 4.75])

        ax1.set_xticks(np.arange(100, 1100, 100))
        ax1.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax1.set_xlabel("Convergence time [yr]")
        ax1.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        fig.savefig("latin/all/dynamic_classes_mean_cpl{}_Tpeak{}_additional.png".format(coupling, Tpeak))
        fig.savefig("latin/all/dynamic_classes_mean_cpl{}_Tpeak{}_additional.pdf".format(coupling, Tpeak))
        # fig.show()
        fig.clf()
        plt.close()
        ##############################################################################################






        ################################################################################################
        ################################################################################################
        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(4, 2, figsize=(17, 26))



        ax0.set_title("GIS Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax0.scatter(tconv, Tlim, c=gis_mean, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax0, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of GIS tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        #ax0.set_title("Mean values")
        ax0.set_xticks(np.arange(100, 1100, 100))
        ax0.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax0.set_xlabel("Convergence time [yr]")
        ax0.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax1.scatter(tconv, Tlim, c=gis_std, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax1, ticks=[0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        sc.set_clim([0.0, 1.0])
        cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Standard deviation")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax1.set_xticks(np.arange(100, 1100, 100))
        ax1.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax1.set_xlabel("Convergence time [yr]")
        ax1.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")


        ax2.set_title("THC Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax2.scatter(tconv, Tlim, c=thc_mean, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax2, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of THC tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax2.set_xticks(np.arange(100, 1100, 100))
        ax2.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax2.set_xlabel("Convergence time [yr]")
        ax2.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax3.scatter(tconv, Tlim, c=thc_std, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax3, ticks=[0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        sc.set_clim([0.0, 1.0])
        cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Standard deviation")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax3.set_xticks(np.arange(100, 1100, 100))
        ax3.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax3.set_xlabel("Convergence time [yr]")
        ax3.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")


        ax4.set_title("WAIS Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax4.scatter(tconv, Tlim, c=wais_mean, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax4, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of WAIS tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax4.set_xticks(np.arange(100, 1100, 100))
        ax4.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax4.set_xlabel("Convergence time [yr]")
        ax4.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax5.scatter(tconv, Tlim, c=wais_std, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax5, ticks=[0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        sc.set_clim([0.0, 1.0])
        cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Standard deviation")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax5.set_xticks(np.arange(100, 1100, 100))
        ax5.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax5.set_xlabel("Convergence time [yr]")
        ax5.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")


        ax6.set_title("AMAZ Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax6.scatter(tconv, Tlim, c=amaz_mean, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax6, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of AMAZ tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax6.set_xticks(np.arange(100, 1100, 100))
        ax6.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax6.set_xlabel("Convergence time [yr]")
        ax6.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax7.scatter(tconv, Tlim, c=amaz_std, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax7, ticks=[0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        sc.set_clim([0.0, 1.0])
        cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Standard deviation")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax7.set_xticks(np.arange(100, 1100, 100))
        ax7.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax7.set_xlabel("Convergence time [yr]")
        ax7.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")



        sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        fig.savefig("latin/all/dynamic_classes_singletes_mean_cpl{}_Tpeak{}.png".format(coupling, Tpeak))
        fig.savefig("latin/all/dynamic_classes_singletes_mean_cpl{}_Tpeak{}.pdf".format(coupling, Tpeak))
        # fig.show()
        fig.clf()
        plt.close()



        ##############COMPUTE DIFFERENCE TO CONSTANT TEMPERATURE [I.E. THE ACTUAL EFFECT OF THE OVERSHOOT]##############
        ####################################################SINGLE TEs##################################################
        #load constant temperature profiles for the respective scenario
        all_mean_const_array_gis = tipping_difference("gis", Tlim_vals, coupling, tconv_vals)
        all_mean_const_array_gis = np.subtract(gis_mean, all_mean_const_array_gis)

        all_mean_const_array_thc = tipping_difference("thc", Tlim_vals, coupling, tconv_vals)
        all_mean_const_array_thc = np.subtract(thc_mean, all_mean_const_array_thc)

        all_mean_const_array_wais = tipping_difference("wais", Tlim_vals, coupling, tconv_vals)
        all_mean_const_array_wais = np.subtract(wais_mean, all_mean_const_array_wais)

        all_mean_const_array_amaz = tipping_difference("amaz", Tlim_vals, coupling, tconv_vals)
        all_mean_const_array_amaz = np.subtract(amaz_mean, all_mean_const_array_amaz)



        #Plotting
        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(4, 2, figsize=(17, 26))



        ax0.set_title("GIS Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax0.scatter(tconv, Tlim, c=all_mean_const_array_gis, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax0, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. GIS tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        #ax0.set_title("Mean values")
        ax0.set_xticks(np.arange(100, 1100, 100))
        ax0.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax0.set_xlabel("Convergence time [yr]")
        ax0.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax1.scatter(tconv, Tlim, c=all_mean_const_array_gis, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax1, pad=0.1)#, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        #sc.set_clim([0.0, 1.0])
        #cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. GIS tipping")#, fontsize=10)
        #cbar.set_clim([-0.1, 1.1])
        ax1.set_xticks(np.arange(100, 1100, 100))
        ax1.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax1.set_xlabel("Convergence time [yr]")
        ax1.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")


        ax2.set_title("THC Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax2.scatter(tconv, Tlim, c=all_mean_const_array_thc, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax2, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. THC tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax2.set_xticks(np.arange(100, 1100, 100))
        ax2.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax2.set_xlabel("Convergence time [yr]")
        ax2.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax3.scatter(tconv, Tlim, c=all_mean_const_array_thc, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax3, pad=0.1)#, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        #sc.set_clim([0.0, 1.0])
        #cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. THC tipping")#, fontsize=10)
        #cbar.set_clim([-0.1, 1.1])
        ax3.set_xticks(np.arange(100, 1100, 100))
        ax3.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax3.set_xlabel("Convergence time [yr]")
        ax3.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")


        ax4.set_title("WAIS Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax4.scatter(tconv, Tlim, c=all_mean_const_array_wais, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax4, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. WAIS tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax4.set_xticks(np.arange(100, 1100, 100))
        ax4.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax4.set_xlabel("Convergence time [yr]")
        ax4.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax5.scatter(tconv, Tlim, c=all_mean_const_array_wais, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax5, pad=0.1)#, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        #sc.set_clim([0.0, 1.0])
        #cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. WAIS tipping")#, fontsize=10)
        #cbar.set_clim([-0.1, 1.1])
        ax5.set_xticks(np.arange(100, 1100, 100))
        ax5.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax5.set_xlabel("Convergence time [yr]")
        ax5.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")


        ax6.set_title("AMAZ Coupling d={}, Tpeak={}°C".format(coupling, Tpeak))
        sc = ax6.scatter(tconv, Tlim, c=all_mean_const_array_amaz, cmap=cmap, marker="s", s=500, edgecolors="k")
        sc.set_clim([0.0, 1.0])
        cbar = fig.colorbar(sc, ax=ax6, ticks=[0.0, 0.25, 0.5, 0.75, 1.0], pad=0.1)
        cbar.ax.set_yticklabels(['0.0', '0.15', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. AMAZ tipping")#, fontsize=10)
        cbar.set_clim([-0.1, 1.1])
        ax6.set_xticks(np.arange(100, 1100, 100))
        ax6.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax6.set_xlabel("Convergence time [yr]")
        ax6.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")

        sc = ax7.scatter(tconv, Tlim, c=all_mean_const_array_amaz, cmap=cmap, marker="s", s=500, edgecolors="k")
        cbar = fig.colorbar(sc, ax=ax7, pad=0.1)# ticks=[0, 0.25, 0.5, 0.75, 1.0])
        #sc.set_clim([0.0, 1.0])
        #cbar.ax.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        cbar.set_label(label="Risk of add. AMAZ tipping")#, fontsize=10)
        #cbar.set_clim([-0.1, 1.1])
        ax7.set_xticks(np.arange(100, 1100, 100))
        ax7.set_yticks(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        ax7.set_xlabel("Convergence time [yr]")
        ax7.set_ylabel("Saturation temperature (T$_{lim}$) [°C]")



        sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        fig.savefig("latin/all/dynamic_classes_singletes_mean_cpl{}_Tpeak{}_additional.png".format(coupling, Tpeak))
        fig.savefig("latin/all/dynamic_classes_singletes_mean_cpl{}_Tpeak{}_additional.pdf".format(coupling, Tpeak))
        # fig.show()
        fig.clf()
        plt.close()



print("Finish")