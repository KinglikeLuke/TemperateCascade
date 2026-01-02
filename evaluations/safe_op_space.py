import numpy as np
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set(font_scale=1.5)
import re
import glob
import os



# read all files
networks = np.sort(glob.glob("latin/*0_0.0_*"))
print(networks)

for network in networks:
    print(network)
    #Saving and plotting structure
    data = np.loadtxt("{}/data_preparator.txt".format(network))


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

            output = []
            for Tlim in Tlim_vals:
                #print(Tlim)
                for tconv in tconv_vals:
                    #print(tconv)
                    data_sampled = np.argwhere((data.T[0]==Tlim) & (data.T[1]==Tpeak) & (data.T[2]==tconv) & (data.T[3]==coupling))
                    data_sampled = data[data_sampled]

                    if len(data_sampled) != 100 and len(data_sampled)!=99: #until last run has been finalized keep 99 
                        print("Error: data_sampled should have a length of 100, but has {}".format(len(data_sampled)))
                        die

                    output.append([Tlim, Tpeak, tconv, coupling, 
                        np.mean(data_sampled.T[4]), np.std(data_sampled.T[4]), #all tipping elements
                        np.mean(data_sampled.T[5]), np.std(data_sampled.T[5]), #gis
                        np.mean(data_sampled.T[6]), np.std(data_sampled.T[6]), #thc
                        np.mean(data_sampled.T[7]), np.std(data_sampled.T[7]), #wais
                        np.mean(data_sampled.T[8]), np.std(data_sampled.T[8])  #amaz
                        ])
            output = np.array(output)
            Tlim = output.T[0]
            tconv = output.T[2]
            tipped_mean = output.T[4]
            tipped_std = output.T[5]


            #save figure
            np.savetxt("{}/dynamic_classes_mean_cpl{}_Tpeak{}.txt".format(network, coupling, Tpeak), output)

            # plot mean figure
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 6.5))

            #more colomaps from Fabio Crameri (perceptually uniform)
            cmap_data = np.loadtxt("ScientificColourMaps4/lajolla/lajolla.txt")
            cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)


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
            fig.savefig("{}/dynamic_classes_mean_cpl{}_Tpeak{}.png".format(network, coupling, Tpeak))
            fig.savefig("{}/dynamic_classes_mean_cpl{}_Tpeak{}.pdf".format(network, coupling, Tpeak))
            # fig.show()
            fig.clf()
            plt.close()


print("Finish")