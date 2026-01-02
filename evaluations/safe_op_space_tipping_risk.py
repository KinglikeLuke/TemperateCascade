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



##PREPARE COLORMAP
#more colormaps from Fabio Crameri (perceptually uniform)
cmap_data = np.loadtxt("ScientificColourMaps4/roma/roma.txt")[::-1]
cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)

cmaplist = [cmap(i) for i in range(cmap.N)]
colors = np.linspace(0, 255, 11)
colors = np.array([int(x) for x in colors])
############

# read all files
networks = np.sort(glob.glob("latin/*0_0.0_*"))
print(networks)

#find all possible values to loop over them
data = np.loadtxt("{}/data_preparator.txt".format(networks[0]))



Tlim_vals = np.unique(data.T[0])
Tpeak_vals = np.unique(data.T[1])
tconv_vals = np.unique(data.T[2])
coupling_vals = np.unique(data.T[3])



######################################################################
#Absolute difference in tipping between steady state scenarios and overshoot scenarios
def tipping_difference(name, Tlim_val, coupling, tconv_vals):
    if name == "empty":
        const_load = np.loadtxt("../../evaluations_no_enso/latin/all/dynamic_classes_mean.txt")
    else:
        const_load = np.loadtxt("../../evaluations_no_enso/latin/all/dynamic_classes_mean_{}.txt".format(name))

    gmt_const = const_load.T[0]
    cpl_const = const_load.T[1]
    all_mean_const = const_load.T[2]
    all_std_const = const_load.T[3]

    #find indices of correct coupling
    all_mean_const_array = []
    idx_cpl = np.where((cpl_const==coupling) & (gmt_const==Tlim_val))[0]
    for j in range(0, len(tconv_vals)):
        all_mean_const_array.append(all_mean_const[idx_cpl][0])
    all_mean_const_array = np.array(all_mean_const_array)
    return all_mean_const_array




for Tpeak in Tpeak_vals:
    print(Tpeak)
    for Tlim in Tlim_vals:
        print(Tlim)


        #start first figure - plot mean figure
        fig, (ax0) = plt.subplots(1, 1, figsize=(8, 6.5))
        ################################ALL TIPPING ELEMENTS################################
        #ax0.axis([-0.1, 8.1, -0.05, 1.05])
        ax0.set_title("Tlim={}°C, Tpeak={}°C".format(Tlim, Tpeak))
        ax0.set_prop_cycle(color=[cmaplist[i] for i in colors])


        #start second figure
        fig2, ((ax00, ax01), (ax02, ax03)) = plt.subplots(2, 2, figsize=(17, 13))
        ax00.set_title("GIS Tlim={}°C, Tpeak={}°C".format(Tlim, Tpeak))
        ax01.set_title("THC Tlim={}°C, Tpeak={}°C".format(Tlim, Tpeak))
        ax02.set_title("WAIS Tlim={}°C, Tpeak={}°C".format(Tlim, Tpeak))
        ax03.set_title("AMAZ Tlim={}°C, Tpeak={}°C".format(Tlim, Tpeak))

        ax00.set_prop_cycle(color=[cmaplist[i] for i in colors])
        ax01.set_prop_cycle(color=[cmaplist[i] for i in colors])
        ax02.set_prop_cycle(color=[cmaplist[i] for i in colors])
        ax03.set_prop_cycle(color=[cmaplist[i] for i in colors])


        #interplot loops
        for coupling in coupling_vals:
            output_all = []
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

            #all TEs
            all_mean_const_array = tipping_difference("empty", Tlim, coupling, tconv_vals)
            all_mean_const_array = np.subtract(tipped_mean, all_mean_const_array)

            #single TEs
            all_mean_const_array_gis = tipping_difference("gis", Tlim, coupling, tconv_vals)
            all_mean_const_array_gis = np.subtract(gis_mean, all_mean_const_array_gis)

            all_mean_const_array_thc = tipping_difference("thc", Tlim, coupling, tconv_vals)
            all_mean_const_array_thc = np.subtract(thc_mean, all_mean_const_array_thc)

            all_mean_const_array_wais = tipping_difference("wais", Tlim, coupling, tconv_vals)
            all_mean_const_array_wais = np.subtract(wais_mean, all_mean_const_array_wais)

            all_mean_const_array_amaz = tipping_difference("amaz", Tlim, coupling, tconv_vals)
            all_mean_const_array_amaz = np.subtract(amaz_mean, all_mean_const_array_amaz)



            #plot part of first figure
            ax0.errorbar(tconv, all_mean_const_array, yerr=0.0, label="{}".format(coupling))

            #plot part of second figure
            ax00.errorbar(tconv, all_mean_const_array_gis, yerr=0.0, label="{}".format(coupling))
            ax01.errorbar(tconv, all_mean_const_array_thc, yerr=0.0, label="{}".format(coupling))
            ax02.errorbar(tconv, all_mean_const_array_wais, yerr=0.0, label="{}".format(coupling))
            ax03.errorbar(tconv, all_mean_const_array_amaz, yerr=0.0, label="{}".format(coupling))


        
        #ax0.set_title("Mean values")
        ax0.set_xticks(np.arange(100, 1100, 100))
        ax0.set_yticks(np.arange(0.0, 3.5, 0.5))
        ax0.set_xlabel("Convergence time [yr]")
        ax0.set_ylabel("N(tipped) [a.u.]")
        ax0.legend(loc="best")

        sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        fig.savefig("latin/all/tipping_risk_coupling/dynamic_classes_mean_Tlim{}_Tpeak{}_additional.png".format(Tlim, Tpeak))
        fig.savefig("latin/all/tipping_risk_coupling/dynamic_classes_mean_Tlim{}_Tpeak{}_additional.pdf".format(Tlim, Tpeak))
        # fig.show()
        fig.clf()
        #plt.close()


        ################################################################################################
        ################################################################################################
        



        ax00.set_xticks(np.arange(100, 1100, 100))
        ax00.set_yticks(np.array([0.0, 0.25, 0.50, 0.75, 1.0]))
        ax00.set_xlabel("Convergence time [yr]")
        ax00.set_ylabel("Tipping risk [a.u.]")
        ax00.legend(loc="best")

        ax01.set_xticks(np.arange(100, 1100, 100))
        ax01.set_yticks(np.array([0.0, 0.25, 0.50, 0.75, 1.0]))
        ax01.set_xlabel("Convergence time [yr]")
        ax01.set_ylabel("Tipping risk [a.u.]")
        ax01.legend(loc="best")

        ax02.set_xticks(np.arange(100, 1100, 100))
        ax02.set_yticks(np.array([0.0, 0.25, 0.50, 0.75, 1.0]))
        ax02.set_xlabel("Convergence time [yr]")
        ax02.set_ylabel("Tipping risk [a.u.]")
        ax02.legend(loc="best")

        ax03.set_xticks(np.arange(100, 1100, 100))
        ax03.set_yticks(np.array([0.0, 0.25, 0.50, 0.75, 1.0]))
        ax03.set_xlabel("Convergence time [yr]")
        ax03.set_ylabel("Tipping risk [a.u.]")
        ax03.legend(loc="best")



        sns.despine(bottom=True, left=True) #no right and upper border lines
        fig2.tight_layout()
        fig2.savefig("latin/all/tipping_risk_coupling/dynamic_classes_singletes_mean_Tlim{}_Tpeak{}_additional.png".format(Tlim, Tpeak))
        fig2.savefig("latin/all/tipping_risk_coupling/dynamic_classes_singletes_mean_Tlim{}_Tpeak{}_additional.pdf".format(Tlim, Tpeak))
        #fig2.show()      
        fig2.clf()
        plt.close()


print("Finish")