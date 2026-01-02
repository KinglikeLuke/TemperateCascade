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


def data_plotter(jj, kk): #jj=overshoot_data_index; kk=tipping_risk_index
    density = 50 #density=number of samples on the x-axis
    min_num_sam = 5 # minimum acceptable number of samples in interval

    plot_data = []
    plot_vals = np.linspace(np.amin(mean.T[jj].flatten()), np.amax(mean.T[jj].flatten()), density+1)
    for i in range(1, len(plot_vals)):
        plot_arg = np.argwhere((mean.T[jj].flatten() >= plot_vals[i-1]) & (mean.T[jj].flatten() <= plot_vals[i]))
        if len(plot_arg) >= min_num_sam:
            plot_aim_y = mean.T[kk].flatten()[plot_arg]
            plot_data.append([plot_vals[i-1]+0.5*(plot_vals[i]-plot_vals[i-1]), np.mean(plot_aim_y), np.std(plot_aim_y)]) #pretty larger errors - if standard error is used, this problem is solved: divide np.std(plot_aim_y)/np.sqrt(len(plot_arg))
    
    return np.array(plot_data)



def data_plotter_all(all_data, tip_data): #data=overshoot data; tip_data=tipping_data
    density = 50 #density=number of samples on the x-axis
    min_num_sam = 5 # minimum acceptable number of samples in interval

    plot_data = []
    plot_vals = np.linspace(np.amin(all_data), np.amax(all_data), density+1)
    for i in range(1, len(plot_vals)):
        plot_arg = np.argwhere((all_data >= plot_vals[i-1]) & (all_data <= plot_vals[i]))
        if len(plot_arg) >= min_num_sam:
            plot_aim_y = tip_data[plot_arg]
            plot_data.append([plot_vals[i-1]+0.5*(plot_vals[i]-plot_vals[i-1]), np.mean(plot_aim_y), np.std(plot_aim_y)]) 

    return np.array(plot_data)



# read all files
networks = np.sort(glob.glob("latin/*0_0.0_*"))
print(networks)


#find all possible values to loop over them
data = np.loadtxt("{}/data_preparator_overshoot.txt".format(networks[0]))
##


Tlim_vals = np.unique(data.T[0])
Tpeak_vals = np.unique(data.T[1])
tconv_vals = np.unique(data.T[2])
coupling_vals = np.unique(data.T[3])


output_all = []
for coupling in coupling_vals:
    print(coupling)
    output = []
    for network in networks:
        #print(network)
        data = np.loadtxt("{}/data_preparator_overshoot.txt".format(network))

        ##rounding errors in T_peak must be handled appropriately
        data.T[1] = np.where(data.T[1]==1.9, 2.0, data.T[1])
        data.T[1] = np.where(data.T[1]==2.9, 3.0, data.T[1])
        data.T[1] = np.where(data.T[1]==3.9, 4.0, data.T[1])
        data.T[1] = np.where(data.T[1]==4.9, 5.0, data.T[1])
        data.T[1] = np.where(data.T[1]==5.9, 6.0, data.T[1])
        ##

        t_max = 1000.0 #maximum number of years that a tipping element is allowed to be transgressd (otherwise it would be infinity)


        data_sampled = np.argwhere((data.T[3]==coupling) & (data.T[10]<=t_max) & (data.T[13]<=t_max) & (data.T[16]<=t_max) & (data.T[19]<=t_max))
        data_sampled = data[data_sampled]
        #print(len(data_sampled))

        output.append(data_sampled)

    output = np.array(output)


    std = np.std(output, axis=0)
    mean = np.mean(output, axis=0)

    for i in range(0, len(mean.T[0].flatten())):
        output_all.append([
            mean.T[0].flatten()[i], mean.T[1].flatten()[i], mean.T[2].flatten()[i], mean.T[3].flatten()[i], #Tlim, Tpeak, tconv, coupling,
            mean.T[4].flatten()[i], std.T[4].flatten()[i],    #all tipping elements

            mean.T[5].flatten()[i], std.T[5].flatten()[i],    #gis tipping
            mean.T[6].flatten()[i], std.T[6].flatten()[i],    #thc tipping
            mean.T[7].flatten()[i], std.T[7].flatten()[i],    #wais tipping
            mean.T[8].flatten()[i], std.T[8].flatten()[i],    #amaz tipping

            mean.T[9].flatten()[i], std.T[9].flatten()[i],    #gis_ov
            mean.T[10].flatten()[i], std.T[10].flatten()[i],  #gis_time_ov
            mean.T[11].flatten()[i], std.T[11].flatten()[i],  #gis_max_ov

            mean.T[12].flatten()[i], std.T[12].flatten()[i],  #thc_ov
            mean.T[13].flatten()[i], std.T[13].flatten()[i],  #thc_time_ov
            mean.T[14].flatten()[i], std.T[14].flatten()[i],  #thc_max_ov        

            mean.T[15].flatten()[i], std.T[15].flatten()[i],  #wais_ov
            mean.T[16].flatten()[i], std.T[16].flatten()[i],  #wais_time_ov
            mean.T[17].flatten()[i], std.T[17].flatten()[i],  #wais_max_ov     

            mean.T[18].flatten()[i], std.T[18].flatten()[i],  #amaz_ov
            mean.T[19].flatten()[i], std.T[19].flatten()[i],  #amaz_time_ov
            mean.T[20].flatten()[i], std.T[20].flatten()[i]   #amaz_max_ov        
            ])
        


    # plot mean figure
    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11)) = plt.subplots(4, 3, figsize=(25, 26))
    ################################ALL TIPPING ELEMENTS################################
    ax0.set_title("GIS Coupling d={}".format(coupling))
    data_plot = data_plotter(9, 5)
    ax0.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="c")
    #ax0.set_xticks(np.arange(100, 1100, 100))
    #ax0.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax0.set_xlabel("Overshoot strength [yr*°C]")
    ax0.set_ylabel("Tipping risk [%]")


    ax1.set_title("GIS Coupling d={}".format(coupling))
    data_plot = data_plotter(10, 5)
    ax1.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="c")
    #ax1.set_xticks(np.arange(100, 1100, 100))
    #ax1.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax1.set_xlabel("Time over thresh. [yr]")
    ax1.set_ylabel("Tipping risk [%]")


    ax2.set_title("GIS Coupling d={}".format(coupling))
    data_plot = data_plotter(11, 5)
    ax2.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="c")
    #ax2.set_xticks(np.arange(100, 1100, 100))
    #ax2.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax2.set_xlabel("Max. over thresh. [°C]")
    ax2.set_ylabel("Tipping risk [%]")


    #########################################################
    #########################################################
    ax3.set_title("THC Coupling d={}".format(coupling))
    data_plot = data_plotter(12, 6)
    ax3.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="b")
    #ax3.set_xticks(np.arange(100, 1100, 100))
    #ax3.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax3.set_xlabel("Overshoot strength [yr*°C]")
    ax3.set_ylabel("Tipping risk [%]")


    ax4.set_title("THC Coupling d={}".format(coupling))
    data_plot = data_plotter(13, 6)
    ax4.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="b")
    #ax4.set_xticks(np.arange(100, 1100, 100))
    #ax4.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax4.set_xlabel("Time over thresh. [yr]")
    ax4.set_ylabel("Tipping risk [%]")


    ax5.set_title("THC Coupling d={}".format(coupling))
    data_plot = data_plotter(14, 6)
    ax5.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="b")
    #ax5.set_xticks(np.arange(100, 1100, 100))
    #ax5.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax5.set_xlabel("Max. over thresh. [°C]")
    ax5.set_ylabel("Tipping risk [%]")



    #########################################################
    #########################################################
    ax6.set_title("WAIS Coupling d={}".format(coupling))
    data_plot = data_plotter(15, 7)
    ax6.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="k")
    #ax6.set_xticks(np.arange(100, 1100, 100))
    #ax6.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax6.set_xlabel("Overshoot strength [yr*°C]")
    ax6.set_ylabel("Tipping risk [%]")


    ax7.set_title("WAIS Coupling d={}".format(coupling))
    data_plot = data_plotter(16, 7)
    ax7.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="k")
    #ax7.set_xticks(np.arange(100, 1100, 100))
    #ax7.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax7.set_xlabel("Time over thresh. [yr]")
    ax7.set_ylabel("Tipping risk [%]")


    ax8.set_title("WAIS Coupling d={}".format(coupling))
    data_plot = data_plotter(17, 7)
    ax8.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="k")
    #ax8.set_xticks(np.arange(100, 1100, 100))
    #ax8.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax8.set_xlabel("Max. over thresh. [°C]")
    ax8.set_ylabel("Tipping risk [%]")



    #########################################################
    #########################################################
    ax9.set_title("AMAZ Coupling d={}".format(coupling))
    data_plot = data_plotter(18, 8)
    ax9.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="g")
    #ax9.set_xticks(np.arange(100, 1100, 100))
    #ax9.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax9.set_xlabel("Overshoot strength [yr*°C]")
    ax9.set_ylabel("Tipping risk [%]")


    ax10.set_title("AMAZ Coupling d={}".format(coupling))
    data_plot = data_plotter(19, 8)
    ax10.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="g")
    #ax10.set_xticks(np.arange(100, 1100, 100))
    #ax10.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax10.set_xlabel("Time over thresh. [yr]")
    ax10.set_ylabel("Tipping risk [%]")


    ax11.set_title("AMAZ Coupling d={}".format(coupling))
    data_plot = data_plotter(20, 8)
    ax11.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="g")
    #ax11.set_xticks(np.arange(100, 1100, 100))
    #ax11.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
    ax11.set_xlabel("Max. over thresh. [°C]")
    ax11.set_ylabel("Tipping risk [%]")



    sns.despine(bottom=True, left=True) #no right and upper border lines
    fig.tight_layout()
    fig.savefig("latin/all/overshoot/overshoot_evaluator_cpl{}.png".format(coupling))
    fig.savefig("latin/all/overshoot/overshoot_evaluator_cpl{}.pdf".format(coupling))
    # fig.show()
    fig.clf()
    plt.close()




output_all = np.array(output_all)
np.savetxt("latin/all/overshoot/overshoot_evaluator_all.txt", output_all)


# plot mean figure
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(25, 6.5))

################################ALL TIPPING ELEMENTS################################
#must be divided by 4 since 4 different tipping experiments are added and the average time/overshoot_temp per experiment can be obtained by dividing through 4
data_plot = data_plotter_all((output_all.T[14] + output_all.T[20] + output_all.T[26] + output_all.T[32])/4, output_all.T[4])
ax0.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="r")
#ax0.set_xticks(np.arange(100, 1100, 100))
#ax0.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax0.set_xlabel("Overshoot strength[yr*°C]")
ax0.set_ylabel("N(tipped)")

data_plot = data_plotter_all((output_all.T[16] + output_all.T[22] + output_all.T[28] + output_all.T[34])/4, output_all.T[4])
ax1.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="r")
#ax1.set_xticks(np.arange(100, 1100, 100))
#ax1.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax1.set_xlabel("Time over thresh. [yr]")
ax1.set_ylabel("N(tipped)")

data_plot = data_plotter_all((output_all.T[18] + output_all.T[24] + output_all.T[30] + output_all.T[36])/4, output_all.T[4])
ax2.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="r")
#ax2.set_xticks(np.arange(100, 1100, 100))
#ax2.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax2.set_xlabel("Max. over thresh. [°C]")
ax2.set_ylabel("N(tipped)")


sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("latin/all/overshoot/overshoot_evaluator_all.png")
fig.savefig("latin/all/overshoot/overshoot_evaluator_all.pdf")
# fig.show()
fig.clf()
plt.close()



################################################################################################
################################################################################################
# plot mean figure
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11)) = plt.subplots(4, 3, figsize=(25, 26))

################################ALL TIPPING ELEMENTS################################
ax0.set_title("GIS")
data_plot = data_plotter_all(output_all.T[14], output_all.T[6])
ax0.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="c")
#ax0.set_xticks(np.arange(100, 1100, 100))
#ax0.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax0.set_xlabel("Overshoot strength [yr*°C]")
ax0.set_ylabel("Tipping risk [%]")


ax1.set_title("GIS")
data_plot = data_plotter_all(output_all.T[16], output_all.T[6])
ax1.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="c")
#ax1.set_xticks(np.arange(100, 1100, 100))
#ax1.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax1.set_xlabel("Time over thresh. [yr]")
ax1.set_ylabel("Tipping risk [%]")


ax2.set_title("GIS")
data_plot = data_plotter_all(output_all.T[18], output_all.T[6])
ax2.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="c")
#ax2.set_xticks(np.arange(100, 1100, 100))
#ax2.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax2.set_xlabel("Max. over thresh. [°C]")
ax2.set_ylabel("Tipping risk [%]")


#########################################################
#########################################################
ax3.set_title("THC")
data_plot = data_plotter_all(output_all.T[20], output_all.T[8])
ax3.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="b")
#ax3.set_xticks(np.arange(100, 1100, 100))
#ax3.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax3.set_xlabel("Overshoot strength [yr*°C]")
ax3.set_ylabel("Tipping risk [%]")


ax4.set_title("THC")
data_plot = data_plotter_all(output_all.T[22], output_all.T[8])
ax4.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="b")
#ax4.set_xticks(np.arange(100, 1100, 100))
#ax4.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax4.set_xlabel("Time over thresh. [yr]")
ax4.set_ylabel("Tipping risk [%]")


ax5.set_title("THC")
data_plot = data_plotter_all(output_all.T[24], output_all.T[8])
ax5.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="b")
#ax5.set_xticks(np.arange(100, 1100, 100))
#ax5.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax5.set_xlabel("Max. over thresh. [°C]")
ax5.set_ylabel("Tipping risk [%]")



#########################################################
#########################################################
ax6.set_title("WAIS")
data_plot = data_plotter_all(output_all.T[26], output_all.T[10])
ax6.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="k")
#ax6.set_xticks(np.arange(100, 1100, 100))
#ax6.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax6.set_xlabel("Overshoot strength [yr*°C]")
ax6.set_ylabel("Tipping risk [%]")


ax7.set_title("WAIS")
data_plot = data_plotter_all(output_all.T[28], output_all.T[10])
ax7.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="k")
#ax7.set_xticks(np.arange(100, 1100, 100))
#ax7.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax7.set_xlabel("Time over thresh. [yr]")
ax7.set_ylabel("Tipping risk [%]")


ax8.set_title("WAIS")
data_plot = data_plotter_all(output_all.T[30], output_all.T[10])
ax8.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="k")
#ax8.set_xticks(np.arange(100, 1100, 100))
#ax8.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax8.set_xlabel("Max. over thresh. [°C]")
ax8.set_ylabel("Tipping risk [%]")



#########################################################
#########################################################
ax9.set_title("AMAZ")
data_plot = data_plotter_all(output_all.T[32], output_all.T[12])
ax9.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="g")
#ax9.set_xticks(np.arange(100, 1100, 100))
#ax9.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax9.set_xlabel("Overshoot strength [yr*°C]")
ax9.set_ylabel("Tipping risk [%]")


ax10.set_title("AMAZ")
data_plot = data_plotter_all(output_all.T[34], output_all.T[12])
ax10.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="g")
#ax10.set_xticks(np.arange(100, 1100, 100))
#ax10.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax10.set_xlabel("Time over thresh. [yr]")
ax10.set_ylabel("Tipping risk [%]")


ax11.set_title("AMAZ")
data_plot = data_plotter_all(output_all.T[36], output_all.T[12])
ax11.errorbar(data_plot.T[0], data_plot.T[1], yerr=data_plot.T[2], color="g")
#ax11.set_xticks(np.arange(100, 1100, 100))
#ax11.set_yticks(np.array([0.0, 1.0, 1.5, 2.0]))
ax11.set_xlabel("Max. over thresh. [°C]")
ax11.set_ylabel("Tipping risk [%]")



sns.despine(bottom=True, left=True) #no right and upper border lines
fig.tight_layout()
fig.savefig("latin/all/overshoot/overshoot_evaluator_all_singleTEs.png")
fig.savefig("latin/all/overshoot/overshoot_evaluator_all_singleTEs.pdf")
# fig.show()
fig.clf()
plt.close()





print("Finish")