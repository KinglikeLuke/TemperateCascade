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
cmap_data = np.loadtxt("ScientificColourMaps4/lajolla/lajolla.txt")
cmap = colors.LinearSegmentedColormap.from_list('CBname', cmap_data)

cmaplist = [cmap(i) for i in range(cmap.N)]
colors = np.linspace(75, 255, 6)
colors = np.array([int(x) for x in colors])
############


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



#Do all averging for all four investigated variables
var_array = np.array([Tlim_vals, Tpeak_vals, tconv_vals, coupling_vals])
name_file = np.array(["Tlim", "Tpeak", "tconv", "cpl"])
for i in range(0, len(name_file)):
    print(name_file[i])
    ####COUPLING
    output_all = []
    output_all_long = []
    for var in var_array[i]:
        print(var)
        output = []
        output_long = []
        for network in networks:
            output_network = []
            output_network_long = []
            #print(network)
            data = np.loadtxt("{}/data_preparator_overshoot.txt".format(network))

            t_max = 1000.0 #maximum number of years that a tipping element is allowed to be transgressd (otherwise it would be infinity)

            data_sampled = np.argwhere((data.T[i]==var) & (data.T[10]<=t_max) & (data.T[13]<=t_max) & (data.T[16]<=t_max) & (data.T[19]<=t_max))
            data_sampled = data[data_sampled]
            output.append(data_sampled)

            for jj in data_sampled:
                output_network.append([j for j in jj[0]])
            output_network = np.array(output_network)
            np.savetxt("{}/overshoot_hist_{}_{}.txt".format(network, name_file[i], str(var)), output_network)



            data_sampled_long = np.argwhere((data.T[i]==var))
            data_sampled_long = data[data_sampled_long]
            output_long.append(data_sampled_long)
            
            for jj in data_sampled_long:
                output_network_long.append([j for j in jj[0]])
            output_network_long = np.array(output_network_long)
            np.savetxt("{}/overshoot_histlong_{}_{}.txt".format(network, name_file[i], str(var)), output_network_long)


        output = np.array(output)
        #########################Overshoot Histogramm##########################
        #count how many samples show tipping for zero, all elements and for each element individually
        zero_tipped  = np.count_nonzero(output.T[4].flatten()==0.)/len(output.T[4].flatten())
        one_tipped   = np.count_nonzero(output.T[4].flatten()==1.)/len(output.T[4].flatten())
        two_tipped   = np.count_nonzero(output.T[4].flatten()==2.)/len(output.T[4].flatten())
        three_tipped = np.count_nonzero(output.T[4].flatten()==3.)/len(output.T[4].flatten())
        four_tipped  = np.count_nonzero(output.T[4].flatten()==4.)/len(output.T[4].flatten())

        gis_tipped  = np.count_nonzero(output.T[5].flatten()==1.)/len(output.T[5].flatten())
        thc_tipped  = np.count_nonzero(output.T[6].flatten()==1.)/len(output.T[6].flatten())
        wais_tipped = np.count_nonzero(output.T[7].flatten()==1.)/len(output.T[7].flatten())
        amaz_tipped = np.count_nonzero(output.T[8].flatten()==1.)/len(output.T[8].flatten())


        output_all.append([var, zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped, gis_tipped, thc_tipped, wais_tipped, amaz_tipped])


        #plotting
        fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        ax0.grid(True)
        ax0.bar(np.arange(5), np.array([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped]), color=[cmaplist[i] for i in colors], align='center', alpha=0.85, ecolor='k')
        ax0.set_xticks(np.arange(5))
        ax0.set_xticklabels(["0", "1", "2", "3", "4"])
        ax0.set_xlabel("Elements tipped")
        ax0.set_ylabel("Tipping risk [%]")
        ax0.set_ylim([0., 1.0])
        ax0.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


        ax1.grid(True)
        ax1.bar(np.arange(4), np.array([gis_tipped, thc_tipped, wais_tipped, amaz_tipped]), color=["c", "b", "k", "g"], align='center', alpha=0.85, ecolor='k')
        ax1.set_xticks(np.arange(4))
        ax1.set_xticklabels(["GIS", "THC", "WAIS", "AMAZ"])
        ax1.set_xlabel("Tipped")
        ax1.set_ylabel("Tipping risk [%]")
        ax1.set_ylim([0., 1.0])
        ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


        #sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        var_val = np.round(var, 2) #to get normally rounded coupling values
        fig.savefig("latin/all/overshoot/hist/overshoot_hist_{}_{}.png".format(name_file[i], var_val))
        fig.savefig("latin/all/overshoot/hist/overshoot_hist_{}_{}.pdf".format(name_file[i], var_val))
        # fig.show()
        fig.clf()
        plt.close()
        #########################END##########################



        output_long = np.array(output_long)
        #########################Overshoot Histogramm##########################
        #count how many samples show tipping for zero, all elements and for each element individually
        zero_tipped  = np.count_nonzero(output_long.T[4].flatten()==0.)/len(output_long.T[4].flatten())
        one_tipped   = np.count_nonzero(output_long.T[4].flatten()==1.)/len(output_long.T[4].flatten())
        two_tipped   = np.count_nonzero(output_long.T[4].flatten()==2.)/len(output_long.T[4].flatten())
        three_tipped = np.count_nonzero(output_long.T[4].flatten()==3.)/len(output_long.T[4].flatten())
        four_tipped  = np.count_nonzero(output_long.T[4].flatten()==4.)/len(output_long.T[4].flatten())

        gis_tipped  = np.count_nonzero(output_long.T[5].flatten()==1.)/len(output_long.T[5].flatten())
        thc_tipped  = np.count_nonzero(output_long.T[6].flatten()==1.)/len(output_long.T[6].flatten())
        wais_tipped = np.count_nonzero(output_long.T[7].flatten()==1.)/len(output_long.T[7].flatten())
        amaz_tipped = np.count_nonzero(output_long.T[8].flatten()==1.)/len(output_long.T[8].flatten())


        output_all_long.append([var, zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped, gis_tipped, thc_tipped, wais_tipped, amaz_tipped])


        #plotting
        fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        ax0.grid(True)
        ax0.bar(np.arange(5), np.array([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped]), color=[cmaplist[i] for i in colors], align='center', alpha=0.85, ecolor='k')
        ax0.set_xticks(np.arange(5))
        ax0.set_xticklabels(["0", "1", "2", "3", "4"])
        ax0.set_xlabel("Elements tipped")
        ax0.set_ylabel("Tipping risk [%]")
        ax0.set_ylim([0., 1.0])
        ax0.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


        ax1.grid(True)
        ax1.bar(np.arange(4), np.array([gis_tipped, thc_tipped, wais_tipped, amaz_tipped]), color=["c", "b", "k", "g"], align='center', alpha=0.85, ecolor='k')
        ax1.set_xticks(np.arange(4))
        ax1.set_xticklabels(["GIS", "THC", "WAIS", "AMAZ"])
        ax1.set_xlabel("Tipped")
        ax1.set_ylabel("Tipping risk [%]")
        ax1.set_ylim([0., 1.0])
        ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


        #sns.despine(bottom=True, left=True) #no right and upper border lines
        fig.tight_layout()
        var_val = np.round(var, 2) #to get normally rounded coupling values
        fig.savefig("latin/all/overshoot/hist/long/overshoot_hist_long_{}_{}.png".format(name_file[i], var_val))
        fig.savefig("latin/all/overshoot/hist/long/overshoot_hist_long_{}_{}.pdf".format(name_file[i], var_val))
        # fig.show()
        fig.clf()
        plt.close()
        #########################END##########################



    #########################Overshoot Histogramm##########################
    output_all = np.array(output_all)
    np.savetxt("latin/all/overshoot/hist/overshoot_hist_all_{}.txt".format(name_file[i]), output_all)

    zero_tipped = np.mean(output_all.T[1], axis=0)
    zero_tipped_std = np.std(output_all.T[1], axis=0)
    one_tipped  = np.mean(output_all.T[2], axis=0)
    one_tipped_std  = np.std(output_all.T[2], axis=0)
    two_tipped  = np.mean(output_all.T[3], axis=0)
    two_tipped_std  = np.std(output_all.T[3], axis=0)
    three_tipped = np.mean(output_all.T[4], axis=0)
    three_tipped_std = np.std(output_all.T[4], axis=0)
    four_tipped = np.mean(output_all.T[5], axis=0)
    four_tipped_std = np.std(output_all.T[5], axis=0)

    gis_tipped  = np.mean(output_all.T[6], axis=0)
    gis_tipped_std  = np.std(output_all.T[6], axis=0)
    thc_tipped  = np.mean(output_all.T[7], axis=0)
    thc_tipped_std  = np.std(output_all.T[7], axis=0)
    wais_tipped = np.mean(output_all.T[8], axis=0)
    wais_tipped_std = np.std(output_all.T[8], axis=0)
    amaz_tipped = np.mean(output_all.T[9], axis=0)
    amaz_tipped_std = np.std(output_all.T[9], axis=0)



    #plotting
    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    ax0.grid(True)
    ax0.bar(np.arange(5), np.array([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped]), 
        yerr=np.array([zero_tipped_std, one_tipped_std, two_tipped_std, three_tipped_std, four_tipped_std]),
        color=[cmaplist[i] for i in colors], align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=3, capsize=6, capthick=4, alpha=1.0))
    ax0.set_xticks(np.arange(5))
    ax0.set_xticklabels(["0", "1", "2", "3", "4"])
    ax0.set_xlabel("Elements tipped")
    ax0.set_ylabel("Tipping risk [%]")
    ax0.set_ylim([0., 1.0])
    ax0.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


    ax1.grid(True)
    ax1.bar(np.arange(4), np.array([gis_tipped, thc_tipped, wais_tipped, amaz_tipped]), 
        yerr=np.array([gis_tipped_std, thc_tipped_std, wais_tipped_std, amaz_tipped_std]),
        color=["c", "b", "k", "g"], align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=3, capsize=6, capthick=4, alpha=1.0))
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(["GIS", "THC", "WAIS", "AMAZ"])
    ax1.set_xlabel("Tipped")
    ax1.set_ylabel("Tipping risk [%]")
    ax1.set_ylim([0., 1.0])
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


    #sns.despine(bottom=True, left=True) #no right and upper border lines
    fig.tight_layout()
    fig.savefig("latin/all/overshoot/hist/overshoot_hist_all_{}.png".format(name_file[i]))
    fig.savefig("latin/all/overshoot/hist/overshoot_hist_all_{}.pdf".format(name_file[i]))
    # fig.show()
    fig.clf()
    plt.close()
    #########################END##########################




    #########################Overshoot Histogramm##########################
    output_all_long = np.array(output_all_long)
    np.savetxt("latin/all/overshoot/hist/long/overshoot_hist_long_all_{}.txt".format(name_file[i]), output_all_long)

    zero_tipped = np.mean(output_all_long.T[1], axis=0)
    zero_tipped_std = np.std(output_all_long.T[1], axis=0)
    one_tipped  = np.mean(output_all_long.T[2], axis=0)
    one_tipped_std  = np.std(output_all_long.T[2], axis=0)
    two_tipped  = np.mean(output_all_long.T[3], axis=0)
    two_tipped_std  = np.std(output_all_long.T[3], axis=0)
    three_tipped = np.mean(output_all_long.T[4], axis=0)
    three_tipped_std = np.std(output_all_long.T[4], axis=0)
    four_tipped = np.mean(output_all_long.T[5], axis=0)
    four_tipped_std = np.std(output_all_long.T[5], axis=0)

    gis_tipped  = np.mean(output_all_long.T[6], axis=0)
    gis_tipped_std  = np.std(output_all_long.T[6], axis=0)
    thc_tipped  = np.mean(output_all_long.T[7], axis=0)
    thc_tipped_std  = np.std(output_all_long.T[7], axis=0)
    wais_tipped = np.mean(output_all_long.T[8], axis=0)
    wais_tipped_std = np.std(output_all_long.T[8], axis=0)
    amaz_tipped = np.mean(output_all_long.T[9], axis=0)
    amaz_tipped_std = np.std(output_all_long.T[9], axis=0)



    #plotting
    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    ax0.grid(True)
    ax0.bar(np.arange(5), np.array([zero_tipped, one_tipped, two_tipped, three_tipped, four_tipped]), 
        yerr=np.array([zero_tipped_std, one_tipped_std, two_tipped_std, three_tipped_std, four_tipped_std]),
        color=[cmaplist[i] for i in colors], align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=3, capsize=6, capthick=4, alpha=1.0))
    ax0.set_xticks(np.arange(5))
    ax0.set_xticklabels(["0", "1", "2", "3", "4"])
    ax0.set_xlabel("Elements tipped")
    ax0.set_ylabel("Tipping risk [%]")
    ax0.set_ylim([0., 1.0])
    ax0.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


    ax1.grid(True)
    ax1.bar(np.arange(4), np.array([gis_tipped, thc_tipped, wais_tipped, amaz_tipped]), 
        yerr=np.array([gis_tipped_std, thc_tipped_std, wais_tipped_std, amaz_tipped_std]),
        color=["c", "b", "k", "g"], align='center', alpha=0.85, ecolor='k', error_kw=dict(ecolor='k', lw=3, capsize=6, capthick=4, alpha=1.0))
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(["GIS", "THC", "WAIS", "AMAZ"])
    ax1.set_xlabel("Tipped")
    ax1.set_ylabel("Tipping risk [%]")
    ax1.set_ylim([0., 1.0])
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


    #sns.despine(bottom=True, left=True) #no right and upper border lines
    fig.tight_layout()
    fig.savefig("latin/all/overshoot/hist/long/overshoot_hist_long_all_{}.png".format(name_file[i]))
    fig.savefig("latin/all/overshoot/hist/long/overshoot_hist_long_all_{}.pdf".format(name_file[i]))
    # fig.show()
    fig.clf()
    plt.close()
    #########################END##########################


print("Finish")