import numpy as np
import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set(font_scale=1.)
import re
import glob
import os


#needs to be mounted for cluster run
os.chdir(os.getcwd() + "evaluations")

# read all files
networks = np.array(np.sort(glob.glob("../results/no_feedbacks/*0_0.0_*")))

for network in networks:
    print(network)
    net_splitter = re.split("network", network)[-1] #used for the saving structure


    #open and read file
    output = []

    # read all files
    folders = np.array(np.sort(glob.glob(network + "/0*")))
    for folder in folders:
        print(folder)
        subfolders = np.array(np.sort(glob.glob(folder + "/feedbacks_*.txt")))


        for kk in subfolders:
            #find coupling
            #print(kk)
            resplit = re.split("Tlim|Tpeak|tconv|_|.txt", kk)

            Tlim = float(resplit[-7])/10.0
            Tpeak = float(resplit[-5])/10.0
            tconv = np.round(float(resplit[-3]),-2) 
            coupling = float(resplit[-2])


            tip_info = np.loadtxt(kk)[5:]
            if len(tip_info) != 0:

                tip_info = np.insert(tip_info, 0, np.array([Tlim, Tpeak, tconv, coupling]))
                output.append([float(i) for i in tip_info])
            else:
                print("Missing file: {}".format(kk))

    data = np.array(output)

    #Saving and plotting structure
    np.savetxt("latin/network{}/data_preparator.txt".format(net_splitter), data)
