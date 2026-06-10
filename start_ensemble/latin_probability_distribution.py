import pandas as pd
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set(font_scale=1.25)
from pydoe import lhs #function name >>> lhs

elements = {}
#Tipping limits, see Schellnhuber, et al., 2016:
elements["limits_GIS"]= [0.8, 3.2]
elements["limits_AMOC"]= [1.4, 8.0]
elements["limits_WAIS"] = [1.0, 3.0]
elements["limits_Amazonas"] = [2.0, 6.0]
elements["limits_NINO"] = [3.0, 6.0]
elements["limits_REEF"]=[1.0, 2.0]
elements["limits_WAM"]=[2.0, 3.5]
elements["limits_AWSI"]=[4.5, 8.7]
elements["limits_PERM"]=[3.0, 6.0]
# elements["limits_ASSI"] = [1.3, 2.9]

###################################################
# PFs as given in Kriegler, with sign changed depending on what newer GTPRs list
#TO GIS
elements["pf_WAIS_to_GIS"] = [1, 2]
elements["pf_AMOC_to_GIS"] = [0.1, 1.]
# TO AMOC
elements["pf_GIS_to_AMOC"] = [1., 10.]
elements["pf_NINO_to_AMOC"] = [0.5, 2]
elements["pf_WAIS_to_AMOC"] = [0.6, 1] # reaching 0.3 is effectively impossible. Also, weak, but stabilizing influence in GTPR2025
# elements["pf_ASSI_to_AMOC"] = [0.1, 0.5]
# TO WAIS. Generally, WAIS tips so often in an "intermediate" temperature trajectory that any factor greater than 2 is unattainable
elements["pf_NINO_to_WAIS"] = [1, 5] # TODO rerun with 5
elements["pf_AMOC_to_WAIS" ]= [1, 1.5]
elements["pf_GIS_to_WAIS" ]= [1, 5.0] # ultra sketch. I dont know how they imagine a tenfold increase between ice sheets (having very similar limiting temperature and all...)
#TO AMAZ
elements["pf_NINO_to_Amazonas"] = [1, 10]
elements["pf_AMOC_to_Amazonas" ]= [0.5, 1] # sketch. Probably stabilizing, regional variations
#TO NINO
elements["pf_AMOC_to_NINO"] = [0.1, 0.2]
# # TO ASSI
# elements["pf_AMOC_to_ASSI"] = [0.5, 0.1]
# Bara Connections
elements["pf_AWSI_to_AMOC"] = [1, 3]
elements["pf_AWSI_to_GIS"] = [1, 2]
elements["pf_AWSI_to_PERM"] = [1, 2]
elements["pf_AMOC_to_AWSI"] = [0.3, 1]
elements["pf_AMOC_to_WAM"] = [1, 1.5]
elements["pf_NINO_to_REEF"] = [1, 10]
elements["pf_PERM_to_AMOC"] = [1, 1.5]

# TIMINGS
# Rosser 2024
elements["GIS_time"]=[1000, 15000]
elements["AMOC_time"]=[15,300]
elements["WAIS_time"]=[500, 13000]
elements["NINO_time"]=[25, 200]
elements["Amazonas_time"]=[50, 200]
elements["REEF_time"]=[10, 11]
elements["WAM_time"]=[10, 500]
elements["AWSI_time"]=[10, 100]
elements["PERM_time"]=[10, 300]
# elements["ASSI_time"]=[10,50]

with open("limits.json", "w") as file:
    json.dump(elements, file)
# limit_filename = r"start_ensemble\limits.json"
# with open(limit_filename, "r") as file:
#     elements = json.load(file)
"""
Latin hypercube sampling
"""
points = np.array(lhs(len(elements.keys()), samples=200)) #give dimensions and sample size, here shown for a Latin hypercube

#rescaling function from latin hypercube
def latin_function(limits, rand):
    resc_rand = limits[0] + (limits[1] - limits[0]) * rand
    return resc_rand

#MAIN
array_limits = []
sh_file = []
for i in range(0, len(points)):
    print(i)
    array_limits.append([latin_function(value, points[i][element_ind]) for element_ind, value in enumerate(elements.values())])

array_limits = pd.DataFrame(array_limits, columns=list(elements.keys()))
array_limits = array_limits.to_csv("latin_prob.txt", index=False)
# np.savetxt("latin_prob.txt", array_limits, delimiter=" ")


#Create .sh file to run on the cluster
# sh_file = np.array(sh_file)
# np.savetxt("latin_sh_file.txt", sh_file, delimiter=" ", fmt="%s")




#tipping ranges and plots
# GIS = array_limits.T[0]
# AMOC = array_limits.T[1]
# WAIS = array_limits.T[2]
# Amazonas = array_limits.T[3]
#
#
# plt.grid(True)
# plt.hist(GIS, 24, facecolor='c', alpha=0.5, label="GIS")
# plt.hist(AMOC, 25, facecolor='b', alpha=0.5, label="AMOC")
# plt.hist(WAIS, 47, facecolor='k', alpha=0.5, label="WAIS")
# plt.hist(Amazonas, 10, facecolor='g', alpha=0.5, label="Amazonas")
# plt.legend(loc='best')
# plt.xlabel("Tipping range [°C]")
# plt.ylabel("N [#]")
# plt.tight_layout()
# plt.savefig("latin_prob_TR.png")
# plt.savefig("latin_prob_TR.pdf")
# #plt.show()
# plt.clf()
# plt.close()
#
#
# #coupling strength
# WAIS_to_GIS = array_limits.T[4]
# AMOC_to_GIS = array_limits.T[5]
# GIS_to_AMOC = array_limits.T[6]
# WAIS_to_AMOC = array_limits.T[7]
# AMOC_to_WAIS = array_limits.T[8]
# GIS_to_WAIS = array_limits.T[9]
# AMOC_to_Amazonas_pos = array_limits.T[10]
#
#
# plt.grid(True)
# plt.hist(WAIS_to_GIS, 10, facecolor='c', alpha=0.5, label="WAIS_to_GIS")
# plt.hist(AMOC_to_GIS, 100, facecolor='b', alpha=0.5, label="AMOC_to_GIS")
# plt.hist(GIS_to_AMOC, 100, facecolor='k', alpha=0.5, label="GIS_to_AMOC")
# plt.hist(WAIS_to_AMOC, 30, facecolor='r', alpha=0.5, label="WAIS_to_AMOC")
# plt.hist(AMOC_to_WAIS, 5, facecolor='#2D9575', alpha=0.5, label="AMOC_to_WAIS")
# plt.hist(GIS_to_WAIS, 100, facecolor='#8E58C3', alpha=0.5, label="GIS_to_WAIS")
# plt.hist(AMOC_to_Amazonas_pos, 40, facecolor='#FF5733', alpha=0.5, label="AMOC_to_Amazonas")
# plt.legend(loc='best')
# plt.xlabel("Probability fraction [a.u.]")
# plt.ylabel("N [#]")
# plt.tight_layout()
# plt.savefig("latin_prob_PF.png")
# plt.savefig("latin_prob_PF.pdf")
# #plt.show()
# plt.clf()
# plt.close()

print("Finish")



