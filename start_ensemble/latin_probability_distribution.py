from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.25)
from pyDOE import * #function name >>> lhs

elements = {}
#Tipping limits, see Schellnhuber, et al., 2016:
elements["limits_gis"]= [0.8, 3.2]
elements["limits_thc"]= [1.4, 8.0]
elements["limits_wais"] = [1.0, 3.0]
elements["limits_amaz"] = [2.0, 6.0]
elements["limits_nino"] = [3.0, 6.0]
elements["limits_assi"] = [1.3, 2.9]

###################################################
# for now also rosser 2024
#TO GIS
elements["pf_wais_to_gis"] = [0.1, 0.2]
elements["pf_thc_to_gis"] = [0.1, 1.]
# TO THC
elements["pf_gis_to_thc"] = [0.1, 1.]
elements["pf_nino_to_thc"] = [0.1, 0.2]
elements["pf_wais_to_thc"] = [0.1, 0.3] 
elements["pf_assi_to_thc"] = [0.1, 0.5]
# TO WAIS"
elements["pf_nino_to_wais"] = [0.1, 0.5]
elements["pf_thc_to_wais" ]= [0.1, 0.15]
elements["pf_gis_to_wais" ]= [0.1, 1.0]
#TO AMAZ
elements["pf_nino_to_amaz"] = [0.1, 1.0]
elements["pf_thc_to_amaz" ]= [0.1, 0.4] 
#TO NINO
elements["pf_thc_to_nino"] = [0.1, 0.2]
# TO ASSI
elements["pf_thc_to_assi"] = [0.5, 0.1]

# TIMINGS
# Rosser 2024
elements["tau_gis"]=[1000, 15000]
elements["tau_thc"]=[15,300]
elements["tau_wais"]=[500, 13000]
elements["tau_nino"]=[25, 200]
elements["tau_amaz"]=[50, 200]
elements["tau_assi"]=[10,50]
"""
Latin hypercube sampling
"""
points = np.array(lhs(len(elements.keys()), samples=100)) #give dimensions and sample size, here shown for a Latin hypercube

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

array_limits = np.array(array_limits)
np.savetxt("latin_prob.txt", array_limits, delimiter=" ")


#Create .sh file to run on the cluster
# sh_file = np.array(sh_file)
# np.savetxt("latin_sh_file.txt", sh_file, delimiter=" ", fmt="%s")




#tipping ranges and plots
gis = array_limits.T[0]
thc = array_limits.T[1]
wais = array_limits.T[2]
amaz = array_limits.T[3]


plt.grid(True)
plt.hist(gis, 24, facecolor='c', alpha=0.5, label="GIS")
plt.hist(thc, 25, facecolor='b', alpha=0.5, label="THC")
plt.hist(wais, 47, facecolor='k', alpha=0.5, label="WAIS")
plt.hist(amaz, 10, facecolor='g', alpha=0.5, label="AMAZ")
plt.legend(loc='best')
plt.xlabel("Tipping range [°C]")
plt.ylabel("N [#]")
plt.tight_layout()
plt.savefig("latin_prob_TR.png")
plt.savefig("latin_prob_TR.pdf")
#plt.show()
plt.clf()
plt.close()


#coupling strength
wais_to_gis = array_limits.T[4]
thc_to_gis = array_limits.T[5]
gis_to_thc = array_limits.T[6]
wais_to_thc = array_limits.T[7]
thc_to_wais = array_limits.T[8]
gis_to_wais = array_limits.T[9]
thc_to_amaz_pos = array_limits.T[10]


plt.grid(True)
plt.hist(wais_to_gis, 10, facecolor='c', alpha=0.5, label="wais_to_gis")
plt.hist(thc_to_gis, 100, facecolor='b', alpha=0.5, label="thc_to_gis")
plt.hist(gis_to_thc, 100, facecolor='k', alpha=0.5, label="gis_to_thc")
plt.hist(wais_to_thc, 30, facecolor='r', alpha=0.5, label="wais_to_thc")
plt.hist(thc_to_wais, 5, facecolor='#2D9575', alpha=0.5, label="thc_to_wais")
plt.hist(gis_to_wais, 100, facecolor='#8E58C3', alpha=0.5, label="gis_to_wais")
plt.hist(thc_to_amaz_pos, 40, facecolor='#FF5733', alpha=0.5, label="thc_to_amaz")
plt.legend(loc='best')
plt.xlabel("Probability fraction [a.u.]")
plt.ylabel("N [#]")
plt.tight_layout()
plt.savefig("latin_prob_PF.png")
plt.savefig("latin_prob_PF.pdf")
#plt.show()
plt.clf()
plt.close()

print("Finish")



