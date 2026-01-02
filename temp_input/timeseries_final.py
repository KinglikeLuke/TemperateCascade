# global imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
sns.despine()



#global variables
t_vals = np.arange(0, 50001, 1) 

"""
reference see Ritchie et al., 2021, Nature and references therein
"""

#Temperature input funciton
def T_input(t, T_0, mu_0, mu_1, T_lim, y):
	temp_input = T_0 + y*t - (1 - np.exp(-(mu_0+mu_1*t)*t))*(y*t - (T_lim - T_0))
	return temp_input

#proportionality factor (based on current warming rates)
def gamma(T_0, mu_0, T_lim, R):
	y = R + mu_0*(T_0-T_lim)
	return y



####################################MAIN####################################
T_0 = 1.0

#Data
data = np.loadtxt(r"D:\Dokumente\Uni\PhD application\Climate\figshare_overshoots_paper\temp_input\Tpeak_tconv_values\temp_input_values.txt", comments=['#']) #T_peak T_lim t_conv R mu_0 mu_1


for i in data:
	#print(i)
	T_peak_det = i[0]
	T_lim = i[1]
	t_conv_det = i[2]
	R = i[3]
	mu_0 = i[4]
	mu_1 = i[5]


	y = gamma(T_0, mu_0, T_lim, R) #y can also be fixed at 0.02
	output = []
	for t in t_vals:
		temp_input = T_input(t, T_0, mu_0, mu_1, T_lim, y)
		output.append([t, mu_0, mu_1, T_0, T_lim, temp_input])
	output = np.array(output)

	#saving structure
	time = output.T[0]
	temp = output.T[-1]
	T_peak = np.amax(temp) #peak temperature


	#convergence time
	tolerance = 0.01
	arg_conv = np.argwhere(temp < T_lim+tolerance).flatten()
	#check that the temperature is going down
	for i in range(1, len(arg_conv)):
		index0 = arg_conv[i-1]
		index1 = arg_conv[i]
		if temp[index0]-temp[index1] > 0:
			t_conv = time[index1]
			break


	############
	print("Peak temperature: {}°C".format(T_peak))
	print("Convergence time to {}°C: {} years".format(T_lim, t_conv))
	############

	#Data saving
	np.savetxt("timeseries_final/Tlim{}_Tpeak{}_tconv{}.txt".format(int(10*T_lim), int(np.round(10*T_peak, 1)), int(t_conv)), output)
	#Plotting
	fig, ax0 = plt.subplots(1, 1, figsize=(8.5, 6.5))
	ax0.set_title("T_lim={} °C, T_peak={} °C, t_conv={} yr".format(T_lim, T_peak_det, t_conv_det))
	ax0.plot(time, temp, color="#922B21", label="GMT timeseries")
	ax0.set_xlim([-10, 2000.0])
	ax0.set_ylim([-0.1, 5.1])
	ax0.set_xlabel("Time [yr]")
	ax0.set_ylabel(r"$\Delta$ GMT [°C]")
	ax0.legend(loc="best")


	fig.tight_layout()
	fig.savefig("timeseries_final/Tlim{}_Tpeak{}_tconv{}.png".format(int(10*T_lim), int(np.round(10*T_peak, 1)), int(np.round(t_conv, -2))))
	fig.savefig("timeseries_final/Tlim{}_Tpeak{}_tconv{}.pdf".format(int(10*T_lim), int(np.round(10*T_peak, 1)), int(np.round(t_conv, -2))))
	#fig.show()
	fig.clf()
	plt.close()

print("Finish")