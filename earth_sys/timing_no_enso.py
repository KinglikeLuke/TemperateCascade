"""
Timing module: This module computes the conversion factor
between one year in the simulation and one "real" year depending on the tipping time scale of the Amazon rainforest
"""
import sys
sys.path.append('')
import copy

import numpy as np
from scipy.integrate import odeint
from core.tipping_element import cusp
from core.tipping_network import tipping_network
from core.coupling import linear_coupling
from core.evolve import evolve
from earth_sys.functions_earth_system_no_enso import global_functions
from earth_sys.earth_no_enso import EarthParams



class timing():

    def __init__(self, earth_params:EarthParams):
        #Timescales
        self.earth_params = earth_params


        #Compute conversion factor
        self._real_timescale = self.earth_params.gis_time                   					 #value normed to GIS
        self._timescale = self.earth_params.gis_time/self.earth_params.amaz_time                #value normed to GIS
        self._tip_point_gis = 1.8  # most probable tipping point (see Robinson, 2012)    #value normed to GIS
        self._c_krit = np.sqrt(4 / 27)
        self._GMT_cal = 4.0                                        						 #normed temperature
        self._epsilon_c = global_functions.CUSPc(0., self._tip_point_gis, self._GMT_cal) - self._c_krit
        self._initial_state = [-1.]
        self._threshold = 1.0


    """
    Time scale, normed to the shortest tipping scale, in years
    N.B.: Note that we can only insert a RELATIVE time scale, in principle the time scale is dependent on the GMT,
    Here we insert tipping time scales at a temperature around 4°C above pre-industrial, since time scales are shifting during simulation due to structure of CUSP-catastrophe
    """
    def timescales(self):
        new_params = copy.deepcopy(self.earth_params)
        new_params.gis_time /= self.earth_params.amaz_time
        new_params.thc_time /= self.earth_params.amaz_time
        new_params.wais_time /= self.earth_params.amaz_time
        new_params.nino_time /= self.earth_params.amaz_time
        new_params.amaz_time /= self.earth_params.amaz_time
        new_params.assi_time /= self.earth_params.assi_time
        return new_params


    """
    Here we insert a conversion factor to get a translation from a.u. to "true" years
    """
    def conversion(self):
        cusp_deq = cusp(a=-1/self._timescale, b=1/self._timescale, c=self._c_krit/self._timescale)
        net = tipping_network()
        net.add_element(cusp_deq)
        cusp_deq._par['c'] = self._c_krit/self._timescale + self._epsilon_c/self._timescale

        timestep = 0.01
        t_end = 5000
        t_arr = np.arange(0, t_end, timestep)
        sol = odeint( net.f , self._initial_state, t_arr, Dfun=net.jac )
        cusp_deq = sol[:, 0]

        # find point where state crosses threshold
        th = cusp_deq > self._threshold
        th[1:][th[:-1] & th[1:]] = False

        # conversion factor from arbitrary units to years
        conv_fac = self._real_timescale / t_arr[np.nonzero(th)[0][0]]
        return conv_fac




