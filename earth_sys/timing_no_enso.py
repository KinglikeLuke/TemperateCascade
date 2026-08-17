"""
Timing module: This module computes the conversion factor
between one year in the simulation and one "real" year depending on the tipping time scale of the Amazon rainforest
"""
import sys
sys.path.append('')
import copy

import numpy as np
from scipy.integrate import solve_ivp
from core.tipping_element import cusp, t_cusp
from core.tipping_network import tipping_network
from core.coupling import linear_coupling
from core.evolve import evolve
from earth_sys.earth_no_enso import earth_elements


def individual_timescales(earth_params, temp_offset=2.0):
    """Calibrate absolute internal timescales for the sampled tipping times.

    For a fixed calibration temperature, the isolated cusp element has the form
    dx/dt = F(x, T) / tau. Therefore its threshold-crossing time is exactly
    linear in tau. We only need one simulation per element with tau = 1, then
    set tau = sampled_tipping_time / base_crossing_time.
    """
    raw_net, node_dict, nodes = earth_elements(earth_params, temp_offset)
    new_params = copy.deepcopy(earth_params)
    initial_state = [-1.0]



    def base_tipping_time(element):
        name = element[0]
        element_data = element[1]
        limit_key = f"limits_{name}"
        if limit_key not in earth_params:
            raise KeyError(f"Missing required earth parameter: {limit_key}")
        limit_temp = earth_params[limit_key]
        calibration_temp = limit_temp + temp_offset
        element_data.c.x=calibration_temp

        threshold = 0 if element_data.c.x1 else 1
        def threshold_event(t, y):
            return y[0] - threshold
        threshold_event.terminal = True
        threshold_event.direction = 1

        net = tipping_network()
        net.add_element(element_data)
        sol = solve_ivp(
            net.f,
            (0, 1000000),
            initial_state,
            events=threshold_event,
            jac=net.jac,
            method='LSODA'
        )
        if len(sol.t_events[0]) == 0:
            raise ValueError(
                f"{name} did not cross {threshold} at "
                f"{calibration_temp}C within the calibration window. Max {sol.y[-1]}"
            )
        return sol.t_events[0][0]

    for element in nodes.items():
        time_key = f"{element[0]}_time"
        if time_key not in earth_params:
            raise KeyError(f"Missing required earth parameter: {time_key}")
        new_params[time_key] = earth_params[time_key]**2 / base_tipping_time(element)

    # NINO is a linear relaxation element in earth_network, so its sampled
    # absolute time is already on the same year-based axis.
    return new_params

# class Timing:
#     def __init__(self, earth_params: dict):
#         #Timescales
#         self.earth_params = earth_params
#
#
#         #Compute conversion factor
#         self._real_timescale = self.earth_params['GIS_time']                   					 #value normed to GIS
#         self._timescale = self.earth_params['GIS_time']/self.earth_params['Amazonas_time']                #value normed to GIS
#         self._tip_point_gis = 1.8  # most probable tipping point (see Robinson, 2012)    #value normed to GIS
#         self._c_krit = np.sqrt(4 / 27)
#         self._GMT_cal = 4.0                                        						 #normed temperature
#         self._epsilon_c = global_functions.CUSPc(0., self._tip_point_gis, self._GMT_cal) - self._c_krit
#         self._initial_state = [-1.]
#         self._threshold = 1.0
#
#
#     """
#     Time scale, normed to the shortest tipping scale, in years
#     N.B.: Note that we can only insert a RELATIVE time scale, in principle the time scale is dependent on the GMT,
#     Here we insert tipping time scales at a temperature around 4°C above pre-industrial, since time scales are shifting during simulation due to structure of CUSP-catastrophe
#     """
#     def timescales(self):
#         # e_p = self.earth_params
#         # for t_e in ["GIS", "AMOC", "WAIS", "Amazonas", "REEF", "AWSI", "PERM", "WAM"]:
#         #     # Fucking lambda only saves the reference to t_e, which would then take the last value of t_e, "WAM", for
#         #     # every node. Hence, I have to pass the t_e variable explicitly in the default values.
#         #     network = tipping_network().add_element(t_cusp(a=-1.0 / e_p[f"{t_e}_time"], b=1.0 / e_p[f"{t_e}_time"],
#         #                         c=(1.0 / e_p[f"{t_e}_time"]) * global_functions.CUSPc(0., e_p[
#         #                             f"limits_{t_e}"], e_p[f"limits_{t_e}"])))
#
#         new_params = copy.deepcopy(self.earth_params)
#         for key in new_params.keys():
#             if key.endswith('time'):
#                 new_params[key] /= self.earth_params['Amazonas_time']
#         return new_params
#
#
#     """
#     Here we insert a conversion factor to get a translation from a.u. to "true" years
#     """
#     def conversion(self):
#         cusp_deq = cusp(a=-1/self._timescale, b=1/self._timescale, c=self._c_krit/self._timescale)
#         net = tipping_network()
#         net.add_element(cusp_deq)
#         cusp_deq._par['c'] = self._c_krit/self._timescale + self._epsilon_c/self._timescale
#
#         # Define event function to detect threshold crossing
#         def threshold_event(t, y):
#             """Event occurs when state crosses threshold"""
#             return y[0] - self._threshold
#
#         threshold_event.terminal = True  # Stop integration when event occurs
#         threshold_event.direction = 1  # Only detect crossing from below
#
#         t_end = 5000
#         sol = solve_ivp(
#             net.f,
#             (0, t_end),
#             self._initial_state,
#             events=threshold_event,
#             jac=net.jac,
#             method='LSODA',
#             dense_output=True  # Enable if you need solution interpolation
#         )
#
#         # Get time when threshold was crossed
#         if len(sol.t_events[0]) > 0:
#             t_cross = sol.t_events[0][0]
#         else:
#             raise ValueError("Threshold was not crossed within simulation time")
#
#         # conversion factor from arbitrary units to years
#         conv_fac = self._real_timescale / t_cross
#         return conv_fac




