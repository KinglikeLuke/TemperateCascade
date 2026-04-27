import sys
import json
from typing import Any

# global imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.interpolate import interpolate
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from overshoot_trajectory import fit_parameters, overshoot_trajectory
from pydoe import lhs

# PyCascades imports
from core.coupling import linear_coupling, cusp_derivative_coupling
from core.tipping_element import t_cusp, linear, state_intervention, derivative_intervention, tipping_element
from core.tipping_network import tipping_network
from earth_sys.functions_earth_system_no_enso import global_functions


KEYS = ['limits_gis','limits_thc','limits_wais','limits_amaz','limits_nino',
    'pf_wais_to_gis','pf_thc_to_gis',
    'pf_gis_to_thc','pf_nino_to_thc','pf_wais_to_thc',
    'pf_nino_to_wais','pf_thc_to_wais','pf_gis_to_wais',
    'pf_thc_to_nino',
    'pf_nino_to_amaz', 'pf_thc_to_amaz',
    'gis_time','thc_time','wais_time','nino_time','amaz_time']
input_file = np.loadtxt(r"start_ensemble\latin_prob_calibration.txt", delimiter=" ")
limit_filename = r"start_ensemble\limits.json"
with open(limit_filename, "r") as file:
    LIMITS = json.load(file)

def hyperbolic_fct(x, a, b, c):
    return a/(x+b) + c

def linear_fct(x, a):
    return a*x + 1

def sa(f):
    """
    Switch arguments to account for solve_ivp vs odeint sensibilities
    
    :param f: Description
    """
    return lambda t, y : f(y, t)

def intervention_effect(cause:tipping_element, effect:tipping_element, coupling_strength, derivative):
    intervention_net = tipping_network()
    intervention_net.add_element(effect)
    intervention_cause = derivative_intervention(**cause.get_par()) if derivative else state_intervention()
    intervention_net.add_element(intervention_cause)
    # implementation TODO: get_par() needs to be amended to make copy
    if derivative:
        intervention_net.add_coupling(1, 0, cusp_derivative_coupling(strength=coupling_strength, 
                                                            params=intervention_cause.get_par()))
    else:
        intervention_net.add_coupling(1, 0, linear_coupling(strength=coupling_strength))

    intervention_state = [-1, -1] if derivative else [-1, 1]
    t = (0, 10000)

    # idk if the equation is actually stiff, but lsoda with jac is an order of magnitude faster than solve_ivp
    intervention_sol = solve_ivp(intervention_net.f, t, intervention_state, jac=intervention_net.jac, method='LSODA', events=lambda t, x: x[0])
    tip_intervention = len(intervention_sol.t_events[0]) > 0 # if the effect ever tipped, it gets registered
    # P(A') = P(B|A') + P(A and not B) 
    # This produces a fun correlation: because B tips faster than A, there is the correlation A tipped first -> B probably won't tip
    # at all (because we rolled a high/low threshold for B/A). Sort of an inverse inference about GMT (A tips first -> GMT lower than B threshold)
    return tip_intervention

def average_treatment_effect(effect:tipping_element, coupling_strength):
    """
    Increasingly dubious interpretation of PF: PF=(1+ATE)/P(B)
    
    :param effect: Description
    :type effect: tipping_element
    :param copuling_strength: Description
    """
    intervention_net = tipping_network()
    intervention_net.add_element(effect)
    intervention_net.add_element(state_intervention())
    intervention_net.add_coupling(1, 0, linear_coupling(strength=coupling_strength, x_0=-1))
    t = np.linspace(0, 1e5, 2)
    treatment_sol = solve_ivp(sa(intervention_net.f), t, [-1, 1], jac=sa(intervention_net.jac), method='LSODA')
    control_sol = solve_ivp(sa(intervention_net.f), t, [-1, -1], jac=sa(intervention_net.jac), method='LSODA')
    ate = np.int16(intervention_net.get_tip_states(treatment_sol.y[:,-1])[0]) - np.int16(intervention_net.get_tip_states(control_sol.y[:,-1])[0])
    return ate

def free_run(cause, effect, coupling_strength):
    """
    Does the effect tip in a free run at this temperature? This is supposed to be P(B), but dubious, notably because P(B) depends on P(B|A'), so it cannot be a
    real norm.
    Consider a tipping element connected to another that tip at a comparable temperature. Then the pf decreases for stronger coupling at temperatures that sometimes allow 
    tipping of both elements, because the marginal probability increases, while the treatment probability is capped by one before it is guaranteed that B tips in a free
    simulation. This is a fundamental incompatibility between linear couplings and PF and also an enormous weakness of PF in themselves because of their very counter-
    intuitive behavior

    
    :param cause: Description
    :param effect: Description
    :param coupling_strength: Description
    """
    base_net = tipping_network()
    base_net.add_element(effect)
    if coupling_strength:
        base_net.add_element(cause)
        base_net.add_coupling(1, 0, linear_coupling(strength=coupling_strength, x_0=-1))

    t = (0, 50000) # maybe because this was 500k instead of 50k, this produced different results? Would be bad

    base_state = [-1, -1] if coupling_strength else [-1]
    base_sol = solve_ivp(base_net.f, t, base_state, jac = base_net.jac, method='LSODA', events=lambda t, x: x[0]) # idk if the equation is actually stiff
    tip_effect = len(base_sol.t_events[0]) > 0 # solve_ivp returns variable x time, unlike odeint which returns time x variable
    return tip_effect

def force_strict_mono(array):
    """Marks all not strictly monotonically increasing elements in an array. Always keeps the first one. Does therefore tend to a long left tail 

    Args:
        array (_type_): _description_

    Returns:
        _type_: _description_
    """
    monotonic = np.ones(array.shape, dtype = bool)
    prev = -np.inf
    for i in range(len(array)):
        if array[i] <= prev:
            monotonic[i] = False
        else:
            prev = array[i]
    return monotonic

def calibrate_interaction(cause:str, effect:str, derivative:bool, n_interaction_strengths:int=100):
    limit = LIMITS[f"pf_{cause}_to_{effect}"]
    interaction_limit = [0, 1]
    pf = []
    total_isolated_tips = 0
    if limit[0] < 1:
        interaction_limit[0] = -1
        if limit[1] <= 1:
            interaction_limit[1] = 0
        else:
            # if I have to scan both directions I need more granularity
            n_interaction_strengths = 200

    temperatures = []
    T_0 = 1
    lhc_distr = np.array(lhs(3, samples=20))
    T_peaks = np.round(4 * lhc_distr[:, 0] + 2, 2)
    T_lims = np.round(2 * lhc_distr[:, 1], 2)
    t_convs = np.round(900 * lhc_distr[:, 2] + 100, 0)
    for T_peak, T_lim, t_conv in zip(T_peaks, T_lims, t_convs):
        R, mu_0, mu_1 = fit_parameters(T_0, T_peak, T_lim, t_conv)
        # fun little lambda behavior again
        temperatures.append(lambda t, tlim = T_lim, r=R, mu0=mu_0, mu1=mu_1: overshoot_trajectory(t, T_0, tlim, r, mu0, mu1))

    for temperature in temperatures:
        for params in input_file:
            earth_params, cause_element, effect_element = initialize_elements(cause, effect, temperature, params)
            tip_isolated = intervention_effect(cause_element, effect_element, 0, derivative)
            total_isolated_tips += tip_isolated

    interaction_facs = np.linspace(*interaction_limit, n_interaction_strengths)
    for i, interaction_fac in enumerate(interaction_facs):
        n_intervention = 0
        for temperature in temperatures:
            for params in input_file:
                earth_params, cause_element, effect_element = initialize_elements(cause, effect, temperature, params)
                interaction_strength = interaction_fac / earth_params[f"{effect}_time"]
                if derivative:
                    interaction_strength *= earth_params[f"{cause}_time"]
                tip_intervention = intervention_effect(cause_element, effect_element, interaction_strength, derivative)
                n_intervention += tip_intervention
        pf.append(n_intervention/total_isolated_tips)
        if n_intervention/total_isolated_tips > limit[1]:
            interaction_facs = interaction_facs[:i+1]
            break
    pf = np.array(pf)
    strictly_monotonic = force_strict_mono(pf)
    df = pd.DataFrame({"pf": pf[strictly_monotonic], "interaction_fac": interaction_facs[strictly_monotonic]})
    return df, f"{pair[0]}_to_{pair[1]}"


def initialize_elements(cause: str, effect: str, temperature, params) -> tuple[
    dict[Any, Any], t_cusp, t_cusp]:
    values = list(map(float, params))  # -1 is the mc_dir

    if len(KEYS) != len(values):
        raise KeyError("KEYS and LHS dont match!")
    earth_params = dict(zip(KEYS, values))
    cause_element = t_cusp(a=-1.0 / earth_params[f"{cause}_time"], b=1.0 / earth_params[f"{cause}_time"],
                         c=lambda t:(1.0 / earth_params[f"{cause}_time"]) * global_functions.CUSPc(0., earth_params[
                             f"limits_{cause}"], temperature(t)))
    effect_element = t_cusp(a=-1.0 / earth_params[f"{effect}_time"], b=1.0 / earth_params[f"{effect}_time"],
                          c=lambda t: (1.0 / earth_params[f"{effect}_time"]) * global_functions.CUSPc(0., earth_params[
                              f"limits_{effect}"], temperature(t)))
    return earth_params, cause_element, effect_element

pairs = [
        ["gis", "thc", True],
         ["wais", "thc", True],
         ["thc", "gis", False],
         ["wais", "gis", False],
         ["gis", "wais", False],
         ["thc", "wais", False],
         ["thc", "amaz", False],
        #  # I think these are fine (if nino=1 can be considered equivalent to "tipped")
         ["nino", "thc", False],
         ["nino", "wais", False],
         ["nino", "amaz", False],
         ]
results = {}

for pair in pairs:
    print(f"Currently calibrating: {pair[0]} to {pair[1]}")
    df, name = calibrate_interaction(*pair)
    results[name] = df

full_df = pd.concat(results, axis=1)
full_df.columns.names = ["component", "axis"]
full_df.to_csv("interaction_calibration.csv")
# TODO somehow, the pf of slightly positive interaction factors is smaller than one (???)
# for params in input_file:
#     values = list(map(float, params)) # -1 is the mc_dir
#     earth_params = dict(zip(KEYS, values))
#     gis = cusp(a=-1.0 / earth_params["gis_time"], b=1.0 / earth_params["gis_time"],
#                     c=(1.0 / earth_params["gis_time"]) * global_functions.CUSPc(0., earth_params["limits_gis"], temperature))
#     thc = cusp(a=-1.0 / earth_params["thc_time"], b=1.0 / earth_params["thc_time"],
#                      c=(1.0 / earth_params["thc_time"]) * global_functions.CUSPc(0., earth_params["limits_thc"], temperature))
#     wais = cusp(a=-1.0 / earth_params["wais_time"], b=1.0 / earth_params["wais_time"],
#                     c=(1.0 / earth_params["wais_time"]) * global_functions.CUSPc(0., earth_params["limits_wais"], temperature))
#     # amaz = cusp(a=-1.0 / earth_params.amaz_time, b=1.0 / earth_params.amaz_time,
#     #                 c=(1.0 / earth_params.amaz_time) * global_functions.CUSPc(0., earth_params.limits_amaz, temperature))
#     # nino = linear(a=-1 / earth_params.nino_time, c=(1.0 / earth_params.nino_time) * global_functions.CUSPc(0., earth_params.limits_nino, temperature), x_0=-1.0)
#     # assi = linear(a=-1 / earth_params.assi_time, c=(1.0 / earth_params.assi_time) * global_functions.CUSPc(0., earth_params.limits_assi, temperature), x_0=-1.0)
#     isolated_effect += free_run(wais, thc, 0)
#     for i, interaction_fac in enumerate(interaction_facs):
#         interaction_strength = interaction_fac / earth_params["thc_time"] * earth_params["wais_time"]
#         tip_intervention = intervention_effect(wais, thc, interaction_strength)
#         n_intervention[i] += tip_intervention
#         # n_effect[i] += free_run(wais, gis, interaction_strength)
# # This falls apart due to the wonky question design: the temperature dependence of both leads to correlations that were excluded by the questionnaire
# # To wit: if you tell me that the WAIS melts before the GIS, I know that temperatures must be too low for GIS to melt
# n_trials = input_file.shape[0]
# # pf = ((n_effect_after_cause/n_cause_first)/(n_effect/n_trials))
# # plt.plot(interaction_facs, n_trials/n_effect, label="1/P(B)", color="tab:orange")
# # plt.axhline(1, color="tab:orange")
# pf = n_intervention/isolated_effect
# p_intervention =  n_intervention/n_trials
# odds = p_intervention/(1-p_intervention) # oddly enough, *inverse* odds -- odds against -- produce a linear relationship (until they hit 0)
# plt.plot(interaction_facs, pf, label="1/Probability Factor")
# # because I fit the inverse max and min and directions are switched
# # right_linear_edge = np.where(pf < 0.9*np.max(pf))[0][-1]
# # left_linear_edge = np.where(pf > 1.1*np.min(pf))[0][0]
# # linear_area = np.zeros(pf.shape, dtype=bool)
# # linear_area[left_linear_edge:right_linear_edge] = 1

# pf_array = np.linspace(0.2, 12, 500)
# interaction_interpolation = np.interp(pf_array, pf[strictly_monotonic], interaction_facs[strictly_monotonic], left=np.nan, right=np.nan)
# plt.plot(interaction_interpolation, pf_array, label="interpolation")
# # assuming constant error in pf
# # fit, _ = curve_fit(linear_fct, interaction_facs[linear_area], 1/pf[linear_area], sigma=1/pf[linear_area]**2)
# # print(fit)
# # plt.plot(interaction_facs[linear_area], linear_fct(interaction_facs[linear_area], *fit))
# plt.legend()
# plt.xlabel(r"Derivative coupling strength / $\tau_\mathrm{cause}$")
# plt.show()
