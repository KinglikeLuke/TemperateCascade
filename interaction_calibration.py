import sys
# global imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from dataclasses import dataclass

import matplotlib.pyplot as plt

# PyCascades imports
from core.coupling import linear_coupling, cusp_derivative_coupling
from core.tipping_element import cusp, linear, state_intervention, derivative_intervention, tipping_element
from core.tipping_network import tipping_network
from earth_sys.functions_earth_system_no_enso import global_functions

from earth_sys.earth_no_enso import EarthParams

KEYS = ['limits_gis','limits_thc','limits_wais','limits_amaz','limits_nino', 'limits_assi',
    'pf_wais_to_gis','pf_thc_to_gis',
    'pf_gis_to_thc','pf_nino_to_thc','pf_wais_to_thc', 'pf_assi_to_thc',
    'pf_nino_to_wais','pf_thc_to_wais','pf_gis_to_wais',
    'pf_thc_to_nino',
    'pf_nino_to_amaz', 'pf_thc_to_amaz',
    'pf_thc_to_assi',
    'gis_time','thc_time','wais_time','nino_time','amaz_time', 'assi_time']
input_file = np.loadtxt(r"start_ensemble\latin_prob.txt", delimiter=" ")
temperature = 2

def hyperbolic_fct(x, a, b, c):
    return a/(x+b) + c

def linear_fct(x, a, b):
    return a*x + b

def sa(f):
    """
    Switch arguments to account for solve_ivp vs odeint sensibilities
    
    :param f: Description
    """
    return lambda t, y : f(y, t)

def compare_with_intervention(cause:tipping_element, effect:tipping_element, coupling_strength):
    net = tipping_network()
    net.add_element(effect)
    base_net = net.copy()
    base_net.add_element(cause)
    intervention_net = net.copy()
    intervention_cause = derivative_intervention(**cause.get_par())
    intervention_net.add_element(intervention_cause)
    # Note: get_par() needs to be amended to make copy
    # base_net.add_coupling(1, 0, cusp_derivative_coupling(strength=coupling_strength, 
    #                                             params=cause.get_par(), 
    #                                             ))
    intervention_net.add_coupling(1, 0, cusp_derivative_coupling(strength=coupling_strength, 
                                                        params=intervention_cause.get_par(),
                                                        ))
    base_state = [-1, -1]
    intervention_state = [-1, -1]
    t = (0, 50000)

    # gotta rem
    def cause_tip(t, y): return y[1]
    def effect_tip(t, y): return y[0] 
    base_sol = solve_ivp(sa(base_net.f), t, base_state, jac = sa(base_net.jac), method='LSODA', events=[effect_tip, cause_tip]) # idk if the equation is actually stiff
    tip_effect = base_net.get_tip_states(base_sol.y[:,-1])[0] # solve_ivp returns variable x time, unlike odeint which returns time x variable
    tip_cause = base_net.get_tip_states(base_sol.y[:,-1])[1]
    if base_net.get_number_tipped(base_sol.y[:,-1]) == 2:
        effect_after_cause = base_sol.t_events[0][0] >= base_sol.t_events[1][0] # if effect tips after cause first tipped
    else:
        effect_after_cause = 0
    intervention_sol = solve_ivp(sa(intervention_net.f), t, intervention_state, jac=sa(intervention_net.jac), method='LSODA') # idk if the equation is actually stiff
    tip_intervention = intervention_net.get_tip_states(intervention_sol.y[:,-1])[0]
    # P(A') = P(B|A') + P(A and not B) 
    # This produces a fun correlation: because B tips faster than A, there is the correlation A tipped first -> B probably won't tip
    # at all (because we rolled a high/low threshold for B/A). Sort of an inverse inference about GMT (A tips first -> GMT lower than B threshold)
    return tip_effect, tip_cause, effect_after_cause, tip_intervention

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
    base_net.add_element(cause)
    base_net.add_coupling(1, 0, linear_coupling(strength=coupling_strength, x_0=-1))

    t = np.linspace(0, 500000, 2)

    base_state = [-1, -1]
    def cause_tip(t, y): return y[1]
    def effect_tip(t, y): return y[0] 
    base_sol = solve_ivp(sa(base_net.f), t, base_state, jac = sa(base_net.jac), method='LSODA', events=[effect_tip, cause_tip]) # idk if the equation is actually stiff
    tip_effect = base_net.get_tip_states(base_sol.y[:,-1])[0] # solve_ivp returns variable x time, unlike odeint which returns time x variable
    return tip_effect

n_interaction_strengths = 100
n_effect = np.zeros(n_interaction_strengths)
n_intervention = np.zeros(n_interaction_strengths)
summary_ate = np.zeros(n_interaction_strengths)
n_cause_first = np.zeros(n_interaction_strengths)
n_effect_after_cause = np.zeros(n_interaction_strengths)
interaction_facs = np.linspace(-1, 1., n_interaction_strengths)

for params in input_file:
    values = list(map(float, params)) # -1 is the mc_dir
    params_dict = dict(zip(KEYS, values))
    earth_params = EarthParams(**params_dict)
    gis = cusp(a=-1.0 / earth_params.gis_time, b=1.0 / earth_params.gis_time,
                    c=(1.0 / earth_params.gis_time) * global_functions.CUSPc(0., earth_params.limits_gis, temperature))
    thc = cusp(a=-1.0 / earth_params.thc_time, b=1.0 / earth_params.thc_time,
                     c=(1.0 / earth_params.thc_time) * global_functions.CUSPc(0., earth_params.limits_thc, temperature))
    wais = cusp(a=-1.0 / earth_params.wais_time, b=1.0 / earth_params.wais_time,
                    c=(1.0 / earth_params.wais_time) * global_functions.CUSPc(0., earth_params.limits_wais, temperature))
    # amaz = cusp(a=-1.0 / earth_params.amaz_time, b=1.0 / earth_params.amaz_time,
    #                 c=(1.0 / earth_params.amaz_time) * global_functions.CUSPc(0., earth_params.limits_amaz, temperature))
    # nino = linear(a=-1 / earth_params.nino_time, c=(1.0 / earth_params.nino_time) * global_functions.CUSPc(0., earth_params.limits_nino, temperature), x_0=-1.0)
    # assi = linear(a=-1 / earth_params.assi_time, c=(1.0 / earth_params.assi_time) * global_functions.CUSPc(0., earth_params.limits_assi, temperature), x_0=-1.0)
    for i, interaction_fac in enumerate(interaction_facs):
        interaction_strength = interaction_fac / earth_params.thc_time * earth_params.wais_time
        tip_effect, tip_cause, effect_after_cause, tip_intervention = compare_with_intervention(wais, thc, interaction_strength)
        n_effect[i] += tip_effect
        n_intervention[i] += tip_intervention
        # n_effect[i] += free_run(wais, gis, interaction_strength)
        # summary_ate[i] += average_treatment_effect(gis, interaction_strength)
        n_cause_first[i] += (effect_after_cause or tip_cause and not tip_effect)
        n_effect_after_cause[i] += effect_after_cause
# This falls apart due to the wonky question design: the temperature dependence of both leads to correlations that were excluded by the questionnaire
# To wit: if you tell me that the WAIS melts before the GIS, I know that temperatures must be too low for GIS to melt
n_trials = input_file.shape[0]
# pf = ((n_effect_after_cause/n_cause_first)/(n_effect/n_trials))
# plt.plot(interaction_facs, n_trials/n_effect, label="1/P(B)", color="tab:orange")
# plt.axhline(1, color="tab:orange")
pf = n_intervention/n_effect
p_intervention =  n_intervention/n_trials
odds = p_intervention/(1-p_intervention) # oddly enough, *inverse* odds -- odds against -- produce a linear relationship (until they hit 0)
plt.plot(interaction_facs, 1/odds, label="Probability Factor")
# fit, _ = curve_fit(linear_fct, interaction_facs, pf)
# print(fit)
# plt.plot(interaction_facs, linear_fct(interaction_facs, *fit))
# plt.plot(interaction_facs, pf)
plt.legend()
plt.xlabel(r"Derivative coupling strength / $\tau_\mathrm{cause}$")
plt.show()
