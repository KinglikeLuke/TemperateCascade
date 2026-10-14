import copy
import json
from typing import Any

# global imports
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from sympy import false

from temp_input.overshoot_trajectory import fit_parameters, overshoot_trajectory
from pydoe import lhs
import tqdm

# PyCascades imports
from core.coupling import linear_coupling, cusp_derivative_coupling
from core.tipping_element import t_cusp, state_intervention, derivative_intervention, tipping_element
from core.tipping_network import tipping_network
from earth_sys.functions_earth_system_no_enso import global_functions
from earth_sys.timing_no_enso import individual_timescales
from earth_sys.earth_no_enso import earth_elements
# TODO ask llm to makes this less ass
from start_ensemble.latin_probability_distribution import update_limits
update_limits()

input_file = pd.read_csv(r"../start_ensemble/latin_prob_calibration.txt", delimiter=",")
limit_filename = r"../start_ensemble/limits.json"
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

def intervention_effect(cause:t_cusp, effect:tipping_element, coupling_strength, derivative):
    intervention_net = tipping_network()
    intervention_net.add_element(effect)
    if derivative:
        intervention_cause = copy.deepcopy(cause)
        intervention_cause.c.set_x(intervention_cause.c.get_x1()*2 if intervention_cause.c.get_x1()
                                   else intervention_cause.c.get_x2()*1.5)
        # intervention_cause = derivative_intervention(**cause.get_par())
    else:
        intervention_cause = state_intervention()
    intervention_net.add_element(intervention_cause)
    # implementation TODO: get_par() needs to be amended to make copy
    if derivative:
        intervention_net.add_coupling(1, 0, cusp_derivative_coupling(strength=coupling_strength, 
                                                            params=intervention_cause.get_par()))
    else:
        intervention_net.add_coupling(1, 0, linear_coupling(strength=coupling_strength))

    intervention_state = [-1, -1] if derivative else [-1, 1]
    t = (0, 10000)

    def threshold_event(t, y):
        return y[0]
    threshold_event.terminal = True
    threshold_event.direction = 1

    # idk if the equation is actually stiff, but lsoda with jac is an order of magnitude faster than solve_ivp
    intervention_sol = solve_ivp(intervention_net.f, t, intervention_state, jac=intervention_net.jac, method='LSODA', events=threshold_event)
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

def calibrate_interaction(cause:str, effect:str, derivative:bool, n_interaction_strengths:int=101):
    limit = LIMITS[f"pf_{cause}_to_{effect}"]
    interaction_limit = [0, 1]
    pf = []
    total_isolated_tips = 0
    if limit[0] < 1:
        interaction_limit = [0, -1]
    interaction_facs: list = [np.linspace(*interaction_limit, n_interaction_strengths)]
    if limit[0] < 1 < limit[1]:
        # if I have to scan both directions I go first toward -1, then toward 1. To break in both directions I need list
        interaction_facs.append(-interaction_facs[0][1:])

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

    for params in new_params.iterrows():
        earth_params = params[1].to_dict()
        for temperature in temperatures:
            cause_element, effect_element = initialize_elements(cause, effect, temperature, earth_params)
            tip_isolated = intervention_effect(cause_element, effect_element, 0, derivative)
            total_isolated_tips += tip_isolated

    for n_run in range(len(interaction_facs)):
        for j, interaction_fac in enumerate(tqdm.tqdm(interaction_facs[n_run])):
            n_intervention = 0
            for params in new_params.iterrows():
                earth_params = params[1].to_dict()
                for temperature in temperatures:
                    cause_element, effect_element = initialize_elements(cause, effect, temperature, earth_params)
                    interaction_strength = interaction_fac / earth_params[f"{effect}_time"]
                    if derivative:
                        interaction_strength *= earth_params[f"{cause}_time"]
                    tip_intervention = intervention_effect(cause_element, effect_element, interaction_strength, derivative)
                    n_intervention += tip_intervention
            new_pf = n_intervention/total_isolated_tips
            pf.append(new_pf)
            if new_pf > limit[1] or new_pf < limit[0] or np.isclose(n_intervention/(len(temperatures)*new_params.shape[0]), 1, 0.05):
                interaction_facs[n_run] = interaction_facs[n_run][:j+1]
                break
    # put results back in the right order
    interaction_facs = np.concatenate(interaction_facs)
    pf = np.array(pf)[np.argsort(interaction_facs)]
    interaction_facs = np.sort(interaction_facs)
    strictly_monotonic = force_strict_mono(pf)
    df = pd.DataFrame({"pf": pf[strictly_monotonic], "interaction_fac": interaction_facs[strictly_monotonic]})
    return df, f"{cause}_to_{effect}"


def initialize_elements(cause: str, effect: str, temperature, earth_params) -> tuple[t_cusp, t_cusp]:
    net, node_dict, nodes = earth_elements(earth_params, temperature)
    cause_element = nodes[cause]
    effect_element = nodes[effect]
    return cause_element, effect_element

# Generate pairs automatically from pf_ entries in limits.json
pairs = []
for key in LIMITS.keys():
    if key.startswith("pf_"):
        # Extract cause and effect from "pf_cause_to_effect"
        parts = key[3:].split("_to_")
        if len(parts) == 2:
            cause, effect = parts
            derivative = (cause in ["WAIS", "GIS"] and effect == "AMOC")
            pairs.append([cause, effect, derivative])

results = {}
new_params = input_file.copy()
for i, params in enumerate(input_file.iterrows()):
    earth_params = params[1].to_dict().copy()
    new_params.loc[i] = individual_timescales(earth_params)
for pair in pairs:
    print(f"Currently calibrating: {pair[0]} to {pair[1]}")
    df, name = calibrate_interaction(*pair)
    results[name] = df

full_df = pd.concat(results, axis=1)
full_df.columns.names = ["component", "axis"]

full_df.to_csv("interaction_calibration.csv")
# TODO If only a single value gets accepted the calibration completely blows up
# TODO stabilizing effects on bistable elements are far too extreme

