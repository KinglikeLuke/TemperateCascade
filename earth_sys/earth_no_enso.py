# Add modules directory to path
import sys
from typing import Any
from numbers import Real
import timeit

sys.path.append('')

# global imports
import numpy as np
import pandas as pd
# private imports from sys.path
from core.coupling import linear_coupling, cusp_derivative_coupling
from core.tipping_element import t_cusp, linear, derivative_intervention, state_intervention
from core.tipping_network import tipping_network
from earth_sys.functions_earth_system_no_enso import global_functions

"""
Here the Earth system network is defined after Kriegler et al., 2009
"""

    # # compute df_gis_to_thc
    # if params['pf_gis_to_thc'] > 1:
    #     params['df_gis_to_thc'] = (params['pf_gis_to_thc'] - 1) * 0.35/9 * params['gis_time']
    # else:
    #     params['df_gis_to_thc'] = params['pf_gis_to_thc'] * params['gis_time']
    # # df_wais_to_thc
    # if params['pf_gis_to_thc'] > 1:
    #     params['df_wais_to_thc'] = (params['pf_wais_to_thc'] - 1) * 0.15/2 * params['wais_time']
    # else:
    #     params['df_wais_to_thc'] = 0.7109571 * params['pf_wais_to_thc'] + 0.94770077
    # return params
global calibration_df
COMPONENTS = ["GIS", "AMOC", "WAIS", "Amazonas", "REEF", "AWSI", "PERM", "WAM", "NINO"]

def pf_to_interaction(earth_params, name):
    """Taken from the calibration curve

    Args:
        pf (_type_): _description_
        negative_influence (_type_): _description_
        positive_influence (_type_): _description_
    """
    columns = calibration_df[name[3:]].dropna()
    return np.interp(earth_params[name], columns["pf"], columns["interaction_fac"])

def intervene_in_network(net, intervention_element, intervention_state, node_dict):
    """
    Performs a do-intervention on the network in-place, changing the type of the specified element and removing all
    incoming links
    Args:
        net: tipping network
        intervention_element: name of the element to be intervened
        intervention_state: into which state it should be put. -1 for untipped, 1 for tipped, 0 for no intervention
        node_dict: assigns indices to element names

    Returns:
        net, initial_state. Net is changed in-place anyway!
    """
    initial_state = -1 * np.ones(len(net.nodes))  # initial state
    if not intervention_state:
        return net, initial_state
    if intervention_element in ["WAIS", "GIS"] and intervention_state == 1:
        intervention_node = derivative_intervention(**net.nodes[node_dict[intervention_element]]['data'].get_par())
    else:
        intervention_node = state_intervention()
        initial_state[node_dict[intervention_element]] = intervention_state
    net.update_element(node_dict[intervention_element], intervention_node)
    net.remove_edges_from(list(net.in_edges(node_dict[intervention_element]))) # remove incoming edges to fix element evolution
    return net, initial_state

def earth_network(e_p: dict, temp, strength, kk0, kk1, kk2):
    """Create the Earth system tipping network.

    e_p: dict, earth_params containing keys used below (accessed directly via params[...] ).
    temperature: callable t -> temperature
    strength: coupling strength scalar
    kk0, kk1, kk2: integers -1, 0, or +1 controlling optional couplings
    """
    # set up network
    net, node_dict, nodes = earth_elements(e_p, temp)
    global calibration_df
    calibration_df = pd.read_csv(r"calibrations\interaction_calibration.csv", header=[0, 1], index_col=0)

    interactions = [(key.split("_")[1], key.split("_")[3]) for key in e_p.keys() if key.startswith("pf")]
    # I should have dones this a long time ago
    ######################################Set edges to active state#####################################
    for cause, effect in interactions:
        if not (cause in nodes and effect in nodes):
            continue
        if cause in ["WAIS", "GIS"] and effect == "AMOC":
            net.add_coupling(0, 1,
                             cusp_derivative_coupling(strength=(e_p[f'{cause}_time'] / e_p[f'{effect}_time'])*strength*
                                                               pf_to_interaction(e_p, f'pf_{cause}_to_{effect}'),
                                                      params=nodes[cause].get_par()))
        else:
            net.add_coupling(node_dict[cause], node_dict[effect],
                             linear_coupling(strength=(1.0 / e_p[f'{effect}_time']) * strength *
                                                      pf_to_interaction(e_p, f'pf_{cause}_to_{effect}'), x_0=-1))
    return net, node_dict

def earth_elements(e_p: dict, temp) -> tuple[tipping_network, dict[Any, Any], dict[str, t_cusp]]:
    net = tipping_network()
    if isinstance(temp, Real):
        temp = lambda t, tem=temp: tem
    nodes = {}
    for t_e in ["GIS", "AMOC", "WAIS", "Amazonas", "REEF", "WAM"]:
        # Fucking lambda only saves the reference to t_e, which would then take the last value of t_e, "WAM", for
        # every node. Hence, I have to pass the t_e variable explicitly in the default values.
        nodes[t_e] = t_cusp(a=-1.0 / e_p[f"{t_e}_time"], b=1.0 / e_p[f"{t_e}_time"],
                            c=global_functions.make_CUSPc(e_p[f"{t_e}_time"], 0, e_p[f"limits_{t_e}"], temp))

    for t_e in ["AWSI", "PERM", "NINO"]:
        nodes[t_e] = t_cusp(a=-1.0 / e_p[f"{t_e}_time"], b=-1.0 / e_p[f"{t_e}_time"],
                            c=global_functions.make_CUSPc(e_p[f"{t_e}_time"], e_p[f"limits_{t_e}"], 0, temp, y2=-2))
    for node in nodes.values():
        net.add_element(node)
    # Dicts preserve order since 3.7
    node_dict = {component: i for i, component in enumerate(nodes.keys())}
    return net, node_dict, nodes
