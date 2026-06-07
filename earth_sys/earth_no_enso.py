# Add modules directory to path
import sys

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

calibration_df = pd.read_csv("interaction_calibration.csv", header=[0, 1], index_col=0)

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
    net = tipping_network()
    nodes = {}
    for t_e in ["gis", "thc", "wais", "amaz", "reef", "awsi", "perm", "wam"]:
        nodes[t_e] = t_cusp(a=-1.0 / e_p[f"{t_e}_time"], b=1.0 / e_p[f"{t_e}_time"],
                      c=lambda t: (1.0 / e_p[f"{t_e}_time"]) * global_functions.CUSPc(0., e_p[f"limits_{t_e}"], temp(t)))
    nodes["nino"] = linear(a=-1 / e_p['nino_time'],
                  c=lambda t: (1.0 / e_p['nino_time']) * global_functions.CUSPc(0., e_p['limits_nino'], temp(t)), x_0=-1.0)
    for node in nodes.values():
        net.add_element(node)
    # Dicts preserve order since 3.7
    node_dict = {"GIS":0, "AMOC":1, "WAIS":2, "Amazonas":3, "REEF":4, "AWSI":5, "PERM":6, "WAM":7, "NINO":8}
    ######################################Set edges to active state#####################################
    net.add_coupling(1, 0, linear_coupling(strength=(1.0 / e_p['gis_time']) * strength * pf_to_interaction(e_p, 'pf_thc_to_gis'), x_0=-1))
    net.add_coupling(2, 0, linear_coupling(strength=(1.0 / e_p['gis_time']) * strength * pf_to_interaction(e_p, 'pf_wais_to_gis'), x_0=-1))

    # derivative coupling has pretty substantial impact on performance (20% more)
    net.add_coupling(0, 1, cusp_derivative_coupling(strength=(1.0 / e_p['thc_time']) * e_p['gis_time'] * strength *
                                                             pf_to_interaction(e_p, 'pf_gis_to_thc'), params=nodes["gis"].get_par()))
    net.add_coupling(2, 1, cusp_derivative_coupling(strength=(1.0 / e_p['thc_time']) * e_p['wais_time'] * strength *
                                                             pf_to_interaction(e_p, 'pf_wais_to_thc'), params=nodes["wais"].get_par()))
    net.add_coupling(8, 1, linear_coupling(strength=(1.0 / e_p['thc_time']) * strength * pf_to_interaction(e_p, 'pf_nino_to_thc') * kk1, x_0=-1))
    # net.add_coupling(5, 1, linear_coupling(strength=(1.0 / earth_params.thc_time) * strength * earth_params.pf_assi_to_thc, x_0=-1))

    net.add_coupling(0, 2, linear_coupling(strength=(1.0 / e_p['wais_time']) * strength * pf_to_interaction(e_p, 'pf_gis_to_wais'), x_0=-1))
    net.add_coupling(1, 2, linear_coupling(strength=(1.0 / e_p['wais_time']) * strength * pf_to_interaction(e_p, 'pf_thc_to_wais'), x_0=-1))
    net.add_coupling(8, 2, linear_coupling(strength=(1.0 / e_p['wais_time']) * strength * pf_to_interaction(e_p, 'pf_nino_to_wais'), x_0=-1))

    net.add_coupling(1, 3, linear_coupling(strength=(1.0 / e_p['amaz_time']) * strength * pf_to_interaction(e_p, 'pf_thc_to_amaz') * kk2, x_0=-1))
    net.add_coupling(8, 3, linear_coupling(strength=(1.0 / e_p['amaz_time']) * strength * pf_to_interaction(e_p, 'pf_nino_to_amaz'), x_0=-1))
    # nino doesnt tip, so I use Nicos dimensional-analysis based approach (reconfigured so that it matches the new strength of the calibrated interactions)
    net.add_coupling(2, 8, linear_coupling(strength=(1.0 / e_p['nino_time']) * strength * e_p['pf_thc_to_nino'], x_0=-1))

    # net.add_coupling(3, 4, linear_coupling(strength=(1.0 / params.nino_time) * strength * params.pf_amaz_to_nino * kk1, x_0=-1)) # doesnt appear in GTP2025
    # Bara Couplings
    net.add_coupling(5, 1, linear_coupling(strength=(1.0 / e_p['thc_time']) * strength * pf_to_interaction(e_p, 'pf_awsi_to_thc'), x_0=-1))
    net.add_coupling(5, 0, linear_coupling(strength=(1.0 / e_p['gis_time']) * strength * pf_to_interaction(e_p, 'pf_awsi_to_gis'), x_0=-1))
    net.add_coupling(5, 6, linear_coupling(strength=(1.0 / e_p['perm_time']) * strength * pf_to_interaction(e_p, 'pf_awsi_to_perm'), x_0=-1))
    net.add_coupling(1, 5, linear_coupling(strength=(1.0 / e_p['awsi_time']) * strength * pf_to_interaction(e_p, 'pf_thc_to_awsi'), x_0=-1))
    net.add_coupling(1, 7, linear_coupling(strength=(1.0 / e_p['wam_time']) * strength * pf_to_interaction(e_p, 'pf_thc_to_wam'), x_0=-1))
    net.add_coupling(8, 4, linear_coupling(strength=(1.0 / e_p['reef_time']) * strength * pf_to_interaction(e_p, 'pf_nino_to_reef'), x_0=-1))
    net.add_coupling(6, 1, linear_coupling(strength=(1.0 / e_p['thc_time']) * strength * pf_to_interaction(e_p, 'pf_perm_to_thc'), x_0=-1))
    return net, node_dict
