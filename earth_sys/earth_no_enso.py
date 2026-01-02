# Add modules directory to path
import sys

sys.path.append('')

# global imports
import numpy as np
from dataclasses import dataclass
# private imports from sys.path
from core.coupling import linear_coupling
from core.tipping_element import t_cusp
from core.tipping_network import tipping_network
from earth_sys.functions_earth_system_no_enso import global_functions

"""
Here the Earth system network is defined after Kriegler et al., 2009
"""


@dataclass
class EarthParams:
    gis_time: float
    thc_time: float
    wais_time: float
    nino_time: float
    amaz_time: float

    limits_gis: float
    limits_thc: float
    limits_wais: float
    limits_nino: float
    limits_amaz: float

    pf_wais_to_gis: float
    pf_thc_to_gis: float
    pf_gis_to_thc: float
    pf_nino_to_thc: float
    pf_wais_to_thc: float
    pf_gis_to_wais: float
    pf_thc_to_wais: float
    pf_nino_to_wais: float
    pf_thc_to_nino: float
    pf_amaz_to_nino: float
    pf_nino_to_amaz: float
    pf_thc_to_amaz: float

def earth_network(params: EarthParams, temperature, strength, kk0, kk1, kk2):
    """Create the Earth system tipping network.

    params: dict containing keys used below (accessed directly via params[...] ).
    temperature: callable t -> temperature
    strength: coupling strength scalar
    kk0, kk1, kk2: integers -1, 0, or +1 controlling optional couplings
    """
    gis = t_cusp(a=-1.0 / params.gis_time, b=1.0 / params.gis_time,
                 c=lambda t: (1.0 / params.gis_time) * global_functions.CUSPc(0., params.limits_gis, temperature(t)))
    thc = t_cusp(a=-1.0 / params.thc_time, b=1.0 / params.thc_time,
                 c=lambda t: (1.0 / params.thc_time) * global_functions.CUSPc(0., params.limits_thc, temperature(t)))
    wais = t_cusp(a=-1.0 / params.wais_time, b=1.0 / params.wais_time,
                 c=lambda t: (1.0 / params.wais_time) * global_functions.CUSPc(0., params.limits_wais, temperature(t)))
    amaz = t_cusp(a=-1.0 / params.amaz_time, b=1.0 / params.amaz_time,
                 c=lambda t: (1.0 / params.amaz_time) * global_functions.CUSPc(0., params.limits_amaz, temperature(t)))

    # set up network
    net = tipping_network()
    net.add_element(gis)
    net.add_element(thc)
    net.add_element(wais)
    net.add_element(amaz)

    ######################################Set edges to active state#####################################
    net.add_coupling(1, 0, linear_coupling(strength=-(1.0 / params.gis_time) * strength * params.pf_thc_to_gis, x_0=-1))
    net.add_coupling(2, 0, linear_coupling(strength=(1.0 / params.gis_time) * strength * params.pf_wais_to_gis, x_0=-1))

    net.add_coupling(0, 1, linear_coupling(strength=(1.0 / params.thc_time) * strength * params.pf_gis_to_thc, x_0=-1))
    net.add_coupling(2, 1, linear_coupling(strength=(1.0 / params.thc_time) * strength * params.pf_wais_to_thc * kk0, x_0=-1))

    net.add_coupling(0, 2, linear_coupling(strength=(1.0 / params.wais_time) * strength * params.pf_gis_to_wais, x_0=-1))
    net.add_coupling(1, 2, linear_coupling(strength=(1.0 / params.wais_time) * strength * params.pf_thc_to_wais, x_0=-1))

    net.add_coupling(1, 3, linear_coupling(strength=(1.0 / params.amaz_time) * strength * params.pf_thc_to_amaz * kk2, x_0=-1))

    return net
