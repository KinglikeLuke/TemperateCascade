"""
Timing calibration module: This module calibrates the c parameter for each tipping element
to match the minimum and maximum tipping times specified in limits.json
"""
import sys

from colorlog import exception

sys.path.append('')
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from core.tipping_element import cusp, t_cusp
from core.tipping_network import tipping_network
from earth_sys.functions_earth_system_no_enso import global_functions


class timing_calibration:

    def __init__(self, limits_file='../start_ensemble/limits.json'):
        """Initialize calibration with limits data"""
        with open(limits_file, 'r') as f:
            self.limits = json.load(f)

        self.elements = ["GIS", "AMOC", "WAIS", "Amazonas", "REEF", "AWSI", "PERM", "WAM"]
        self.initial_state = [-1.0]
        self.threshold = 1.0

    def simulate_tipping_time(self, temp_offset, limit_temp, timescale):
        """
        Simulate a single element and return tipping time

        Args:
            c_param: The c parameter value to test
            temp_offset: Temperature above limit (1C or 3C)
            limit_temp: The tipping point temperature
            timescale: The base timescale for the element

        Returns:
            Time taken to tip from -1 to 1, or inf if no tipping
        """
        # Calculate c value using CUSP function
        GMT = limit_temp + temp_offset
        c_krit = np.sqrt(4 / 27)

        # Create single element network
        net = tipping_network()
        element = cusp(a=-1.0 / timescale, b=1.0 / timescale, c=(global_functions.CUSPc(0., limit_temp, GMT) / timescale))
        net.add_element(element)

        # Simulate
        t_end = 1000000  # Long enough to capture tipping

        sol = solve_ivp(net.f, (0, t_end), self.initial_state,
                        jac=net.jac, method='LSODA', events=[lambda t, x, i=i: x[i]-1 for i in range(len(net.nodes))])

        if len(sol.t_events[0]) > 0:
            return sol.t_events[0][0]
        else:
            raise Exception("No tipping time found")

    def calibrate_element(self, limit_temp, target_time, temp_offset):
        """
        Find c parameter that matches target tipping time

        Args:
            limit_temp: appropriate limiting temperature
            target_time: Target tipping time to match
            temp_offset: Temperature offset (1C or 3C)

        Returns:
            Optimal c parameter value
        """

        def objective(timescale):
            sim_time = self.simulate_tipping_time(temp_offset, limit_temp, timescale)
            return abs(sim_time - target_time)

        # Search for optimal c parameter
        result = minimize_scalar(objective, bounds=(1, 10000  ), method='bounded')
        return result.x

    def calibrate_all(self):
        """
        Calibrate all elements for min and max tipping times

        Returns:
            Dictionary with calibrated c parameters for each element
        """
        results = {}

        for element in self.elements:
            print(f"Calibrating {element}...")

            # Get parameters from limits
            time_key = f"{element}_time"

            if time_key in self.limits:
                min_time = self.limits[time_key][0]
                max_time = self.limits[time_key][1]
                min_limit = self.limits[f"limits_{element}"][0]
                max_limit = self.limits[f"limits_{element}"][1]
                # Calibrate for minimum time (3C over limit)
                c_min = self.calibrate_element(min_limit, min_time, 1.0)

                # Calibrate for maximum time (1C over limit)
                c_max = self.calibrate_element(max_limit, min_time, 3.0)

                results[element] = {
                    'c_min': c_min,
                    'c_max': c_max,
                    'min_time': min_time,
                    'max_time': max_time
                }

                print(f"  c_min (3C offset): {c_min:.6f}")
                print(f"  c_max (1C offset): {c_max:.6f}")
            else:
                print(f"  Skipping {element} - missing data in limits.json")

        return results


if __name__ == "__main__":
    calibrator = timing_calibration()
    results = calibrator.calibrate_all()

    print("\n=== Calibration Results ===")
    for element, params in results.items():
        print(f"\n{element}:")
        print(f"  c_min: {params['c_min']:.6f} (target time: {params['min_time']})")
        print(f"  c_max: {params['c_max']:.6f} (target time: {params['max_time']})")

    # Save results
    with open('timing_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to calibration_results.json")
