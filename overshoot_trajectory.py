import numpy as np
from scipy.optimize import root, minimize_scalar
from matplotlib import pyplot as plt
import time

def overshoot_trajectory(t, T0, T_lim, R, mu0, mu1):
    y = R + mu0 * (T0 - T_lim)
    return (
        T0
        + y * t
        - (1 - np.exp(-(mu0 + mu1 * t) * t)) * (y * t - (T_lim - T0))
    )


def _find_peak_time(T0, T_lim, R, mu0, mu1, t_upper):
    res = minimize_scalar(
        lambda t: -overshoot_trajectory(t, T0, T_lim, R, mu0, mu1),
        bounds=(0, t_upper),
        method="bounded",
    )
    return res.x


def _residuals(params, T0, T_lim, mu0, Tmax, tconv, eps):
    R, mu1 = params

    # threshold instead of exact T_lim
    T_thresh = T_lim + eps

    # condition 1: near convergence
    r1 = overshoot_trajectory(tconv, T0, T_lim, R, mu0, mu1) - T_thresh

    # condition 2: peak value
    t_peak = _find_peak_time(T0, T_lim, R, mu0, mu1, tconv)
    r2 = overshoot_trajectory(t_peak, T0, T_lim, R, mu0, mu1) - Tmax

    return [r1, r2]

def _initial_guess(T0, Tmax, T_lim, tconv, mu0=0.0015, delta=1e-2):
    """Match slope to slope to slope to peak, match exponential to position of peak. See chatgpt

    Args:
        T0 (_type_): _description_
        Tmax (_type_): _description_
        T_lim (_type_): _description_
        tconv (_type_): _description_
        mu0 (float, optional): _description_. Defaults to 0.0015.

    Returns:
        _type_: _description_
    """
    dT = T_lim - T0

    # peak time estimate
    t_peak = 0.5 * tconv

    # R guess
    R = 2 * (Tmax - T0) / tconv + mu0 * dT

    # mu1 guess
    mu1 = (1 - mu0 * t_peak) / (t_peak ** 2)
    mu1 = max(mu1, 1e-6)

    return R, mu1

def fit_parameters(T0, Tmax, T_lim, tconv, mu0=0.0015):
    """Find parameters R and mu1 so that the temperature trajectory has max Tmax and reaches 0.01 above T_lim at tconv

    Args:
        T0 (_type_): _description_
        Tmax (_type_): _description_
        T_lim (_type_): _description_
        tconv (_type_): _description_
        mu0 (float, optional): _description_. Defaults to 0.0015.
        R_guess (float, optional): _description_. Defaults to 1.0.
        mu1_guess (float, optional): _description_. Defaults to 0.1.

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    eps = 1e-3
    if np.isclose(Tmax, T_lim, atol = eps):
        Tmax += 10*eps
    x0 = _initial_guess(T0, Tmax, T_lim, tconv, mu0=mu0, delta=eps)
    sol = root(
        _residuals,
        x0=x0,
        args=(T0, T_lim, mu0, Tmax, tconv, eps),
        method="hybr",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    R, mu1 = sol.x
    return R, mu0, mu1

def timeit(f):
    def wrap(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        print(f"{f.__name__} ran in {time.time() - t1:.6f}s")
        return res
    return wrap

@timeit
def test_consistency():
    from pydoe import lhs
    T0 = 1
    n_tests = 1000
    lhc_distr = np.array(lhs(3, samples=n_tests))
    Tmax = np.round(4*lhc_distr[:,0] + 2, 2)
    T_lim = np.round(2*lhc_distr[:,1], 2)
    tconv = np.round(900*lhc_distr[:,2] + 100, 0)
    errors = 0
    for i in range(n_tests):
        try:
            fit_parameters(T0, Tmax[i], T_lim[i], tconv[i])
        except RuntimeError as error:
            print(i)
            print(Tmax[i], T_lim[i], tconv[i])
            errors += 1
    print(f"done: {errors} errors")

if __name__=="__main__":
    t = np.linspace(0, 2100)
    plt.plot(t, overshoot_trajectory(t, 1, 2, *fit_parameters(1, 2, 2, 500)))
    plt.show()
    # test_consistency()