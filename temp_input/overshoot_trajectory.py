import numpy as np
from scipy.optimize import root, minimize_scalar
from matplotlib import pyplot as plt
import time


def projection_forcing(times, temperatures, extrapolation="edge"):
    """Return an interpolated temperature forcing from an existing projection."""
    times = np.asarray(times, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)

    if times.ndim != 1 or temperatures.ndim != 1:
        raise ValueError("Projection times and temperatures must be one-dimensional.")
    if len(times) != len(temperatures):
        raise ValueError("Projection times and temperatures must have the same length.")
    if len(times) < 2:
        raise ValueError("Projection forcing needs at least two time points.")

    order = np.argsort(times)
    times = times[order]
    temperatures = temperatures[order]

    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(temperatures)):
        raise ValueError("Projection forcing contains non-finite values.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Projection forcing times must be unique and strictly increasing.")
    if extrapolation not in {"edge", "linear"}:
        raise ValueError("extrapolation must be 'edge' or 'linear'.")

    def forcing(t):
        scalar_input = np.isscalar(t)
        t_array = np.atleast_1d(np.asarray(t, dtype=float))
        result = np.interp(t_array, times, temperatures)

        if extrapolation == "linear":
            left = t_array < times[0]
            right = t_array > times[-1]
            if np.any(left):
                slope = (temperatures[1] - temperatures[0]) / (times[1] - times[0])
                result = np.asarray(result)
                result[left] = temperatures[0] + slope * (t_array[left] - times[0])
            if np.any(right):
                slope = (temperatures[-1] - temperatures[-2]) / (times[-1] - times[-2])
                result = np.asarray(result)
                result[right] = temperatures[-1] + slope * (t_array[right] - times[-1])

        if scalar_input:
            return float(result[0])
        return result

    return forcing


def load_projection_forcing(
    path,
    time_column=None,
    temperature_column=None,
    time_zero=None,
    extrapolation="edge",
):
    """Load an existing temperature projection and return interpolation metadata."""
    columns, data = _read_projection_table(path)
    if data.size == 0:
        raise ValueError(f"No projection data found in {path}.")

    time_index, temperature_index = _projection_column_indices(
        columns, time_column, temperature_column, data.shape[1]
    )
    return _projection_from_columns(
        data,
        time_index,
        temperature_index,
        time_zero=time_zero,
        extrapolation=extrapolation,
        name=temperature_column,
    )


def load_projection_forcings(
    path,
    time_column=None,
    temperature_columns=None,
    time_zero=None,
    extrapolation="edge",
):
    """Load one or more temperature projections from one table."""
    columns, data = _read_projection_table(path)
    if data.size == 0:
        raise ValueError(f"No projection data found in {path}.")

    time_index = _projection_time_column_index(columns, time_column, data.shape[1])
    temperature_indices = _projection_temperature_column_indices(
        columns, temperature_columns, data.shape[1], time_index
    )

    forcings = []
    for temperature_index in temperature_indices:
        name = columns[temperature_index] if columns is not None else f"temperature_{temperature_index}"
        forcings.append(
            _projection_from_columns(
                data,
                time_index,
                temperature_index,
                time_zero=time_zero,
                extrapolation=extrapolation,
                name=name,
            )
        )
    return forcings


def _projection_from_columns(data, time_index, temperature_index, time_zero, extrapolation, name):
    times = data[:, time_index]
    temperatures = data[:, temperature_index]
    if time_zero is None:
        time_zero = np.nanmin(times)
    times = times - float(time_zero)

    forcing = projection_forcing(times, temperatures, extrapolation=extrapolation)
    return {
        "forcing": forcing,
        "times": times,
        "temperatures": temperatures,
        "T_peak": float(np.nanmax(temperatures)),
        "T_lim": float(temperatures[np.nanargmax(times)]),
        "t_conv": float(np.nanmax(times)),
        "name": name,
    }


def _read_projection_table(path):
    delimiter = None
    columns = None
    rows = []

    with open(path, "r") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if delimiter is None:
                delimiter = _infer_delimiter(line)
            values = _split_projection_line(line, delimiter)
            if columns is None and not _all_float(values):
                columns = values
                continue
            rows.append([float(value) for value in values])

    if not rows:
        return columns, np.empty((0, 0))
    return columns, np.asarray(rows, dtype=float)


def _projection_column_indices(columns, time_column, temperature_column, n_columns):
    time_index = _projection_time_column_index(columns, time_column, n_columns)
    temperature_indices = _projection_temperature_column_indices(
        columns, temperature_column, n_columns, time_index
    )
    if len(temperature_indices) != 1:
        raise ValueError("load_projection_forcing needs exactly one temperature column.")
    return time_index, temperature_indices[0]


def _projection_time_column_index(columns, time_column, n_columns):
    if columns is None:
        if time_column is not None:
            raise ValueError("Named projection columns require a header row.")
        if n_columns != 2:
            raise ValueError("Headerless projection files must have exactly two columns.")
        return 0

    if time_column is None:
        time_column = _first_matching_column(
            columns,
            exact={"time", "t", "year", "years"},
            contains={"time", "year"},
        )

    if time_column is None:
        raise ValueError("Could not infer projection time column.")
    return columns.index(time_column)


def _projection_temperature_column_indices(columns, temperature_columns, n_columns, time_index):
    if columns is None:
        if temperature_columns is not None:
            raise ValueError("Named projection columns require a header row.")
        return [1]

    if isinstance(temperature_columns, str):
        if temperature_columns.strip() == "*":
            temperature_columns = None
        else:
            temperature_columns = [
                column.strip()
                for column in temperature_columns.split(",")
                if column.strip()
            ]

    if temperature_columns is None:
        temperature_columns = [
            column for i, column in enumerate(columns)
            if i != time_index
        ]

    if not temperature_columns and n_columns == 2:
        temperature_columns = [columns[1 - time_index]]
    if not temperature_columns:
        raise ValueError("Could not infer projection temperature column.")

    missing_columns = [column for column in temperature_columns if column not in columns]
    if missing_columns:
        raise ValueError(f"Projection temperature columns not found: {missing_columns}")
    if columns[time_index] in temperature_columns:
        raise ValueError("Temperature columns must not include the time column.")

    return [columns.index(column) for column in temperature_columns]


def _infer_delimiter(line):
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None


def _split_projection_line(line, delimiter):
    if delimiter is None:
        return line.split()
    return [value.strip() for value in line.split(delimiter)]


def _all_float(values):
    try:
        [float(value) for value in values]
    except ValueError:
        return False
    return True


def _first_matching_column(columns, exact, contains):
    for column in columns:
        if column.lower() in exact:
            return column
    for column in columns:
        if any(pattern in column.lower() for pattern in contains):
            return column
    return None


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
    # Somewhat hacky workaround: flat trajectories with tconv = 0
    if Tmax == T_lim and T0 == Tmax:
        return 0, 0, 0
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
    print(load_projection_forcing("temperature_HL.csv", "timebounds", "temperature", time_zero=2020))

    # test_consistency()
