from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score, mean_squared_error, \
    mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_predict, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from statsmodels.formula import api as smf
from tqdm import tqdm

from data_analysis import load_longform_df, FOLDER, OVERSHOOT_PROPERTIES
from temp_input.overshoot_trajectory import fit_parameters, overshoot_trajectory


def calculate_integral(state_df):
    state_df = state_df.copy()
    integral_mapping = {}
    for T_peak, T_lim, t_conv in tqdm(np.unique(state_df.reset_index()[["T_peak", "T_lim", "t_conv"]].values, axis=0)):
        parameters = fit_parameters(1, T_peak, T_lim, t_conv)
        integral_mapping[T_peak, T_lim, t_conv] = np.round(
        scipy.integrate.quad(overshoot_trajectory, 0, 1000, args=(1, T_lim, *parameters))[0], 3)
    index_df = state_df.index.to_frame(index=False)

    index_df["integral"] = [
        integral_mapping[(T_peak, T_lim, t_conv)]
        for T_peak, T_lim, t_conv in index_df[
            ["T_peak", "T_lim", "t_conv"]
        ].itertuples(index=False, name=None)
    ]

    state_df.index = pd.MultiIndex.from_frame(
        index_df[
            ["lhc", "T_peak", "T_lim", "t_conv", "integral",
             "strength", "component"]
        ]
    )
    state_df.to_csv(FOLDER/"state_df_integral.csv")
    return state_df


# OVERSHOOT_PROPERTIES is assumed to exist already, e.g. ["T_peak", "T_lim", "t_conv"]
OVS_BASE = [*OVERSHOOT_PROPERTIES, "integral"]
OVS_COLS = [*OVS_BASE, "Tpeak2tconv"]  # derived feature gives the linear model its model-c term


# ---------- data ----------
def _table(y, tipping_properties):
    """Attach the lhc parameters as columns but KEEP lhc itself as the CV group id.
    (add_system_properties drops lhc, which makes grouped CV impossible.)"""
    props = list(tipping_properties.columns)
    data = (y.reset_index()
             .merge(tipping_properties, left_on="lhc", right_index=True,
                    how="left", validate="m:1"))
    data["Tpeak2tconv"] = data["T_peak"] ** 2 * data["t_conv"]
    return data, props


# ---------- model builders: each returns fit_predict(train_df, test_df) -> np.ndarray ----------
def _mean():
    return lambda tr, te: np.full(len(te), tr["y"].mean())


def _linear(cols):
    def fp(tr, te):
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[cols], tr["y"])
        return m.predict(te[cols])
    return fp


def _gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                         early_stopping=False, random_state=0)


def _tree(cols):
    return lambda tr, te: _gbm().fit(tr[cols], tr["y"]).predict(te[cols])


def _hybrid(param_cols, ovs_cols, n_iter=10):
    """Additive: tree(params) + linear(overshoot), fitted by backfitting."""
    def fp(tr, te):
        lin = make_pipeline(StandardScaler(), LinearRegression())
        tree = _gbm()
        y = tr["y"].to_numpy()
        f_tree = np.zeros(len(tr))
        for _ in range(n_iter):
            lin.fit(tr[ovs_cols], y - f_tree)
            f_lin = lin.predict(tr[ovs_cols])
            tree.fit(tr[param_cols], y - f_lin)
            f_tree = tree.predict(tr[param_cols])
        return tree.predict(te[param_cols]) + lin.predict(te[ovs_cols])
    return fp


# ---------- the ladder ----------
def run_ladder(data, param_cols, ovs_cols=OVS_COLS, binary=False,
               group="lhc", k=5, seeds=range(5)):
    """Grouped-CV skill of each rung relative to the out-of-fold intercept model.
    skill = 1 - SSE_model / SSE_M0  (= CV R2 for counts, Brier skill score for binary y)."""
    ladder = {
        "M0 intercept": _mean(),
        "M1 linear(overshoot)": _linear(ovs_cols),
        "M2 tree(overshoot)": _tree(ovs_cols),
        "M3 tree(params)": _tree(param_cols),
        "M4 tree(params)+linear(overshoot)": _hybrid(param_cols, ovs_cols),
        "M5 tree(params+overshoot)": _tree(param_cols + ovs_cols),
    }
    names = list(ladder)
    y = data["y"].to_numpy(dtype=float)
    skills = {n: [] for n in names}

    for seed in seeds:  # same folds for every model within a seed
        rng = np.random.default_rng(seed)
        groups = data[group].unique()
        rng.shuffle(groups)
        oof = {n: np.empty(len(data)) for n in names}
        for held in np.array_split(groups, k):
            te_mask = data[group].isin(held).to_numpy()
            tr, te = data[~te_mask], data[te_mask]
            for n, fp in ladder.items():
                p = fp(tr, te)
                oof[n][te_mask] = np.clip(p, 0, 1) if binary else p
        sse0 = ((y - oof[names[0]]) ** 2).sum()
        for n in names:
            skills[n].append(1 - ((y - oof[n]) ** 2).sum() / sse0)

    table = pd.DataFrame({n: {"skill_mean": np.mean(s), "skill_sd": np.std(s)}
                          for n, s in skills.items()}).T.round(3)
    m = table["skill_mean"]
    decomposition = pd.Series({
        "linear overshoot share (M1)": m.iloc[1],
        "total marginal overshoot share (M2)": m.iloc[2],
        "nonlinearity in overshoot (M2-M1)": m.iloc[2] - m.iloc[1],
        "parameter-only share (M3)": m.iloc[3],
        "overshoot added to params (M5-M3)": m.iloc[5] - m.iloc[3],
        "interaction params x overshoot (M5-M4)": m.iloc[5] - m.iloc[4],
        "unexplained by params+summaries (1-M5)": 1 - m.iloc[5],
    }).round(3)
    return table, decomposition


# ---------- model-free sanity check (needs the full crossed design) ----------
def anova_shares(data, ovs_base=OVS_BASE, group="lhc"):
    """First-order variance shares from group means; remainder = interaction + unresolved."""
    v = data["y"].var(ddof=0)
    s_par = data.groupby(group)["y"].transform("mean").var(ddof=0) / v
    s_ovs = data.groupby(ovs_base)["y"].transform("mean").var(ddof=0) / v
    return pd.Series({"params (lhc)": s_par, "overshoot": s_ovs,
                      "interaction/rest": 1 - s_par - s_ovs}).round(3)


# ---------- wrappers ----------
def ladder_count(df, tipping_properties, **kw):
    y = (df > 0).groupby(level=["lhc", *OVS_BASE]).sum().rename("y")
    data, props = _table(y, tipping_properties)
    # print(anova_shares(data)) # usless bc study not crossed
    return run_ladder(data, props, binary=False, **kw)


def ladder_component(df, component, tipping_properties, **kw):
    y = (df.xs(component, level="component") > 0).astype(int).rename("y")
    data, props = _table(y, tipping_properties)
    # print(anova_shares(data))
    return run_ladder(data, props, binary=True, **kw)

if __name__ == "__main__":
    state_df = load_longform_df(FOLDER / "dataframe.csv")
    try:
        integral_df = load_longform_df(FOLDER/"state_df_integral.csv")
    except FileNotFoundError:
        integral_df = calculate_integral(state_df)
    integral_df = integral_df[50_000].xs(1.0, level="strength")
    tipping_properties = pd.read_csv(f"{FOLDER}/tipping_properties.csv", index_col=0)
    table, decomp = ladder_count(integral_df, tipping_properties)
    print(decomp)
    # What can this tell me?
    # Beyond the obvious (Tpeak^2 tconv is a decent surrogate for integral):
    # about half our output spread is due to temperature
    # Dependence between temperature and tipping numbers seems to be almost wholly linear - TODO: linear-nonlinear model with nonlinear lhc fit
    # TODO more lhc samples!