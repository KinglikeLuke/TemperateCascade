from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score, mean_squared_error, \
    mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_predict, GroupKFold
from statsmodels.formula import api as smf
from tqdm import tqdm

from data_analysis import load_longform_df, FOLDER, OVERSHOOT_PROPERTIES
from temp_input.overshoot_trajectory import fit_parameters, overshoot_trajectory


def overshoot_classification(state_df):
    """
    Quantify overshoots by fitting a boosted tree model.
    Args:
        state_df:

    Returns:

    """

    state_df = state_df[50_000].drop("total", level="component", errors="ignore") > 0
    iso_df = state_df.xs(0.0, level="strength")
    con_df = state_df.xs(1.0, level="strength")
    try:
        integral_df = load_longform_df(FOLDER/"state_df_integral.csv")
    except FileNotFoundError:
        integral_df = calculate_integral(state_df)
    con_int_df = integral_df[50_000].xs(1.0, level="strength")
    # for component in ["AMOC"]: #con_df.index.get_level_values("component").unique():
    #     print(f"R2 predicting {component}: {tree_models(con_df, component)}")
    print(f"R2 predicting total: {tree_models(con_df, "AMOC")}")
    linear_component_models(con_int_df, component="AMOC")


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


def linear_component_models(df, component):
    y = (df.xs(component, level="component") > 0).rename("y").astype(int)
    data = y.reset_index()
    # --- model specifications ---
    formulas = {
        "a) integral": "y ~ integral",
        "b) T_peak+T_lim+t_conv+I": "y ~ T_peak + T_lim + t_conv + integral",
        "c) T_peak²·t_conv": "y ~ I(T_peak * t_conv**2)",
        "0) intercept only": "y ~ 1"
    }

    # --- fit on full data ---
    fits = {
        name: smf.logit(f, data=data).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": data["lhc"]}
        )
        for name, f in formulas.items()
    }

    def grouped_cv_logit(formula, data, group="lhc", k=5, seed=0):
        rng = np.random.default_rng(seed)
        groups = data[group].unique()
        rng.shuffle(groups)
        preds = pd.Series(index=data.index, dtype=float)
        for held_out in np.array_split(groups, k):
            test_mask = data[group].isin(held_out)
            m = smf.logit(formula, data=data[~test_mask]).fit(disp=0)
            preds[test_mask] = m.predict(data[test_mask])
        return preds

    rows = []
    for name, f in formulas.items():
        p = grouped_cv_logit(f, data).clip(1e-6, 1 - 1e-6)
        y = data["y"]
        rows.append({
            "model": name,
            "Accuracy_cv": accuracy_score(y, p > 0.5),
            #"McFadden_R2": fits[name].prsquared,
            "logloss_cv": log_loss(y, p),
            "brier_cv": brier_score_loss(y, p),
            "AUC_cv": roc_auc_score(y, p),
        })
    print(pd.DataFrame(rows).set_index("model").round(3))


def linear_models(df):
    # --- target: number of tipped components per run ---
    y = (df > 0).groupby(level=["lhc", *OVERSHOOT_PROPERTIES, "integral"]).sum().rename("y")
    data = y.reset_index()
    # --- model specifications ---
    formulas = {
        "a) integral": "y ~ integral",
        "b) T_peak+T_lim+t_conv+I": "y ~ T_peak + T_lim + t_conv",
        "c) T_peak²·t_conv": "y ~ I(T_peak**2 * t_conv)",
        "0) intercept only": "y ~ 1"
    }
    # --- fit on full data ---
    fits = {
        name: smf.ols(f, data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["lhc"]}
        )
        for name, f in formulas.items()
    }

    # --- grouped k-fold CV (whole lhc samples held out together, avoids leakage) ---
    def grouped_cv_rmse(formula, data, group="lhc", k=5, seed=0):
        rng = np.random.default_rng(seed)
        groups = data[group].unique()
        rng.shuffle(groups)
        errs = []
        for held_out in np.array_split(groups, k):
            test = data[data[group].isin(held_out)]
            train = data[~data[group].isin(held_out)]
            m = smf.ols(formula, data=train).fit()
            errs.append((test["y"] - m.predict(test)) ** 2)
        return np.sqrt(pd.concat(errs).mean())

    # --- comparison table ---
    rows = []
    for name, res in fits.items():
        pred = res.fittedvalues
        rows.append({
            "model": name,
            "n_params": int(res.df_model) + 1,
            "R2": res.rsquared,
            "adj_R2": res.rsquared_adj,
            "AIC": res.aic,
            "BIC": res.bic,
            "RMSE_in": np.sqrt(res.mse_resid),
            "RMSE_cv": grouped_cv_rmse(formulas[name], data),
            "MAE_in": np.abs(res.resid).mean(),
        })

    summary = pd.DataFrame(rows).set_index("model").round(3)
    print(summary)
    # fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    # for ax, (name, res) in zip(axes, fits.items()):
    #     ax.scatter(res.fittedvalues, data["y"], alpha=0.3, s=10)
    #     ax.plot([0, 5], [0, 5], "k--")
    #     ax.set(title=name, xlabel="predicted")
    # axes[0].set_ylabel("observed # tipped")
    # plt.show()


def tree_models(df, component):
    if component == "total":
        df = (df > 0).groupby(level=["lhc", *OVERSHOOT_PROPERTIES]).sum().rename("y")
    else:
        df = (df.xs(component, level="component") > 0).rename("y")
    df = df.reset_index()
    x_train, x_test, y_train, y_test= train_test_split(df[[*OVERSHOOT_PROPERTIES]], df["y"], test_size=0.2, random_state=42)
    if component == "total":
        tree_model = HistGradientBoostingRegressor(
            learning_rate=0.5, max_leaf_nodes=20, min_samples_leaf=50, early_stopping=True, l2_regularization=True,
        )
        tree_model.fit(x_train, y_train)
        y_pred = tree_model.predict(x_test)  # poisson deviance only on total component
    else:
        tree_model = HistGradientBoostingClassifier(
            learning_rate=0.1, max_leaf_nodes=20, min_samples_leaf=50, early_stopping=True, l2_regularization=True
        )
        tree_model.fit(x_train, y_train)
        y_pred = cross_val_predict(tree_model, df[[*OVERSHOOT_PROPERTIES]], df["y"], groups=df["lhc"],
                                   cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1] # poisson deviance only on total component
    if component == "total":
        return tree_model.score(x_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)
    return tree_model.score(x_test, y_test), log_loss(df["y"], y_pred), brier_score_loss(df["y"], y_pred)
