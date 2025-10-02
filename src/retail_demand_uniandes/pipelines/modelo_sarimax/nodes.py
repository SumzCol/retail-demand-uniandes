from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

def clean_and_aggregate(df_raw: pd.DataFrame, params: dict) -> pd.DataFrame:

    df = df_raw.copy()
    date_col = "order_purchase_timestamp"
    cat_col = "product_category_name_english"
    freq = params.get("freq", "D")

    # fechas
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[cat_col] = df[cat_col].astype("category")

    # columnas exógenas candidatas (prefijo configurable)
    exog_prefix = params.get("exog_prefix", "promedio")
    # si ya vienen calculadas en el dataset final del notebook, se filtran por prefijo
    exog_cols = [c for c in df.columns if c.startswith(exog_prefix)]

    # métrica objetivo
    if "total_ventas" in df.columns:
        target_col = "total_ventas"
    else:
        # fallback: quantity*price si aplica al dominio; ajusta a tu dataset
        target_col = "price"
    agg_dict = {target_col: "sum"}
    for c in exog_cols:
        agg_dict[c] = "mean"

    # agregar
    df["ds"] = df[date_col].dt.floor(freq)
    g = df.groupby([cat_col, "ds"], observed=True).agg(agg_dict).reset_index()

    # asegurar continuidad temporal por categoría
    def reindex_cat(grp):
        idx = pd.date_range(grp["ds"].min(), grp["ds"].max(), freq=freq)
        grp = grp.set_index("ds").reindex(idx)
        grp.index.name = "ds"
        return grp

    parts = []
    for cat, sub in g.groupby(cat_col, observed=True):
        sub2 = reindex_cat(sub)
        sub2[cat_col] = cat
        parts.append(sub2.reset_index())

    out = pd.concat(parts, ignore_index=True)
    # imputaciones mínimas: objetivo puede ser 0, exógenas con forward/back fill + mean
    out[target_col] = out[target_col].fillna(0.0)
    for c in exog_cols:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = out[c].fillna(method="ffill").fillna(method="bfill")
        out[c] = out[c].fillna(out[c].mean())

    # ordenar columnas
    cols = [cat_col, "ds", target_col] + exog_cols
    return out[cols]


def adf_test_by_category(agg: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, List[str]]:

    cat_col = "product_category_name_english"
    target_col = "total_ventas" if "total_ventas" in agg.columns else "price"
    alpha = params.get("adf_alpha", 0.05)

    rows = []
    non_stationary = []

    for cat, sub in agg.groupby(cat_col, observed=True):
        y = sub[target_col].astype(float).dropna()
        if len(y) < params.get("min_obs", 10):
            rows.append({
                "category": cat, "n_obs": len(y),
                "adf_stat": np.nan, "pvalue": np.nan,
                "is_stationary": False, "note": "insufficient_obs"
            })
            non_stationary.append(cat)
            continue
        try:
            res = adfuller(y, autolag="AIC")
            pval = res[1]
            is_stat = bool(pval < alpha)
            rows.append({
                "category": cat, "n_obs": len(y),
                "adf_stat": res[0], "pvalue": pval,
                "is_stationary": is_stat, "note": ""
            })
            if not is_stat:
                non_stationary.append(cat)
        except Exception as e:
            rows.append({
                "category": cat, "n_obs": len(y),
                "adf_stat": np.nan, "pvalue": np.nan,
                "is_stationary": False, "note": f"error:{e}"
            })
            non_stationary.append(cat)

    table = pd.DataFrame(rows).sort_values(["is_stationary","pvalue"], ascending=[False, True])
    return table, non_stationary


def difference_series(
    agg: pd.DataFrame,
    non_stationary: List[str],
    params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cat_col = "product_category_name_english"
    target_col = "total_ventas" if "total_ventas" in agg.columns else "price"
    alpha = params.get("adf_alpha", 0.05)
    max_diff = params.get("max_diff", 2)

    rows = []
    parts = []

    for cat, sub in agg.groupby(cat_col, observed=True):
        sub = sub.sort_values("ds").reset_index(drop=True)
        y = sub[target_col].astype(float).copy()

        d = 0
        if cat in non_stationary:
            for step in range(1, max_diff + 1):
                y_candidate = y.diff(step).dropna()
                try:
                    res = adfuller(y_candidate, autolag="AIC")
                    if res[1] < alpha:
                        d = step
                        break
                except Exception:
                    pass
            # si no se logró, usar el máximo d probado
            if d == 0:
                d = max_diff

        sub["y_diff"] = y if d == 0 else y.diff(d)
        parts.append(sub)
        rows.append({"category": cat, "d": d})

    agg_diff = pd.concat(parts, ignore_index=True)
    d_map = pd.DataFrame(rows)
    return agg_diff, d_map


def prepare_exog(agg_diff: pd.DataFrame, params: dict) -> Dict[str, pd.DataFrame]:

    cat_col = "product_category_name_english"
    target_col = "total_ventas" if "total_ventas" in agg_diff.columns else "price"
    exog_prefix = params.get("exog_prefix", "promedio")

    exog_cols = [c for c in agg_diff.columns if c.startswith(exog_prefix)]
    exog_dict = {}

    for cat, sub in agg_diff.groupby(cat_col, observed=True):
        sub = sub.sort_values("ds").copy()
        y = sub["y_diff"] if "y_diff" in sub.columns else sub[target_col]
        mask = ~y.isna()
        X = sub.loc[mask, exog_cols].copy()
        # limpieza
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method="ffill").fillna(method="bfill")
        X = X.fillna(X.mean(numeric_only=True))
        exog_dict[str(cat)] = X.reset_index(drop=True)

    return exog_dict


def train_sarimax_by_category(
    agg_diff: pd.DataFrame,
    d_map: pd.DataFrame,
    exog_dict: Dict[str, pd.DataFrame],
    params: dict
) -> pd.DataFrame:

    cat_col = "product_category_name_english"
    target_col = "total_ventas" if "total_ventas" in agg_diff.columns else "price"

    train_days = params.get("train_days", 30)
    test_days = params.get("test_days", 15)
    seasonal_period = params.get("seasonal_period", 7)
    min_obs = params.get("min_obs", 10)

    results = []

    for cat, sub in agg_diff.groupby(cat_col, observed=True):
        sub = sub.sort_values("ds").reset_index(drop=True)
        y = sub["y_diff"] if "y_diff" in sub.columns else sub[target_col]
        y = y.astype(float).dropna().reset_index(drop=True)
        n = len(y)

        if n < (train_days + test_days) or n < min_obs:
            results.append({
                "category": cat,
                "status": "skipped",
                "reason": "insufficient_data",
                "n_obs": n
            })
            continue

        y_train = y.iloc[-(train_days + test_days):-test_days]
        y_test = y.iloc[-test_days:]

        d = int(d_map.loc[d_map["category"] == cat, "d"].values[0]) if cat in d_map["category"].values else 0
        order = (1, d, 1)
        seasonal_order = (1, 0, 1, seasonal_period)

        # exógenas alineadas
        X = exog_dict.get(str(cat))
        X = None if X is None or X.empty else X
        if X is not None:
            X = X.iloc[-(n):].reset_index(drop=True)  # asegurar misma longitud base
            X_train = X.iloc[-(train_days + test_days):-test_days]
            X_test = X.iloc[-test_days:]
        else:
            X_train = None
            X_test = None

        try:
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit = model.fit(disp=False)
            fc = fit.get_forecast(steps=test_days, exog=X_test)
            mean_fc = fc.predicted_mean
            mae = float(np.mean(np.abs(mean_fc.values - y_test.values)))
            mse = float(np.mean((mean_fc.values - y_test.values) ** 2))

            results.append({
                "category": cat,
                "status": "ok",
                "n_obs": n,
                "train_days": int(train_days),
                "test_days": int(test_days),
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "mae": mae,
                "mse": mse
            })
        except Exception as e:
            results.append({
                "category": cat,
                "status": "failed",
                "reason": str(e),
                "n_obs": n,
                "order": str(order),
                "seasonal_order": str(seasonal_order)
            })

    return pd.DataFrame(results)


def summarize_results(results: pd.DataFrame, params: dict) -> pd.DataFrame:
    if "mae" in results.columns:
        results = results.sort_values("mae", na_position="last").reset_index(drop=True)
    top_n = params.get("top_n_plots", 3)
    results["rank"] = np.arange(1, len(results) + 1)
    return results.head(top_n)