"""
train_forecast.py
-----------------
Rigorous forecasting of CityPulse health_score with multi-model rolling-origin CV.

Outputs:
  data/processed/forecast/metrics_cv.csv
  data/processed/forecast/backtest_predictions.parquet
  data/processed/forecast/next7_predictions.parquet

Run:
  python -m src.models.train_forecast
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from src.data.utils import load_config
from src.utils.logger import setup_logger


# --------------------------
# Metrics
# --------------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100.0)


# --------------------------
# Feature engineering
# --------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.weekday           # 0-6
    df["month"] = df["date"].dt.month           # 1-12
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    return df

def add_lag_roll(df: pd.DataFrame, target_col="health_score",
                 lags=(1, 7, 14), rolls=(7, 14, 30),
                 extra_cols=("AQI","temp_min","temp_max","precip","wind_max")) -> pd.DataFrame:
    """
    Create lag & rolling stats for target and selected exogenous columns (if present).
    """
    df = df.sort_values("date").copy()

    # target lags/rolls
    for L in lags:
        df[f"{target_col}_lag{L}"] = df[target_col].shift(L)
    for W in rolls:
        df[f"{target_col}_roll{W}_mean"] = df[target_col].rolling(W, min_periods=max(3, W//3)).mean()
        df[f"{target_col}_roll{W}_std"]  = df[target_col].rolling(W, min_periods=max(3, W//3)).std()

    # exogenous: if available, use lags/rolls too
    for col in extra_cols:
        if col in df.columns:
            for L in (1, 7):
                df[f"{col}_lag{L}"] = df[col].shift(L)
            for W in (7, 14):
                df[f"{col}_roll{W}_mean"] = df[col].rolling(W, min_periods=max(3, W//3)).mean()

    return df


# --------------------------
# Rolling-origin CV splitter
# --------------------------
def rolling_origin_splits(dates: pd.Series, min_train_days=240, n_folds=5, horizon=7, gap=0):
    """
    Expanding window CV:
      - start with min_train_days
      - each fold predicts next 'horizon' days
    """
    dates = pd.to_datetime(dates).sort_values().unique()
    idx = pd.Series(range(len(dates)), index=dates)

    folds = []
    start_idx = 0
    train_end = min_train_days - 1

    for k in range(n_folds):
        test_start = train_end + 1 + gap
        test_end = test_start + horizon - 1
        if test_end >= len(dates):
            break
        tr_dates = dates[start_idx:train_end+1]
        te_dates = dates[test_start:test_end+1]
        folds.append((tr_dates, te_dates))
        # expand train window
        train_end = test_end
    return folds


# --------------------------
# Model zoo
# --------------------------
def get_models():
    models = {
        "linreg": Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", LinearRegression())]),
        "ridge":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", Ridge(alpha=10.0))]),
        "rf":     RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
    return models


# --------------------------
# Training / evaluation per city
# --------------------------
def train_eval_city(df_city: pd.DataFrame, city: str, horizon=7, logger=None):
    """
    Build features, run rolling-origin CV, return:
      - metrics_df (per model & fold)
      - backtest_preds (stacked predictions for all folds)
      - next7_preds (final fit on all data → next horizon)
    """
    data = df_city.copy()
    data = add_time_features(data)
    data = add_lag_roll(data)

    # Drop early rows without lags/rolls
    data = data.sort_values("date")
    min_start = data.filter(regex="lag|roll").isna().any(axis=1)
    if min_start.any():
        first_valid_idx = (~min_start).idxmax()
        data = data.loc[first_valid_idx:].copy()

    # Define features
    blacklist = {"date","city","health_band"}
    y_col = "health_score"
    X_cols = [c for c in data.columns if c not in blacklist and c != y_col]

    folds = rolling_origin_splits(data["date"], min_train_days=240, n_folds=6, horizon=horizon, gap=0)
    models = get_models()

    all_metrics = []
    all_bt_preds = []

    for fold_id, (tr_dates, te_dates) in enumerate(folds, start=1):
        tr = data[data["date"].isin(tr_dates)]
        te = data[data["date"].isin(te_dates)]

        X_tr, y_tr = tr[X_cols], tr[y_col]
        X_te, y_te = te[X_cols], te[y_col]

        for name, model in models.items():
            # fit
            model.fit(X_tr, y_tr)
            # predict
            y_hat = model.predict(X_te)

            fold_metrics = {
                "city": city,
                "model": name,
                "fold": fold_id,
                "n_train": len(tr),
                "n_test": len(te),
                "MAE": mean_absolute_error(y_te, y_hat),
                "RMSE": rmse(y_te, y_hat),
                "MAPE": mape(y_te, y_hat)
            }
            all_metrics.append(fold_metrics)

            # save backtest preds
            dfp = pd.DataFrame({
                "city": city,
                "model": name,
                "date": te["date"].values,
                "y_true": y_te.values,
                "y_pred": y_hat
            })
            all_bt_preds.append(dfp)

        if logger:
            logger.info(f"{city} | fold {fold_id}/{len(folds)} done (test {te['date'].min()}→{te['date'].max()})")

    metrics_df = pd.DataFrame(all_metrics)
    backtest_preds = pd.concat(all_bt_preds, ignore_index=True) if all_bt_preds else pd.DataFrame()

    # Fit final models on ALL data and predict next horizon days
    next7_list = []
    last_date = data["date"].max()
    fut_dates = [last_date + timedelta(days=i) for i in range(1, horizon+1)]

    # For multi-step, we’ll do a simple recursive strategy on features that depend on target lags.
    # We freeze exogenous features (calendar, AQ lags that exist) to last known stats.
    # This keeps it simple and robust for the portfolio.
    last_row = data.iloc[-1:].copy()
    cur_frame = data.copy()

    for name, model in models.items():
        model.fit(cur_frame[X_cols], cur_frame[y_col])

        preds = []
        tmp_frame = cur_frame.copy()

        for d in fut_dates:
            # roll one step: create a new row with calendar features & lagged/rolling recomputed
            new_row = tmp_frame.iloc[-1:].copy()
            new_row["date"] = d
            # update calendar features
            new_row["dow"] = pd.to_datetime(new_row["date"]).dt.weekday
            new_row["month"] = pd.to_datetime(new_row["date"]).dt.month
            new_row["year"] = pd.to_datetime(new_row["date"]).dt.year
            new_row["weekofyear"] = pd.to_datetime(new_row["date"]).dt.isocalendar().week.astype(int)

            # For simplicity, carry forward exogenous variables (AQI/temp/precip/wind) in recursive loop.
            # They could be forecasted separately in an advanced version.
            # Recompute target lags/rolls using predicted value when needed.
            # Append a placeholder to compute feature columns:
            tmp_plus = pd.concat([tmp_frame, new_row], ignore_index=True)
            tmp_plus = add_lag_roll(tmp_plus)

            # Extract just the last computed feature row
            X_new = tmp_plus.iloc[-1:][X_cols]

            y_new = model.predict(X_new)[0]
            preds.append(y_new)

            # append a complete row with the new prediction as health_score to continue recursion
            full_new = tmp_plus.iloc[-1:].copy()
            full_new["health_score"] = y_new
            tmp_frame = pd.concat([tmp_frame, full_new], ignore_index=True)

        df_next = pd.DataFrame({"city": city, "model": name, "date": fut_dates, "y_hat": preds})
        next7_list.append(df_next)

    next7_df = pd.concat(next7_list, ignore_index=True) if next7_list else pd.DataFrame()

    return metrics_df, backtest_preds, next7_df


# --------------------------
# Main
# --------------------------
def main():
    logger = setup_logger("train_forecast")
    cfg = load_config()

    processed = Path(cfg["paths"]["processed"])
    outdir = processed / "forecast"
    outdir.mkdir(parents=True, exist_ok=True)

    src = processed / "citypulse_health_daily.parquet"
    if not src.exists():
        logger.error(f"Missing input: {src}")
        return

    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    logger.info(f"Loaded health daily: {df.shape}")

    cities = sorted(df["city"].unique())
    horizon = 7

    all_metrics = []
    all_backtest = []
    all_next7 = []

    for city in cities:
        g = df[df["city"] == city].copy()
        if g["date"].nunique() < 360:  # need enough history
            logger.warning(f"Not enough history for {city} ({g['date'].nunique()} days). Skipping.")
            continue

        logger.info(f"Training city: {city}")
        m, bt, n7 = train_eval_city(g, city, horizon=horizon, logger=logger)
        if not m.empty:
            all_metrics.append(m)
        if not bt.empty:
            all_backtest.append(bt)
        if not n7.empty:
            all_next7.append(n7)

    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics.to_csv(outdir / "metrics_cv.csv", index=False)
        logger.info(f"Saved CV metrics → {outdir / 'metrics_cv.csv'}")
        # leaderboard
        lb = (metrics.groupby(["city","model"])[["MAE","RMSE","MAPE"]]
              .mean().reset_index().sort_values(["city","RMSE"]))
        logger.info(f"\nCV Leaderboard (avg over folds):\n{lb.to_string(index=False)}")
    else:
        logger.warning("No metrics computed.")

    if all_backtest:
        backtest = pd.concat(all_backtest, ignore_index=True)
        backtest.to_parquet(outdir / "backtest_predictions.parquet", index=False)
        logger.info(f"Saved backtest predictions → {outdir / 'backtest_predictions.parquet'}")

    if all_next7:
        next7 = pd.concat(all_next7, ignore_index=True)
        next7.to_parquet(outdir / "next7_predictions.parquet", index=False)
        logger.info(f"Saved next-7-day predictions → {outdir / 'next7_predictions.parquet'}")

    logger.info("✅ Forecast training complete.")


if __name__ == "__main__":
    main()
