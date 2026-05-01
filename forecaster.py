"""
forecaster.py
Runs a Prophet model per store and returns forecast + anomaly data.
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "retail.db")


def _get_store_data(store_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM weekly_sales WHERE store_id = ? ORDER BY week_start",
        conn,
        params=(store_id,),
    )
    conn.close()
    return df


def run_forecast(store_id: int, forecast_weeks: int = 12) -> dict:
    """
    Fits a Prophet model for store_id and returns a result dict with:
      - forecast_df      : full Prophet forecast dataframe
      - historical_df    : original weekly data with actuals
      - anomalies        : list of dicts describing anomalous historical weeks
      - store_metadata   : store_type, assortment, competition_distance
      - summary_stats    : avg_weekly_sales, peak_week, trough_week, total_forecast_sales
    """
    from prophet import Prophet  # lazy import — prophet is heavy

    df = _get_store_data(store_id)
    if df.empty:
        raise ValueError(f"No data found for store {store_id}.")

    store_meta = {
        "store_type": df["store_type"].iloc[0],
        "assortment": df["assortment"].iloc[0],
        "competition_distance": df["competition_distance"].iloc[0],
    }

    # Prophet expects ds / y
    prophet_df = df.rename(columns={"week_start": "ds", "total_sales": "y"}).copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["had_promo"] = prophet_df["had_promo"].astype(float)
    prophet_df["had_holiday"] = prophet_df["had_holiday"].astype(float)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.add_regressor("had_promo")
    model.add_regressor("had_holiday")

    model.fit(prophet_df[["ds", "y", "had_promo", "had_holiday"]])

    # Future dataframe
    future = model.make_future_dataframe(periods=forecast_weeks, freq="W")
    future["had_promo"] = 0.0
    future["had_holiday"] = 0.0

    # Back-fill historical regressor values for in-sample rows
    hist_lookup = prophet_df.set_index("ds")[["had_promo", "had_holiday"]]
    for col in ["had_promo", "had_holiday"]:
        future[col] = future["ds"].map(hist_lookup[col]).fillna(0.0)

    forecast = model.predict(future)

    # ---- Anomaly detection -----------------------------------------------
    # Join in-sample predictions back to historical
    in_sample = forecast[forecast["ds"].isin(prophet_df["ds"])][["ds", "yhat"]].copy()
    hist_merged = prophet_df[["ds", "y"]].merge(in_sample, on="ds")
    hist_merged["pct_diff"] = (hist_merged["y"] - hist_merged["yhat"]) / hist_merged["yhat"].abs()

    anomaly_threshold = 0.30
    anomaly_rows = hist_merged[hist_merged["pct_diff"].abs() > anomaly_threshold]

    anomalies = []
    for _, row in anomaly_rows.iterrows():
        promo_flag = int(prophet_df.loc[prophet_df["ds"] == row["ds"], "had_promo"].values[0])
        holiday_flag = int(prophet_df.loc[prophet_df["ds"] == row["ds"], "had_holiday"].values[0])
        anomalies.append(
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "actual": round(row["y"], 2),
                "expected": round(row["yhat"], 2),
                "pct_diff": round(row["pct_diff"] * 100, 1),
                "had_promo": promo_flag,
                "had_holiday": holiday_flag,
            }
        )

    # ---- Summary stats -----------------------------------------------
    future_only = forecast[~forecast["ds"].isin(prophet_df["ds"])].copy()
    peak_row = future_only.loc[future_only["yhat"].idxmax()]
    trough_row = future_only.loc[future_only["yhat"].idxmin()]

    summary_stats = {
        "avg_weekly_sales": round(prophet_df["y"].mean(), 2),
        "peak_week": peak_row["ds"].strftime("%Y-%m-%d"),
        "peak_sales": round(peak_row["yhat"], 2),
        "trough_week": trough_row["ds"].strftime("%Y-%m-%d"),
        "trough_sales": round(trough_row["yhat"], 2),
        "total_forecast_sales": round(future_only["yhat"].sum(), 2),
    }

    return {
        "forecast_df": forecast,
        "historical_df": df,
        "anomalies": anomalies,
        "store_metadata": store_meta,
        "summary_stats": summary_stats,
    }


def get_store_summary(store_id: int) -> dict:
    """
    Lightweight summary using last-4-weeks average — no Prophet fit.
    Used for the comparison tab to keep it fast.
    """
    df = _get_store_data(store_id)
    if df.empty:
        return {}

    last4 = df.tail(4)
    return {
        "store_id": store_id,
        "store_type": df["store_type"].iloc[0],
        "assortment": df["assortment"].iloc[0],
        "avg_weekly_sales": round(df["total_sales"].mean(), 2),
        "last_4wk_avg": round(last4["total_sales"].mean(), 2),
        "competition_distance": df["competition_distance"].iloc[0],
    }


def get_all_store_ids() -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT store_id FROM weekly_sales ORDER BY store_id"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_db_meta() -> dict:
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM weekly_sales").fetchone()[0]
    stores = conn.execute("SELECT COUNT(DISTINCT store_id) FROM weekly_sales").fetchone()[0]
    date_range = conn.execute(
        "SELECT MIN(week_start), MAX(week_start) FROM weekly_sales"
    ).fetchone()
    conn.close()
    return {
        "total_rows": count,
        "total_stores": stores,
        "min_date": date_range[0],
        "max_date": date_range[1],
    }
