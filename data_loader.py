"""
data_loader.py
Reads Rossmann train.csv + store.csv, merges, aggregates to weekly,
and loads into retail.db SQLite database.
"""

import os
import sqlite3
import pandas as pd

DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
DEMO_DATA_DIR = os.path.join(os.path.dirname(__file__), "demo_data")
DB_PATH       = os.path.join(os.path.dirname(__file__), "retail.db")
STORES_TO_LOAD = list(range(1, 21))  # Stores 1–20


def _resolve_data_dir() -> tuple[str, bool]:
    """Return (data_dir, is_demo). Falls back to demo_data/ if real data absent."""
    train_real = os.path.join(DATA_DIR, "train.csv")
    train_demo = os.path.join(DEMO_DATA_DIR, "train.csv")
    if os.path.exists(train_real):
        return DATA_DIR, False
    if os.path.exists(train_demo):
        return DEMO_DATA_DIR, True
    raise FileNotFoundError(
        "No data found. Place train.csv and store.csv in /data (real) "
        "or /demo_data (synthetic), then run: python data_loader.py"
    )


def db_has_data() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM weekly_sales").fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def load_rossmann_data() -> None:
    if db_has_data():
        print("Database already exists, skipping load.")
        return

    data_dir, is_demo = _resolve_data_dir()
    train_path = os.path.join(data_dir, "train.csv")
    store_path = os.path.join(data_dir, "store.csv")

    if is_demo:
        print("Using synthetic demo data (no real Kaggle data found).")
    print("Loading train.csv ...")
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)

    print("Loading store.csv ...")
    store = pd.read_csv(store_path, low_memory=False)

    # Merge on Store
    df = train.merge(store, on="Store", how="left")

    # Filter: open stores only, stores 1–20
    df = df[df["Open"] == 1]
    df = df[df["Store"].isin(STORES_TO_LOAD)]

    # Normalise StateHoliday to binary
    df["StateHoliday"] = df["StateHoliday"].apply(lambda x: 0 if str(x) in ("0", "0.0") else 1)

    # Fill missing CompetitionDistance with median
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    # Keep relevant columns
    keep_cols = [
        "Store", "Date", "Sales", "Customers",
        "Promo", "StateHoliday", "SchoolHoliday",
        "StoreType", "Assortment", "CompetitionDistance",
    ]
    df = df[keep_cols].copy()

    # Aggregate to weekly level
    df = df.set_index("Date")
    weekly = (
        df.groupby("Store")
        .resample("W")
        .agg(
            total_sales=("Sales", "sum"),
            total_customers=("Customers", "sum"),
            had_promo=("Promo", "max"),
            had_holiday=("StateHoliday", "max"),
            had_school_holiday=("SchoolHoliday", "max"),
            store_type=("StoreType", "first"),
            assortment=("Assortment", "first"),
            competition_distance=("CompetitionDistance", "first"),
        )
        .reset_index()
    )

    weekly = weekly.rename(columns={"Store": "store_id", "Date": "week_start"})
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")

    # Remove weeks with 0 sales (fully closed weeks after filtering)
    weekly = weekly[weekly["total_sales"] > 0]

    # Write to SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS weekly_sales")
    conn.execute(
        """
        CREATE TABLE weekly_sales (
            record_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id             INTEGER,
            week_start           TEXT,
            total_sales          REAL,
            total_customers      INTEGER,
            had_promo            INTEGER,
            had_holiday          INTEGER,
            had_school_holiday   INTEGER,
            store_type           TEXT,
            assortment           TEXT,
            competition_distance REAL
        )
        """
    )

    weekly.to_sql("weekly_sales", conn, if_exists="append", index=False)
    conn.close()

    rows = len(weekly)
    stores = weekly["store_id"].nunique()
    print(f"Database loaded: {rows:,} rows across {stores} stores.")


if __name__ == "__main__":
    load_rossmann_data()
