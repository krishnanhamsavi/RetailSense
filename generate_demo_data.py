"""
generate_demo_data.py
Generates synthetic Rossmann-style CSVs in demo_data/
Mimics real seasonal patterns, promotions, holidays, and store types.
Run once: python generate_demo_data.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
OUT_DIR = os.path.join(os.path.dirname(__file__), "demo_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Store metadata ──────────────────────────────────────────────────────────
N_STORES = 20

STORE_CONFIGS = {
    "a": dict(base_sales=6500, noise=0.18, promo_lift=0.12),
    "b": dict(base_sales=12000, noise=0.14, promo_lift=0.08),
    "c": dict(base_sales=4800, noise=0.20, promo_lift=0.15),
    "d": dict(base_sales=8200, noise=0.16, promo_lift=0.10),
}

store_types   = (["a"] * 8 + ["b"] * 3 + ["c"] * 5 + ["d"] * 4)[:N_STORES]
assortments   = (["a"] * 10 + ["b"] * 4 + ["c"] * 6)[:N_STORES]
comp_distances = np.random.choice([260, 570, 1200, 2500, 5000, 9000], N_STORES)

store_df = pd.DataFrame({
    "Store":               range(1, N_STORES + 1),
    "StoreType":           store_types,
    "Assortment":          assortments,
    "CompetitionDistance": comp_distances,
    "CompetitionOpenSinceMonth": np.random.randint(1, 13, N_STORES),
    "CompetitionOpenSinceYear":  np.random.randint(2008, 2014, N_STORES),
    "Promo2":              np.random.randint(0, 2, N_STORES),
    "Promo2SinceWeek":     np.random.randint(1, 52, N_STORES),
    "Promo2SinceYear":     np.random.randint(2012, 2015, N_STORES),
    "PromoInterval":       np.random.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", ""], N_STORES),
})

# ── Daily sales generation ──────────────────────────────────────────────────
dates = pd.date_range("2013-01-01", "2015-07-31", freq="D")
records = []

for store_id in range(1, N_STORES + 1):
    cfg = STORE_CONFIGS[store_types[store_id - 1]]
    base   = cfg["base_sales"]
    noise  = cfg["noise"]
    plift  = cfg["promo_lift"]

    for date in dates:
        dow = date.dayofweek  # 0=Mon … 6=Sun
        # Closed Sundays and some Saturdays
        if dow == 6:
            continue
        open_flag = 1

        # Seasonal index: peak Dec, dip Jan/Feb
        month = date.month
        seasonal = {1: 0.82, 2: 0.85, 3: 0.92, 4: 0.95, 5: 0.97,
                    6: 0.96, 7: 0.93, 8: 0.95, 9: 0.98, 10: 1.00,
                    11: 1.05, 12: 1.22}[month]

        # Day-of-week effect
        dow_effect = {0: 1.02, 1: 1.00, 2: 0.99, 3: 1.01,
                      4: 1.05, 5: 1.12}[dow]

        # Promotion: random ~30% of days
        promo = int(np.random.rand() < 0.30)

        # State holiday: ~5% of days
        state_holiday = "0"
        if np.random.rand() < 0.05:
            state_holiday = np.random.choice(["a", "b", "c"])

        # School holiday: ~15% of days
        school_holiday = int(np.random.rand() < 0.15)

        # Sales calculation
        sales = (
            base
            * seasonal
            * dow_effect
            * (1 + plift * promo)
            * (0.85 if state_holiday != "0" else 1.0)
            * (1 + noise * np.random.randn())
        )
        sales = max(100, round(sales))
        customers = max(10, int(sales / np.random.uniform(8, 14)))

        records.append({
            "Store":         store_id,
            "DayOfWeek":     dow + 1,
            "Date":          date.strftime("%Y-%m-%d"),
            "Sales":         sales,
            "Customers":     customers,
            "Open":          open_flag,
            "Promo":         promo,
            "StateHoliday":  state_holiday,
            "SchoolHoliday": school_holiday,
        })

train_df = pd.DataFrame(records)
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
store_df.to_csv(os.path.join(OUT_DIR, "store.csv"), index=False)

print(f"Demo data generated:")
print(f"  train.csv — {len(train_df):,} rows across {N_STORES} stores")
print(f"  store.csv — {len(store_df)} stores")
print(f"  Saved to: {OUT_DIR}")
