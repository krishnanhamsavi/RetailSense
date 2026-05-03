# RetailSense

### AI-Powered Retail Demand Forecasting

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-0064E0?style=for-the-badge)](https://facebook.github.io/prophet/)
[![Claude AI](https://img.shields.io/badge/Claude_AI-Anthropic-8B5CF6?style=for-the-badge)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**[Live Demo](https://retailsense-kkjkdpqxwnmc98smxrfcol.streamlit.app)** | **[Report a Bug](https://github.com/krishnanhamsavi/RetailSense/issues)** | **[Request a Feature](https://github.com/krishnanhamsavi/RetailSense/issues)**

---

## Overview

RetailSense is a demand forecasting web application that combines Facebook Prophet time series models with Claude AI to generate analyst-grade narrative insights from real retail sales data.

A user selects any store, clicks Run Forecast, and receives a 12-week demand forecast with confidence intervals, automatic detection of anomalous historical weeks, a plain-English analyst summary written at a senior merchandising level, and a store-by-store performance comparison across 20 locations.

The project is built on the real Rossmann Store Sales dataset from Kaggle, which contains over 1 million rows of daily European pharmacy retail data.

---

## Live Demo

**App:** https://retailsense-kkjkdpqxwnmc98smxrfcol.streamlit.app

| View | Description |
|---|---|
| Store Forecast | Prophet model with confidence band, anomaly detection, and AI narrative |
| Store Comparison | All 20 stores ranked by average weekly sales with AI comparison |

---

## Features

**Time Series Forecasting** — Facebook Prophet with promotion and holiday regressors, yearly seasonality, and a configurable forecast horizon ranging from 4 to 26 weeks.

**Anomaly Detection** — Flags historical weeks where actual sales deviated more than 30% from model expectation.

**AI Analyst Narratives** — Claude AI generates executive-level summaries, anomaly explanations, and store comparison reports.

**Free Mode** — Full forecasting and visualisation works with zero API key. The AI layer is a pure enhancement and is not required to run the application.

**Interactive Charts** — Plotly charts with actual versus forecast overlay, confidence band shading, and anomaly scatter markers.

**Store Comparison Tab** — Lightweight performance overview across all 20 stores with a bar chart broken down by store type.

**Dark UI** — Custom CSS with metric cards, an analyst card with a blue border accent, and anomaly expanders.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web UI |
| Forecasting | Facebook Prophet | Weekly time series model |
| AI | Anthropic Claude API | Analyst narrative generation |
| Data Storage | SQLite | Local processed data cache |
| Visualisation | Plotly | Interactive charts |
| Data Processing | Pandas and NumPy | ETL and aggregation |
| Environment | python-dotenv | Secret management |

---

## Project Structure

```
RetailSense/
│
├── app.py                  Streamlit UI: layout, charts, and tabs
├── data_loader.py          ETL: raw CSVs to weekly SQLite aggregates
├── forecaster.py           Prophet model and anomaly detection logic
├── analyst.py              Claude AI narrative layer with free fallback
├── generate_demo_data.py   Synthetic data generator for the live demo
│
├── demo_data/              Synthetic Rossmann-style data (committed)
│   ├── train.csv
│   └── store.csv
│
├── data/                   Real Kaggle data goes here (git-ignored)
│   ├── train.csv
│   └── store.csv
│
├── .env.example            Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

**Data Flow**

```
train.csv + store.csv
        |
  data_loader.py   -->   retail.db (SQLite)
        |
  forecaster.py    -->   Prophet model + anomaly list + summary stats
        |
   analyst.py      -->   Claude AI narratives (or fallback strings)
        |
    app.py         -->   Streamlit UI rendered in browser
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/krishnanhamsavi/RetailSense.git
cd RetailSense
```

**2. Create and activate a virtual environment**

```bash
# macOS and Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your API key (optional)**

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

**5. Load the database**

Option A: Use synthetic demo data (no downloads needed)

```bash
python data_loader.py
```

Option B: Use real Rossmann data

Download `train.csv` and `store.csv` from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data), place both files in the `/data` folder, then run:

```bash
python data_loader.py
```

**6. Launch the app**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Forecasting Model

The Prophet model is configured as follows:

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
)
```

Two external regressors are added:

- `had_promo` — binary flag for promotion weeks
- `had_holiday` — binary flag for state holiday weeks

Anomaly detection compares each historical week's actual sales to the model's in-sample prediction. Weeks with greater than 30% deviation are flagged and surfaced in the UI with individual AI-generated explanations.

---

## Installation Notes for Prophet

**Windows — if `pip install prophet` fails:**

```bash
# Option A (recommended)
conda install -c conda-forge prophet

# Option B
pip install pystan==2.19.1.1 && pip install prophet
```

**Mac (Apple Silicon M1/M2):**

```bash
brew install gcc
pip install prophet
```

---

## Example Questions This App Answers

1. Which weeks should Store 5 increase inventory ahead of demand spikes?
2. Did promotions lift sales meaningfully for Type B stores?
3. Which store type has the strongest average weekly sales?
4. Are there anomalous weeks in Store 12's history worth investigating?
5. How does Store 3's 12-week forecast compare to its historical baseline?

---

## Dataset

This project uses the Rossmann Store Sales dataset published on Kaggle.

- 1,115 stores across Germany
- Approximately 2.5 years of daily sales history
- This project filters to stores 1 through 20 for manageability

Due to Kaggle's terms of use, the raw data files are not included in this repository. Download from [kaggle.com/c/rossmann-store-sales](https://www.kaggle.com/c/rossmann-store-sales/data). The app ships with synthetic demo data so it runs immediately without a Kaggle account.

---

## Author

**Hamsavi Krishnan**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR_LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/krishnanhamsavi)

---

## Acknowledgements

- [Rossmann Store Sales on Kaggle](https://www.kaggle.com/c/rossmann-store-sales) for the dataset
- [Facebook Prophet](https://facebook.github.io/prophet/) for the forecasting library
- [Anthropic Claude](https://anthropic.com) for the AI API
- [Streamlit](https://streamlit.io) for the web framework

---


