# RetailSense — AI-Powered Retail Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Prophet](https://img.shields.io/badge/Prophet-Meta-orange?style=flat-square)
![Claude AI](https://img.shields.io/badge/Claude_AI-Optional-8b5cf6?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-ff4b4b?style=flat-square&logo=streamlit)
![Rossmann Data](https://img.shields.io/badge/Data-Rossmann_Kaggle-20b2aa?style=flat-square)

> A demand forecasting web app that combines Facebook Prophet time series models with Claude AI to generate analyst-grade narrative insights from real Rossmann retail sales data.

---

## What It Does

RetailSense ingests real-world Rossmann store sales data, aggregates it to weekly granularity, and fits a Prophet model to generate 12-week demand forecasts per store. It automatically detects historical anomalies — weeks where actual sales deviated meaningfully from expected — and surfaces them with visual callouts. When an Anthropic API key is provided, the Claude AI layer writes analyst-grade narrative summaries and anomaly explanations exactly as a senior merchandising analyst would present them to a VP.

---

## Live Demo

Run locally — see setup below.

---

## Screenshots

> Add screenshots after running: sidebar with store selector, forecast chart with confidence band, anomaly expanders, and AI analyst card.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **Facebook Prophet** | Weekly time series forecasting |
| **SQLite** | Lightweight local data storage |
| **Anthropic Claude API** | AI analyst narratives (optional) |
| **Streamlit** | Interactive web UI |
| **Plotly** | Interactive charts |
| **Pandas** | Data wrangling |
| **python-dotenv** | Environment variable management |

---

## Setup Instructions

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RetailSense.git
cd RetailSense
```

### Step 2 — Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Prophet installation note** — see the [troubleshooting section](#prophet-installation-notes) below if you hit errors.

### Step 4 — Download the Rossmann dataset

1. Go to [kaggle.com/c/rossmann-store-sales/data](https://www.kaggle.com/c/rossmann-store-sales/data)
2. Download `train.csv` and `store.csv`
3. Place both files inside the `/data` folder:

```
RetailSense/
└── data/
    ├── train.csv
    └── store.csv
```

### Step 5 — Add your API key (optional)

```bash
cp .env.example .env
```

Edit `.env` and paste your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Leave this blank to run in free mode — all forecasting still works, only AI narratives are disabled.

### Step 6 — Load the database

```bash
python data_loader.py
```

You should see: `Database loaded: X rows across 20 stores.`

### Step 7 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Free Mode vs AI Mode

| Feature | Free Mode | AI Mode |
|---------|-----------|---------|
| Prophet forecasting | ✅ | ✅ |
| Interactive Plotly charts | ✅ | ✅ |
| Anomaly detection | ✅ | ✅ |
| Store comparison table | ✅ | ✅ |
| AI forecast narrative | ❌ | ✅ |
| AI anomaly explanations | ❌ | ✅ |
| AI store comparison paragraph | ❌ | ✅ |

**Free mode is fully functional** — you get the complete forecasting engine, charts, and anomaly detection. The AI layer is a pure enhancement layer.

---

## What Makes This Different

- **Real production data** — Uses the actual Rossmann Kaggle dataset (1M+ rows), not synthetic CSVs
- **Production-grade forecasting model** — Facebook Prophet with custom regressors (promo flags, holiday flags) and automatic anomaly detection
- **AI narrative layer** — Claude generates analyst-grade summaries written for executive audiences, not technical users
- **Analyst-style output** — Every insight is framed as a business observation, not a model output — the kind of commentary a senior analyst would send to a VP

---

## Example Questions This App Answers

1. *Which weeks should Store 5 increase inventory ahead of forecast demand spikes?*
2. *Did promotions meaningfully lift sales for Type B stores in 2014?*
3. *Which store type has the strongest average weekly sales trajectory?*
4. *Are there anomalous weeks in Store 12's history that warrant an ops review?*
5. *How does Store 3's forecast compare to its historical average?*

---

## Prophet Installation Notes

**Windows:** If `pip install prophet` fails, try:
```bash
pip install pystan==2.19.1.1
pip install prophet
```

Or use conda:
```bash
conda install -c conda-forge prophet
```

**Mac (Apple Silicon / M1/M2):** If you see a compiler error:
```bash
brew install gcc
pip install prophet
```

If that still fails:
```bash
conda install -c conda-forge prophet
```

---

## Project Structure

```
RetailSense/
├── app.py              ← Main Streamlit app (start here to understand the UI)
├── data_loader.py      ← ETL: Rossmann CSVs → SQLite
├── forecaster.py       ← Prophet model + anomaly detection
├── analyst.py          ← Claude AI narrative layer
├── requirements.txt
├── .env.example
├── .gitignore
└── data/               ← Add train.csv + store.csv here (git-ignored)
```

**Files to read first:**
1. `forecaster.py` — core forecasting logic
2. `analyst.py` — AI integration with graceful fallback
3. `app.py` — how it all connects in the UI

---

## Author

Built as a portfolio project demonstrating AI-powered analytics with real retail data.

- **LinkedIn:** [Add your LinkedIn URL]
- **GitHub:** [Add your GitHub URL]

---

*This project was built as a portfolio demonstration of combining classical time series forecasting (Prophet) with modern LLM-powered analysis (Claude AI) in a production-style Streamlit application.*
