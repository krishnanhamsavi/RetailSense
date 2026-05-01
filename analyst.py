"""
analyst.py
Claude AI analyst layer with graceful free-mode fallback.
All functions return strings — the UI just renders whatever comes back.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
AI_AVAILABLE = bool(_API_KEY)

_FALLBACK = (
    "AI narrative unavailable. Add your ANTHROPIC_API_KEY to .env to unlock analyst insights."
)

_MODEL = "claude-sonnet-4-6"


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=_API_KEY)


def _call(prompt: str, max_tokens: int = 400) -> str:
    try:
        client = _client()
        message = client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        return f"{_FALLBACK} (Error: {exc})"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def generate_forecast_narrative(
    store_id: int,
    store_type: str,
    forecast_weeks: int,
    summary_stats: dict,
    anomaly_count: int,
) -> str:
    if not AI_AVAILABLE:
        return _FALLBACK

    avg = summary_stats.get("avg_weekly_sales", 0)
    peak_week = summary_stats.get("peak_week", "unknown")
    peak_sales = summary_stats.get("peak_sales", 0)
    trough_week = summary_stats.get("trough_week", "unknown")
    trough_sales = summary_stats.get("trough_sales", 0)
    total = summary_stats.get("total_forecast_sales", 0)

    prompt = f"""You are a senior retail analyst presenting a demand forecast to a merchandising VP.

Store details:
- Store ID: {store_id}
- Store Type: {store_type}
- Historical average weekly sales: €{avg:,.0f}
- Forecast horizon: {forecast_weeks} weeks
- Total forecast sales: €{total:,.0f}
- Peak forecast week: {peak_week} (€{peak_sales:,.0f})
- Trough forecast week: {trough_week} (€{trough_sales:,.0f})
- Anomalous historical weeks detected: {anomaly_count}

Write a 4–5 sentence forecast summary in a confident, flowing paragraph — no bullet points. Cover:
1. The overall demand trend
2. The peak and trough weeks with their approximate sales figures
3. Whether the anomalies are worth investigating
4. One concrete recommendation for inventory or promotional planning

Write as if presenting directly to the VP. Be specific and data-driven."""

    return _call(prompt, max_tokens=350)


def explain_anomaly(
    store_id: int,
    anomaly_date: str,
    actual: float,
    expected: float,
    pct_diff: float,
    had_promo: int,
    had_holiday: int,
) -> str:
    if not AI_AVAILABLE:
        return _FALLBACK

    direction = "above" if pct_diff > 0 else "below"
    promo_str = "a promotion was running" if had_promo else "no promotion was active"
    holiday_str = "a state holiday occurred" if had_holiday else "no state holiday was recorded"

    prompt = f"""You are a retail analyst writing a brief observation report.

Store {store_id} showed an anomaly on the week of {anomaly_date}:
- Actual sales: €{actual:,.0f}
- Model-expected sales: €{expected:,.0f}
- Deviation: {abs(pct_diff):.1f}% {direction} expectation
- Promotion flag: {promo_str}
- Holiday flag: {holiday_str}

Write exactly 2 sentences in analyst report style. Explain the likely cause referencing the promo and holiday context. Frame it as an observation, not a question."""

    return _call(prompt, max_tokens=120)


def generate_store_comparison(stores_summary_list: list[dict]) -> str:
    if not AI_AVAILABLE:
        return _FALLBACK

    if not stores_summary_list:
        return "No store data available for comparison."

    lines = []
    for s in stores_summary_list:
        lines.append(
            f"Store {s['store_id']} (Type {s['store_type']}, Assortment {s['assortment']}): "
            f"avg weekly sales €{s['avg_weekly_sales']:,.0f}, "
            f"last-4-week avg €{s['last_4wk_avg']:,.0f}"
        )

    stores_text = "\n".join(lines)

    prompt = f"""You are a senior retail analyst comparing performance across store locations.

Store summary data:
{stores_text}

Write a 3-sentence comparison paragraph. Identify which store type is forecast to perform strongest, note any meaningful patterns across assortment types or competition proximity, and give one actionable insight a category manager could act on. Analyst tone — confident and direct."""

    return _call(prompt, max_tokens=200)
