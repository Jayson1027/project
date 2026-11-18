"""Agent 1 用的市場數值：VIX 恐慌指數與美國十年期公債殖利率。"""
from datetime import date, timedelta

import pandas as pd

from core.data_pipeline import ETFDataPipeline

VIX_TICKER = "^VIX"
US10Y_TICKER = "^TNX"  # 數值即殖利率百分比，例如 4.25 代表 4.25%

VIX_LEVEL_THRESHOLD = 25.0   # VIX 達到此值視為恐慌區
VIX_JUMP_THRESHOLD = 0.20    # VIX 5 個交易日內上漲 20% 以上視為急升


def is_vix_spike(vix_latest: float, vix_change_5d: float) -> bool:
    return vix_latest >= VIX_LEVEL_THRESHOLD or vix_change_5d >= VIX_JUMP_THRESHOLD


def _change(series: pd.Series, periods: int) -> float:
    periods = min(periods, len(series) - 1)
    return float(series.iloc[-1] - series.iloc[-1 - periods]) if periods > 0 else 0.0


def fetch_market_snapshot(lookback_days: int = 45) -> dict:
    """抓最近一段時間的 VIX 與十年期殖利率，整理成最新值與 5 日、20 日變化。"""
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    prices = ETFDataPipeline(
        asset=[VIX_TICKER, US10Y_TICKER],
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    ).fetch_data()
    if prices is None or len(prices) < 2:
        raise RuntimeError("抓不到足夠的 VIX／十年期殖利率資料")

    vix, us10y = prices[VIX_TICKER], prices[US10Y_TICKER]
    vix_latest = float(vix.iloc[-1])
    vix_change_5d = _change(vix, 5) / float(vix.iloc[-1 - min(5, len(vix) - 1)])
    vix_change_20d = _change(vix, 20) / float(vix.iloc[-1 - min(20, len(vix) - 1)])

    return {
        "as_of": prices.index[-1].strftime("%Y-%m-%d"),
        "vix": {
            "latest": round(vix_latest, 2),
            "change_5d_pct": round(vix_change_5d * 100, 1),
            "change_20d_pct": round(vix_change_20d * 100, 1),
        },
        "us10y_yield": {
            "latest_pct": round(float(us10y.iloc[-1]), 3),
            "change_5d_bp": round(_change(us10y, 5) * 100, 1),
            "change_20d_bp": round(_change(us10y, 20) * 100, 1),
        },
        "vix_spike": is_vix_spike(vix_latest, vix_change_5d),
    }


def format_market_data(snapshot: dict) -> str:
    vix, y = snapshot["vix"], snapshot["us10y_yield"]
    return (
        f"資料日期：{snapshot['as_of']}\n"
        f"VIX 恐慌指數：{vix['latest']}（5 日 {vix['change_5d_pct']:+}%，20 日 {vix['change_20d_pct']:+}%）\n"
        f"美國十年期公債殖利率：{y['latest_pct']}%（5 日 {y['change_5d_bp']:+} bp，20 日 {y['change_20d_bp']:+} bp）"
    )
