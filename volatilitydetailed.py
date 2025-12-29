# app.py
# 📈 Volatility Analysis Dashboard (Streamlit)
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class Annualization:
    periods_per_year: float
    label: str


def _is_probably_crypto(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    return t.endswith("-USD") or t.endswith("-USDT") or t.endswith("-BTC") or t.endswith("-ETH")


def annualization_for(interval: str, asset_type: str, ticker: str) -> Annualization:
    """
    Returns periods-per-year used to annualize volatility:
        annual_vol = std(returns) * sqrt(periods_per_year)
    """
    interval = (interval or "1d").lower().strip()
    asset_type = (asset_type or "auto").lower().strip()

    if asset_type == "auto":
        asset_type = "crypto" if _is_probably_crypto(ticker) else "stock"

    # Common intervals supported by yfinance: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # We'll map to reasonable period counts.
    if asset_type == "crypto":
        # Crypto trades 24/7
        if interval in {"1m", "2m", "5m", "15m", "30m"}:
            minutes = int(interval.replace("m", ""))
            periods = (365 * 24 * 60) / minutes
            return Annualization(periods_per_year=periods, label=f"{interval} (24/7)")
        if interval in {"60m", "1h"}:
            return Annualization(periods_per_year=365 * 24, label=f"{interval} (24/7)")
        if interval == "90m":
            return Annualization(periods_per_year=(365 * 24) / 1.5, label=f"{interval} (24/7)")
        if interval == "1d":
            return Annualization(periods_per_year=365, label="Daily (crypto)")
        if interval == "5d":
            return Annualization(periods_per_year=365 / 5, label="5-Day (crypto)")
        if interval == "1wk":
            return Annualization(periods_per_year=52, label="Weekly (crypto)")
        if interval == "1mo":
            return Annualization(periods_per_year=12, label="Monthly (crypto)")
        if interval == "3mo":
            return Annualization(periods_per_year=4, label="Quarterly (crypto)")

    # Stocks/ETFs/Forex (as a proxy): use trading days for daily; for intraday approximate trading hours
    # Trading days/year ~ 252; trading hours/day ~ 6.5
    if interval in {"1m", "2m", "5m", "15m", "30m"}:
        minutes = int(interval.replace("m", ""))
        periods = (252 * 6.5 * 60) / minutes
        return Annualization(periods_per_year=periods, label=f"{interval} (market hours approx.)")
    if interval in {"60m", "1h"}:
        return Annualization(periods_per_year=252 * 6.5, label=f"{interval} (market hours approx.)")
    if interval == "90m":
        return Annualization(periods_per_year=(252 * 6.5) / 1.5, label=f"{interval} (market hours approx.)")
    if interval == "1d":
        return Annualization(periods_per_year=252, label="Daily (stocks)")
    if interval == "5d":
        return Annualization(periods_per_year=252 / 5, label="5-Day (stocks)")
    if interval == "1wk":
        return Annualization(periods_per_year=52, label="Weekly (stocks)")
    if interval == "1mo":
        return Annualization(periods_per_year=12, label="Monthly (stocks)")
    if interval == "3mo":
        return Annualization(periods_per_year=4, label="Quarterly (stocks)")

    # Fallback
    return Annualization(periods_per_year=252, label="Daily (fallback)")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance via yfinance.
    """
    ticker = ticker.strip()
    # yfinance end is exclusive-ish; add 1 day to include end date for daily data
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(
        tickers=ticker,
        start=pd.to_datetime(start),
        end=end_dt,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize columns
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        # Sometimes yfinance returns multiindex columns for multiple tickers
        df.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in df.columns]
    return df


def compute_returns(price_series: pd.Series) -> pd.Series:
    # log returns
    px = price_series.dropna()
    rets = np.log(px / px.shift(1))
    return rets.dropna()


def rolling_volatility(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    vol = returns.rolling(window).std(ddof=0) * math.sqrt(periods_per_year)
    return vol


def ewma_volatility(returns: pd.Series, lam: float, periods_per_year: float) -> pd.Series:
    """
    RiskMetrics-style EWMA volatility on returns:
      sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return pd.Series(dtype=float)

    sigma2 = np.empty(len(r), dtype=float)
    # initialize with unconditional variance estimate
    sigma2[0] = np.nanvar(r.values)
    for i in range(1, len(r)):
        sigma2[i] = lam * sigma2[i - 1] + (1 - lam) * (r.values[i - 1] ** 2)

    sigma = np.sqrt(sigma2) * math.sqrt(periods_per_year)
    return pd.Series(sigma, index=r.index, name="EWMA Vol (ann.)")


def summary_stats(returns: pd.Series, periods_per_year: float) -> pd.DataFrame:
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()

    mean = r.mean()
    std = r.std(ddof=0)
    ann_mean = mean * periods_per_year
    ann_vol = std * math.sqrt(periods_per_year)

    # downside / risk measures
    var_95 = np.quantile(r, 0.05)
    cvar_95 = r[r <= var_95].mean() if (r <= var_95).any() else np.nan

    # skew/kurt (excess kurtosis using pandas)
    skew = r.skew()
    kurt = r.kurt()

    out = pd.DataFrame(
        {
            "Metric": [
                "Observations",
                "Mean (per period)",
                "Volatility (per period)",
                "Annualized mean (approx.)",
                "Annualized volatility",
                "Min return",
                "Max return",
                "VaR 95% (5th pct)",
                "CVaR 95% (avg <= VaR)",
                "Skewness",
                "Excess kurtosis",
            ],
            "Value": [
                int(r.shape[0]),
                float(mean),
                float(std),
                float(ann_mean),
                float(ann_vol),
                float(r.min()),
                float(r.max()),
                float(var_95),
                float(cvar_95),
                float(skew),
                float(kurt),
            ],
        }
    )
    return out


def format_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x*100:.4f}%"


def format_num(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.8f}"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Volatility Analysis", layout="wide")
st.title("📈 Volatility Analysis Dashboard")

with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker", value="AAPL", help="Examples: AAPL, MSFT, SPY, BTC-USD, ETH-USD")
    interval = st.selectbox(
        "Interval",
        options=["1d", "1h", "60m", "30m", "15m", "5m", "1wk", "1mo"],
        index=0,
    )

    asset_type = st.selectbox(
        "Asset type",
        options=["Auto", "Stock/ETF", "Crypto"],
        index=0,
        help="Used for annualization assumptions.",
    )
    asset_type_norm = {"auto": "auto", "stock/etf": "stock", "crypto": "crypto"}[asset_type.lower()]

    today = date.today()
    default_start = today - timedelta(days=365 * 2)

    start = st.date_input("Start date", value=default_start)
    end = st.date_input("End date", value=today)

    st.divider()
    st.subheader("Volatility settings")

    rolling_windows = st.multiselect(
        "Rolling windows (periods)",
        options=[5, 10, 20, 30, 60, 90, 126, 252],
        default=[20, 60, 252],
        help="Window length is in number of rows at the selected interval.",
    )

    lam = st.slider("EWMA λ (lambda)", min_value=0.85, max_value=0.995, value=0.94, step=0.005)

    st.divider()
    st.subheader("Display")
    show_hist = st.checkbox("Show returns histogram", value=True)
    show_drawdown = st.checkbox("Show drawdown chart", value=True)

# Basic validation
if not ticker.strip():
    st.warning("Please enter a ticker.")
    st.stop()

if start >= end:
    st.warning("Start date must be before end date.")
    st.stop()

ann = annualization_for(interval=interval, asset_type=asset_type_norm, ticker=ticker)
st.caption(f"Annualization assumption: **{ann.label}** → periods/year ≈ **{ann.periods_per_year:,.2f}**")

with st.spinner("Fetching data from Yahoo Finance..."):
    df = fetch_prices(ticker=ticker, start=start, end=end, interval=interval)

if df.empty:
    st.error("No data returned. Try a different ticker, interval, or date range.")
    st.stop()

# Choose a price column
price_col_candidates = [c for c in df.columns if str(c).lower() in {"close", "adj close", "adj_close"}]
if price_col_candidates:
    price_col = price_col_candidates[0]
else:
    # yfinance with auto_adjust typically provides "Close"
    # fallback: pick any column containing "close"
    close_like = [c for c in df.columns if "close" in str(c).lower()]
    price_col = close_like[0] if close_like else df.columns[0]

prices = df[price_col].dropna().rename("Price")
returns = compute_returns(prices).rename("Log return")

# Rolling vols
rolling_vols = {}
for w in sorted(set(rolling_windows)):
    if w >= 2:
        rolling_vols[w] = rolling_volatility(returns, window=w, periods_per_year=ann.periods_per_year).rename(
            f"Rolling Vol {w} (ann.)"
        )

ewma_vol = ewma_volatility(returns, lam=lam, periods_per_year=ann.periods_per_year)

# Drawdown (from prices)
def compute_drawdown(px: pd.Series) -> pd.Series:
    p = px.dropna()
    peak = p.cummax()
    dd = (p / peak) - 1.0
    return dd.rename("Drawdown")

drawdown = compute_drawdown(prices)

# Summary stats
stats = summary_stats(returns, periods_per_year=ann.periods_per_year)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.subheader("Price")
    fig = plt.figure()
    plt.plot(prices.index, prices.values)
    plt.xlabel("Date")
    plt.ylabel("Price (adjusted)")
    plt.title(f"{ticker.upper()} — Price ({interval})")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Returns (log)")
    fig = plt.figure()
    plt.plot(returns.index, returns.values)
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.title(f"{ticker.upper()} — Log Returns ({interval})")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Volatility (annualized)")
    fig = plt.figure()
    for w, series in rolling_vols.items():
        plt.plot(series.index, series.values, label=series.name)
    if not ewma_vol.empty:
        plt.plot(ewma_vol.index, ewma_vol.values, label=ewma_vol.name)
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.title(f"{ticker.upper()} — Volatility ({interval})")
    if rolling_vols or not ewma_vol.empty:
        plt.legend()
    st.pyplot(fig, clear_figure=True)

    if show_drawdown:
        st.subheader("Drawdown")
        fig = plt.figure()
        plt.plot(drawdown.index, drawdown.values)
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.title(f"{ticker.upper()} — Drawdown")
        st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Summary")
    if stats.empty:
        st.info("Not enough data to compute statistics.")
    else:
        # Format a nicer display
        stats_display = stats.copy()
        pct_metrics = {
            "Mean (per period)",
            "Volatility (per period)",
            "Annualized mean (approx.)",
            "Annualized volatility",
            "Min return",
            "Max return",
            "VaR 95% (5th pct)",
            "CVaR 95% (avg <= VaR)",
        }
        stats_display["Value"] = stats_display.apply(
            lambda row: format_pct(row["Value"]) if row["Metric"] in pct_metrics else str(row["Value"]),
            axis=1,
        )
        st.dataframe(stats_display, use_container_width=True, hide_index=True)

    if show_hist:
        st.subheader("Returns histogram")
        fig = plt.figure()
        plt.hist(returns.values, bins=50)
        plt.xlabel("Log return")
        plt.ylabel("Frequency")
        plt.title(f"{ticker.upper()} — Returns Distribution")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Download")
    # Build an export dataframe
    out = pd.DataFrame({"price": prices, "log_return": returns})
    for w, series in rolling_vols.items():
        out[f"rolling_vol_{w}_ann"] = series
    if not ewma_vol.empty:
        out["ewma_vol_ann"] = ewma_vol
    out["drawdown"] = drawdown

    csv = out.reset_index().rename(columns={"index": "timestamp"}).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv,
        file_name=f"{ticker.upper()}_volatility_{interval}.csv",
        mime="text/csv",
    )

st.subheader("Raw data preview")
st.dataframe(df.tail(25), use_container_width=True)

st.caption(
    "Notes: Yahoo Finance data can be delayed or adjusted; annualization is an approximation based on interval and asset type."
)
