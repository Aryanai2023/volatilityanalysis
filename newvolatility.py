from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

import yfinance as yf


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Volatility Pulse", version="1.0")


# -----------------------------
# Simple TTL cache (avoids hammering Yahoo)
# -----------------------------
@dataclass
class CacheItem:
    ts: float
    value: Any


_CACHE: Dict[str, CacheItem] = {}
CACHE_TTL_SECONDS = 60  # refresh at most once per minute


def cache_get(key: str) -> Optional[Any]:
    item = _CACHE.get(key)
    if not item:
        return None
    if time.time() - item.ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return item.value


def cache_set(key: str, value: Any) -> None:
    _CACHE[key] = CacheItem(ts=time.time(), value=value)


# -----------------------------
# Core quant functions
# -----------------------------
def _annualize(daily_std: pd.Series, trading_days: int) -> pd.Series:
    return daily_std * np.sqrt(trading_days)


def _log_returns(price: pd.Series) -> pd.Series:
    r = np.log(price / price.shift(1))
    return r.dropna()


def _ewma_ann_vol(returns: pd.Series, lam: float, trading_days: int) -> pd.Series:
    r2 = returns.pow(2)
    var = np.zeros(len(r2), dtype=float)
    var[0] = float(r2.iloc[0])

    for i in range(1, len(r2)):
        var[i] = lam * var[i - 1] + (1 - lam) * float(r2.iloc[i - 1])

    ewma_daily = pd.Series(np.sqrt(var), index=returns.index)
    ewma_ann = ewma_daily * np.sqrt(trading_days)
    ewma_ann.name = f"ewma_ann_vol_lam{lam}"
    return ewma_ann


def _rolling_ann_vol(returns: pd.Series, windows: List[int], trading_days: int) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"roll_{w}d_ann_vol"] = _annualize(returns.rolling(w).std(), trading_days)
    return pd.DataFrame(out)


def _percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if len(s) < 30:
        return float("nan")
    return float((s <= value).mean())


def _regime(p: float) -> Tuple[str, str]:
    """
    p = percentile of current vol within its own history.
    """
    if np.isnan(p):
        return ("❓", "UNKNOWN")
    if p < 0.33:
        return ("☀️", "LOW")
    if p < 0.66:
        return ("🌤️", "MEDIUM")
    if p < 0.90:
        return ("🌧️", "HIGH")
    return ("🌪️", "EXTREME")


def fetch_price_series(ticker: str, period: str = "2y", interval: str = "1d") -> pd.Series:
    t = ticker.strip().upper()
    key = f"px::{t}::{period}::{interval}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for ticker '{t}'. Try another ticker/period/interval.")

    if "Close" not in df.columns:
        # fallback
        px = df.iloc[:, 0].dropna()
    else:
        px = df["Close"].dropna()

    px.name = t
    cache_set(key, px)
    return px


def compute_metrics(
    ticker: str,
    period: str,
    interval: str,
    trading_days: int,
    windows: List[int],
    lam: float,
) -> Dict[str, Any]:
    px = fetch_price_series(ticker, period=period, interval=interval)
    rets = _log_returns(px)

    if len(rets) < max(windows + [20]):
        # still compute what we can
        pass

    roll = _rolling_ann_vol(rets, windows=windows, trading_days=trading_days)
    ewma = _ewma_ann_vol(rets, lam=lam, trading_days=trading_days)

    # pick "current vol" as 20d rolling if present, else last available window
    current_vol_series = roll.get("roll_20d_ann_vol")
    if current_vol_series is None:
        # fallback: use smallest window present
        smallest = min(windows) if windows else 20
        current_vol_series = roll.get(f"roll_{smallest}d_ann_vol")

    latest_price = float(px.iloc[-1])
    latest_ret = float(rets.iloc[-1]) if len(rets) else float("nan")

    # latest rolling vols
    latest_roll = {c: float(roll[c].dropna().iloc[-1]) if roll[c].notna().any() else float("nan") for c in roll.columns}
    latest_ewma = float(ewma.dropna().iloc[-1]) if ewma.notna().any() else float("nan")

    # regime based on percentile of current vol over its own history
    p = float("nan")
    current_vol = float("nan")
    if current_vol_series is not None and current_vol_series.notna().any():
        current_vol = float(current_vol_series.dropna().iloc[-1])
        p = _percentile_rank(current_vol_series, current_vol)

    icon, label = _regime(p)

    # shock score: current 20d vol vs median of 252d vol (or vs median of same series if 252 not available)
    baseline_series = roll.get("roll_252d_ann_vol")
    if baseline_series is None or not baseline_series.notna().any():
        baseline_series = current_vol_series
    baseline = float(baseline_series.dropna().median()) if baseline_series is not None and baseline_series.notna().any() else float("nan")

    shock = float(current_vol / baseline) if (baseline and not np.isnan(current_vol) and not np.isnan(baseline) and baseline != 0) else float("nan")

    # trend arrow: compare 20d vs 60d if available
    trend = "→"
    v20 = latest_roll.get("roll_20d_ann_vol", float("nan"))
    v60 = latest_roll.get("roll_60d_ann_vol", float("nan"))
    if not np.isnan(v20) and not np.isnan(v60):
        if v20 > v60 * 1.10:
            trend = "↑"
        elif v20 < v60 * 0.90:
            trend = "↓"

    return {
        "ticker": px.name,
        "price": latest_price,
        "last_log_return": latest_ret,
        "ewma_ann_vol": latest_ewma,
        "regime_icon": icon,
        "regime": label,
        "vol_percentile": p,
        "shock": shock,
        "trend": trend,
        **latest_roll,
    }


def series_payload(
    ticker: str,
    period: str,
    interval: str,
    trading_days: int,
    windows: List[int],
    lam: float,
) -> Dict[str, Any]:
    px = fetch_price_series(ticker, period=period, interval=interval)
    rets = _log_returns(px)
    roll = _rolling_ann_vol(rets, windows=windows, trading_days=trading_days)
    ewma = _ewma_ann_vol(rets, lam=lam, trading_days=trading_days)

    # reduce payload size
    idx = px.index.astype(str).tolist()
    out: Dict[str, Any] = {
        "ticker": px.name,
        "dates": idx,
        "price": px.values.tolist(),
        "ewma": ewma.reindex(px.index).values.tolist(),  # align to price index
    }
    for c in roll.columns:
        out[c] = roll.reindex(px.index).values.tolist()
    return out


# -----------------------------
# API
# -----------------------------
@app.get("/api/radar")
def api_radar(
    tickers: str = Query(..., description="Comma-separated tickers e.g. AAPL,MSFT,BTC-USD"),
    period: str = Query("2y"),
    interval: str = Query("1d"),
    trading_days: int = Query(252, ge=1, le=365),
    lam: float = Query(0.94, ge=0.8, le=0.99),
    windows: str = Query("20,60,252"),
):
    ws = [int(x.strip()) for x in windows.split(",") if x.strip().isdigit()]
    tk = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tk:
        return JSONResponse({"error": "No tickers provided."}, status_code=400)

    results = []
    errors = []

    for t in tk:
        try:
            results.append(compute_metrics(t, period, interval, trading_days, ws, lam))
        except Exception as e:
            errors.append({"ticker": t, "error": str(e)})

    # sort by shock desc (most “interesting” first)
    results.sort(key=lambda d: (np.nan_to_num(d.get("shock", np.nan), nan=-1.0)), reverse=True)
    return {"results": results, "errors": errors}


@app.get("/api/series")
def api_series(
    ticker: str = Query(...),
    period: str = Query("2y"),
    interval: str = Query("1d"),
    trading_days: int = Query(252, ge=1, le=365),
    lam: float = Query(0.94, ge=0.8, le=0.99),
    windows: str = Query("20,60,252"),
):
    ws = [int(x.strip()) for x in windows.split(",") if x.strip().isdigit()]
    try:
        payload = series_payload(ticker, period, interval, trading_days, ws, lam)
        return payload
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# -----------------------------
# UI (no frontend build tools)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Volatility Pulse</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { color-scheme: dark; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
           margin: 0; padding: 18px; background: #0b0f17; color: #e7eefc; }
    .wrap { max-width: 1200px; margin: 0 auto; }
    h1 { margin: 0 0 6px 0; font-size: 28px; }
    .muted { color: #9ab0d0; margin: 0 0 16px 0; }
    .card { background: #11182a; border: 1px solid #1e2a46; border-radius: 14px; padding: 14px; }
    .grid { display: grid; grid-template-columns: 1.1fr 1.9fr; gap: 12px; align-items: start; }
    .controls { display: grid; gap: 10px; }
    label { font-size: 12px; color: #bcd1f6; }
    input, select, textarea, button {
      width: 100%; padding: 10px 10px; border-radius: 10px;
      border: 1px solid #24335a; background: #0c1322; color: #e7eefc;
      outline: none;
    }
    textarea { min-height: 72px; resize: vertical; }
    button { cursor: pointer; border: 1px solid #2d4cff; background: #1730ff; font-weight: 700; }
    button:hover { filter: brightness(1.05); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid #1f2b4a; text-align: left; }
    th { color: #bcd1f6; font-weight: 700; position: sticky; top: 0; background: #11182a; }
    tr:hover { background: #0c1322; }
    .pill { display: inline-flex; gap: 6px; align-items: center; padding: 4px 10px; border-radius: 999px;
            border: 1px solid #24335a; background: #0c1322; }
    .rowclick { cursor: pointer; }
    .small { font-size: 12px; color: #9ab0d0; }
    .chart { height: 520px; }
    .err { color: #ff9aa7; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>⚡ Volatility Pulse</h1>
    <p class="muted">A watchlist-style volatility radar with regimes, shock scores, and interactive charts (FastAPI + Yahoo Finance).</p>

    <div class="grid">
      <div class="card controls">
        <div>
          <label>Tickers (comma-separated)</label>
          <textarea id="tickers">AAPL,MSFT,NVDA,TSLA,SPY,BTC-USD,ETH-USD</textarea>
          <div class="small">Examples: AAPL, MSFT, SPY, QQQ, BTC-USD</div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
          <div>
            <label>Period</label>
            <select id="period">
              <option>6mo</option>
              <option selected>2y</option>
              <option>5y</option>
              <option>10y</option>
              <option>max</option>
            </select>
          </div>
          <div>
            <label>Interval</label>
            <select id="interval">
              <option selected>1d</option>
              <option>1wk</option>
              <option>1mo</option>
            </select>
          </div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
          <div>
            <label>Annualization days</label>
            <select id="td">
              <option selected value="252">252 (stocks)</option>
              <option value="365">365 (crypto)</option>
            </select>
          </div>
          <div>
            <label>EWMA λ</label>
            <input id="lam" type="number" step="0.01" min="0.80" max="0.99" value="0.94"/>
          </div>
        </div>

        <div>
          <label>Rolling windows (days)</label>
          <input id="windows" value="20,60,252"/>
          <div class="small">Try: 10,20,60 or 20,60,120,252</div>
        </div>

        <button id="run">Run Radar</button>
        <div id="err" class="err"></div>
      </div>

      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
          <div>
            <div class="small">Leaderboard (sorted by shock score)</div>
            <div class="small">Click a row to load its chart.</div>
          </div>
          <div class="pill" id="selectedPill">Selected: —</div>
        </div>
        <div style="max-height: 260px; overflow:auto; margin-top: 10px;">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Regime</th>
                <th>Trend</th>
                <th>EWMA</th>
                <th>Roll 20</th>
                <th>Roll 60</th>
                <th>Roll 252</th>
                <th>Shock</th>
                <th>Price</th>
              </tr>
            </thead>
            <tbody id="tbody"></tbody>
          </table>
        </div>

        <div id="chart" class="chart"></div>
      </div>
    </div>
  </div>

<script>
  const fmtPct = (x) => (isFinite(x) ? (x*100).toFixed(1) + "%" : "—");
  const fmtNum = (x) => (isFinite(x) ? x.toFixed(2) : "—");

  async function loadRadar() {
    const tickers = document.getElementById("tickers").value.trim();
    const period = document.getElementById("period").value;
    const interval = document.getElementById("interval").value;
    const td = document.getElementById("td").value;
    const lam = document.getElementById("lam").value;
    const windows = document.getElementById("windows").value.trim();
    const err = document.getElementById("err");
    err.textContent = "";

    const url = `/api/radar?tickers=${encodeURIComponent(tickers)}&period=${encodeURIComponent(period)}&interval=${encodeURIComponent(interval)}&trading_days=${td}&lam=${lam}&windows=${encodeURIComponent(windows)}`;

    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      err.textContent = data.error || "Radar failed.";
      return;
    }

    const tbody = document.getElementById("tbody");
    tbody.innerHTML = "";

    if (data.errors && data.errors.length) {
      err.textContent = "Some tickers failed: " + data.errors.map(e => e.ticker).join(", ");
    }

    const results = data.results || [];
    if (!results.length) {
      err.textContent = "No results. Try different tickers.";
      return;
    }

    results.forEach((r, idx) => {
      const tr = document.createElement("tr");
      tr.className = "rowclick";
      tr.onclick = () => loadSeries(r.ticker);

      const roll20 = r.roll_20d_ann_vol ?? NaN;
      const roll60 = r.roll_60d_ann_vol ?? NaN;
      const roll252 = r.roll_252d_ann_vol ?? NaN;

      tr.innerHTML = `
        <td><b>${r.ticker}</b></td>
        <td>${r.regime_icon} ${r.regime}</td>
        <td>${r.trend || "→"}</td>
        <td>${fmtPct(r.ewma_ann_vol)}</td>
        <td>${fmtPct(roll20)}</td>
        <td>${fmtPct(roll60)}</td>
        <td>${fmtPct(roll252)}</td>
        <td>${isFinite(r.shock) ? r.shock.toFixed(2) : "—"}</td>
        <td>${fmtNum(r.price)}</td>
      `;
      tbody.appendChild(tr);

      if (idx === 0) {
        loadSeries(r.ticker); // auto-load the most "interesting"
      }
    });
  }

  async function loadSeries(ticker) {
    const period = document.getElementById("period").value;
    const interval = document.getElementById("interval").value;
    const td = document.getElementById("td").value;
    const lam = document.getElementById("lam").value;
    const windows = document.getElementById("windows").value.trim();
    document.getElementById("selectedPill").textContent = "Selected: " + ticker;

    const url = `/api/series?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}&interval=${encodeURIComponent(interval)}&trading_days=${td}&lam=${lam}&windows=${encodeURIComponent(windows)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      document.getElementById("err").textContent = data.error || "Series failed.";
      return;
    }

    const dates = data.dates;
    const price = data.price;
    const ewma = data.ewma;
    const roll20 = data.roll_20d_ann_vol;
    const roll60 = data.roll_60d_ann_vol;

    const traces = [
      { x: dates, y: price, name: "Price", yaxis: "y1", type: "scatter", mode: "lines" },
      { x: dates, y: ewma,  name: "EWMA ann vol", yaxis: "y2", type: "scatter", mode: "lines" },
    ];

    if (roll20) traces.push({ x: dates, y: roll20, name: "Roll 20 ann vol", yaxis: "y2", type: "scatter", mode: "lines" });
    if (roll60) traces.push({ x: dates, y: roll60, name: "Roll 60 ann vol", yaxis: "y2", type: "scatter", mode: "lines" });

    const layout = {
      title: `${data.ticker} — Price + Volatility`,
      paper_bgcolor: "#11182a",
      plot_bgcolor: "#11182a",
      font: { color: "#e7eefc" },
      margin: { l: 50, r: 50, t: 40, b: 40 },
      xaxis: { showgrid: false },
      yaxis:  { title: "Price", side: "left", showgrid: true, gridcolor: "#1f2b4a" },
      yaxis2: { title: "Annualized Vol", overlaying: "y", side: "right", tickformat: ".0%", showgrid: false },
      legend: { orientation: "h", y: -0.2 }
    };

    Plotly.newPlot("chart", traces, layout, {displayModeBar: false, responsive: true});
  }

  document.getElementById("run").onclick = loadRadar;
  loadRadar();
</script>
</body>
</html>
"""
    return HTMLResponse(html)
