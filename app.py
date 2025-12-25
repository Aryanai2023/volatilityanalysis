import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

st.set_page_config(page_title="Volatility Analysis (Any Stock)", layout="wide")
st.title("📈 Volatility Analysis (Any Stock / Crypto)")
st.caption("Type a ticker → fetch prices → returns → rolling & EWMA volatility → summary")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_prices_yahoo(ticker: str, start: str, end: str, interval: str) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    t = ticker.strip().upper()
    df = yf.download(
        t,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,   # adjusted prices
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{t}'. Try another ticker or interval.")

    # Prefer Close if present
    if "Close" in df.columns:
        s = df["Close"].copy()
    else:
        # Some intervals/data sources may behave differently
        s = df.iloc[:, 0].copy()

    s = s.dropna()
    s.name = t
    return s

def log_returns(price: pd.Series) -> pd.Series:
    r = np.log(price / price.shift(1))
    return r.dropna()

def annualize_vol(daily_vol: pd.Series, trading_days: int) -> pd.Series:
    return daily_vol * np.sqrt(trading_days)

def rolling_vol(returns: pd.Series, windows: list[int], trading_days: int) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"roll_{w}d_ann_vol"] = annualize_vol(returns.rolling(w).std(), trading_days)
    return pd.DataFrame(out)

def ewma_vol(returns: pd.Series, lam: float, trading_days: int) -> pd.Series:
    r2 = returns**2
    var = np.zeros(len(r2))
    var[0] = float(r2.iloc[0])

    for i in range(1, len(r2)):
        var[i] = lam * var[i - 1] + (1 - lam) * float(r2.iloc[i - 1])

    ewma_daily = pd.Series(np.sqrt(var), index=returns.index)
    ewma_ann = annualize_vol(ewma_daily, trading_days)
    ewma_ann.name = f"ewma_ann_vol_lam{lam}"
    return ewma_ann

def summary_stats(returns: pd.Series, trading_days: int) -> pd.Series:
    daily_std = returns.std()
    stats = {
        "count_days": int(returns.shape[0]),
        "mean_daily_return": float(returns.mean()),
        "std_daily_return": float(daily_std),
        "ann_volatility": float(daily_std * np.sqrt(trading_days)),
        "min_daily_return": float(returns.min()),
        "5pct_daily_return": float(returns.quantile(0.05)),
        "95pct_daily_return": float(returns.quantile(0.95)),
        "max_daily_return": float(returns.max()),
    }
    return pd.Series(stats)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Ticker & Data")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL", help="Examples: AAPL, MSFT, TSLA, NVDA, BTC-USD, ETH-USD")
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.header("Date range")
    start = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("today"))

    st.header("Volatility settings")
    trading_days = st.selectbox("Annualization days", [252, 365], index=0, help="Stocks: 252, Crypto: 365")
    windows = st.multiselect("Rolling windows (days)", [5, 10, 20, 30, 60, 90, 120, 252], default=[20, 60, 252])
    lam = st.slider("EWMA lambda (λ)", 0.80, 0.99, 0.94, 0.01)

    run = st.button("Run analysis")

if not run:
    st.info("Set inputs on the left and click **Run analysis**.")
    st.stop()

# -----------------------------
# Fetch and compute
# -----------------------------
try:
    price = fetch_prices_yahoo(ticker, str(start), str(end), interval)
except Exception as e:
    st.error(f"Data fetch failed: {e}")
    if yf is None:
        st.code("pip install yfinance", language="bash")
    st.stop()

rets = log_returns(price)

if rets.shape[0] < 10:
    st.warning("Very few return observations. Try a wider date range or daily interval (1d).")

roll_df = rolling_vol(rets, sorted(windows), trading_days) if windows else pd.DataFrame(index=rets.index)
ewma = ewma_vol(rets, lam, trading_days)
summ = summary_stats(rets, trading_days)

# -----------------------------
# Layout
# -----------------------------
c1, c2, c3 = st.columns([2, 2, 1])

with c1:
    st.subheader(f"Price: {price.name}")
    st.line_chart(price)
    st.dataframe(price.tail(10), use_container_width=True)

with c2:
    st.subheader("Log returns")
    st.line_chart(rets)
    st.dataframe(rets.tail(10), use_container_width=True)

with c3:
    st.subheader("Summary")
    st.dataframe(summ.to_frame("value"), use_container_width=True)

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Rolling annualized volatility")
    if roll_df.empty:
        st.info("No rolling windows selected.")
    else:
        st.line_chart(roll_df)
        st.dataframe(roll_df.tail(10), use_container_width=True)

with right:
    st.subheader(f"EWMA annualized volatility (λ={lam})")
    st.line_chart(ewma)
    st.dataframe(ewma.tail(10).to_frame("ewma_ann_vol"), use_container_width=True)

st.divider()

# Latest snapshot
latest = {"ewma_ann_vol": float(ewma.dropna().iloc[-1]) if ewma.notna().any() else np.nan}
for col in roll_df.columns:
    latest[col] = float(roll_df[col].dropna().iloc[-1]) if roll_df[col].notna().any() else np.nan

st.subheader("Latest volatility snapshot")
st.dataframe(pd.DataFrame([latest], index=[price.name]), use_container_width=True)

# Downloads
st.subheader("Download results")
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("Download returns.csv", to_csv_bytes(rets.to_frame("log_return")), "returns.csv", "text/csv")
with dl2:
    st.download_button("Download rolling_vol.csv", to_csv_bytes(roll_df), "rolling_vol.csv", "text/csv")
with dl3:
    st.download_button("Download ewma_vol.csv", to_csv_bytes(ewma.to_frame("ewma_ann_vol")), "ewma_vol.csv", "text/csv")
