import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Volatility Analysis Demo", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def to_datetime_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    return out

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(how="all")
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all")

def annualize_vol(daily_vol: pd.DataFrame | pd.Series, trading_days: int) -> pd.DataFrame | pd.Series:
    return daily_vol * np.sqrt(trading_days)

def rolling_vol(returns: pd.DataFrame, windows: list[int], trading_days: int) -> dict[int, pd.DataFrame]:
    vols = {}
    for w in windows:
        vols[w] = annualize_vol(returns.rolling(w).std(), trading_days)
    return vols

def ewma_vol(returns: pd.DataFrame, lam: float, trading_days: int) -> pd.DataFrame:
    """
    EWMA variance update:
      var_t = lam*var_{t-1} + (1-lam)*r_{t-1}^2
    """
    r2 = returns.pow(2)

    ewma_var = pd.DataFrame(index=r2.index, columns=r2.columns, dtype=float)
    # initialize with first available r^2 per column
    first = r2.iloc[0].fillna(0.0)
    ewma_var.iloc[0] = first

    for i in range(1, len(r2)):
        ewma_var.iloc[i] = lam * ewma_var.iloc[i - 1] + (1 - lam) * r2.iloc[i - 1].fillna(0.0)

    ewma_daily_vol = np.sqrt(ewma_var)
    return annualize_vol(ewma_daily_vol, trading_days)

def vol_summary(returns: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    daily_std = returns.std()
    ann_vol = daily_std * np.sqrt(trading_days)

    summary = pd.DataFrame({
        "count_days": returns.count(),
        "mean_daily_return": returns.mean(),
        "std_daily_return": daily_std,
        "ann_volatility": ann_vol,
        "min_daily_return": returns.min(),
        "5pct_daily_return": returns.quantile(0.05),
        "95pct_daily_return": returns.quantile(0.95),
        "max_daily_return": returns.max(),
    })
    return summary.sort_values("ann_volatility", ascending=False)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


# -----------------------------
# UI
# -----------------------------
st.title("📈 Volatility Analysis (Streamlit Demo)")
st.caption("Upload price data → compute returns → rolling & EWMA volatility → summary & export")

with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    trading_days = st.selectbox("Annualization days", options=[252, 365], index=0)
    windows = st.multiselect("Rolling windows (days)", options=[5, 10, 20, 30, 60, 90, 120, 252], default=[20, 60, 252])
    lam = st.slider("EWMA lambda (λ)", min_value=0.80, max_value=0.99, value=0.94, step=0.01)

    st.divider()
    st.markdown("**Notes**")
    st.markdown("- 252 for stocks, 365 for crypto")
    st.markdown("- Volatility shown is **annualized**")

if not uploaded:
    st.info("Upload a CSV to begin. Example columns: Date, Close  (or Date, AAPL, MSFT, BTC)")
    st.stop()

# Read CSV
try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if raw.empty:
    st.error("CSV is empty.")
    st.stop()

# Column selectors
cols = list(raw.columns)
col1, col2 = st.columns(2)

with col1:
    date_col = st.selectbox("Date column", options=cols, index=0)

with col2:
    numeric_candidates = [c for c in cols if c != date_col]
    price_cols = st.multiselect("Price column(s)", options=numeric_candidates, default=numeric_candidates[:1])

if not price_cols:
    st.error("Please select at least one price column.")
    st.stop()

# Prepare data
df = to_datetime_index(raw, date_col)
prices = df[price_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")

if prices.shape[0] < 5:
    st.error("Not enough rows after parsing dates/prices. Check your columns.")
    st.stop()

returns = compute_log_returns(prices)

if returns.shape[0] < max(windows, default=1):
    st.warning("You have fewer return rows than the largest rolling window. Some rolling vol lines may be mostly empty.")

# Compute vols
roll_vols = rolling_vol(returns, windows=sorted(windows), trading_days=trading_days) if windows else {}
ewma = ewma_vol(returns, lam=lam, trading_days=trading_days)
summary = vol_summary(returns, trading_days=trading_days)

# -----------------------------
# Display
# -----------------------------
tab_prices, tab_returns, tab_vol, tab_summary = st.tabs(["Prices", "Returns", "Volatility", "Summary & Export"])

with tab_prices:
    st.subheader("Prices")
    st.line_chart(prices)
    st.dataframe(prices.tail(20), use_container_width=True)

with tab_returns:
    st.subheader("Log Returns")
    st.line_chart(returns)
    st.dataframe(returns.tail(20), use_container_width=True)

with tab_vol:
    st.subheader("Annualized Volatility")

    left, right = st.columns(2)

    with left:
        st.markdown("### Rolling volatility")
        if not roll_vols:
            st.info("No rolling windows selected.")
        else:
            for w, vdf in roll_vols.items():
                st.markdown(f"**{w}-day rolling vol**")
                st.line_chart(vdf)

    with right:
        st.markdown("### EWMA volatility")
        st.markdown(f"**λ = {lam}**")
        st.line_chart(ewma)

    # Latest snapshot table
    st.markdown("### Latest volatility snapshot")
    latest_rows = []
    for asset in returns.columns:
        row = {"asset": asset, "ewma_ann_vol": float(ewma[asset].dropna().iloc[-1]) if ewma[asset].notna().any() else np.nan}
        for w, vdf in roll_vols.items():
            row[f"roll_{w}d_ann_vol"] = float(vdf[asset].dropna().iloc[-1]) if vdf[asset].notna().any() else np.nan
        latest_rows.append(row)

    latest = pd.DataFrame(latest_rows).set_index("asset").sort_values("ewma_ann_vol", ascending=False)
    st.dataframe(latest, use_container_width=True)

with tab_summary:
    st.subheader("Volatility Summary (from returns)")
    st.dataframe(summary, use_container_width=True)

    st.divider()
    st.subheader("Export")
    export_choice = st.selectbox("Choose what to export", ["Summary", "Returns", "EWMA Vol", "Latest Snapshot"] + ([f"Rolling Vol ({w}d)" for w in sorted(roll_vols.keys())] if roll_vols else []))

    if export_choice == "Summary":
        export_df = summary
        filename = "vol_summary.csv"
    elif export_choice == "Returns":
        export_df = returns
        filename = "returns.csv"
    elif export_choice == "EWMA Vol":
        export_df = ewma
        filename = "ewma_vol.csv"
    elif export_choice == "Latest Snapshot":
        export_df = latest
        filename = "latest_vol_snapshot.csv"
    else:
        # Rolling Vol (Xd)
        w = int(export_choice.split("(")[1].split("d")[0])
        export_df = roll_vols[w]
        filename = f"rolling_vol_{w}d.csv"

    st.download_button(
        label=f"Download {filename}",
        data=df_to_csv_bytes(export_df),
        file_name=filename,
        mime="text/csv",
    )

st.caption("Tip: For clean results, make sure prices are regular (daily) and in the same currency/units per column.")
