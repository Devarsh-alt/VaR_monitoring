
from pathlib import Path
import time
import pandas as pd
import numpy as np
import yfinance as yf

# --------------------------------------------------
# RS05: Indian equity external-validation data
# --------------------------------------------------

START_DATE = "2015-01-01"

# Yahoo Finance end date is exclusive.
# This matches the US study's final date, 2026-06-12.
END_DATE = "2026-06-13"

OUT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'PROJECT_ROOT').is_file()) / 'data/monitoring/regime_shifts/RS05/india'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed 24-stock sample across major Indian sectors.
# This is NOT a point-in-time historical constituent list.
INDIAN_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "ULTRACEMCO.NS",
    "ASIANPAINT.NS",
    "HINDUNILVR.NS",
    "WIPRO.NS",
]

# Benchmark only; do not treat this as a 25th stock.
BENCHMARK = "^NSEI"

def download_close(ticker, attempts=3):
    """Download adjusted daily close with retries."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
                timeout=30,
                multi_level_index=False,
            )

            if df is None or df.empty:
                raise ValueError("Yahoo Finance returned no rows.")

            # Handle both flat and MultiIndex column formats.
            if isinstance(df.columns, pd.MultiIndex):
                if "Close" in df.columns.get_level_values(0):
                    close = df.xs("Close", axis=1, level=0)
                else:
                    close = df.xs("Close", axis=1, level=1)

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            else:
                if "Close" not in df.columns:
                    raise ValueError("Close column missing.")
                close = df["Close"]

            close = pd.to_numeric(close, errors="coerce")
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close[~close.index.duplicated(keep="last")]
            close = close.sort_index()
            close.name = ticker

            close = close.loc[
                (close.index >= START_DATE)
                & (close.index < END_DATE)
            ]

            close = close.dropna()

            if close.empty:
                raise ValueError("No valid close prices after cleaning.")

            if (close <= 0).any():
                raise ValueError("Non-positive prices detected.")

            return close

        except Exception as exc:
            last_error = repr(exc)
            print(
                f"Attempt {attempt}/{attempts} failed for "
                f"{ticker}: {last_error}"
            )
            time.sleep(2 * attempt)

    return None, last_error


# Download each ticker individually so one failure does not
# silently remove other stocks from the dataset.
series = {}
failures = []

for i, ticker in enumerate(INDIAN_STOCKS + [BENCHMARK], start=1):
    print(f"[{i}/{len(INDIAN_STOCKS) + 1}] Downloading {ticker}")

    result = download_close(ticker)

    if isinstance(result, tuple):
        failures.append({
            "ticker": ticker,
            "error": result[1],
        })
        continue

    series[ticker] = result
    print(
        f"    rows={len(result)}, "
        f"first={result.index.min().date()}, "
        f"last={result.index.max().date()}"
    )

    time.sleep(0.3)

if not series:
    raise RuntimeError(
        "No tickers downloaded. Check internet access and yfinance."
    )

# Prices: one column per ticker, no forward filling.
prices = pd.concat(series.values(), axis=1).sort_index()
prices.index.name = "Date"

stock_columns = [
    ticker for ticker in INDIAN_STOCKS if ticker in prices.columns
]
stock_prices = prices[stock_columns]

# Log returns are calculated independently per ticker.
# Missing observations remain missing; no prices are imputed.
log_returns = np.log(stock_prices / stock_prices.shift(1))
log_returns.index.name = "Date"

# Long-form price file for panel-based downstream experiments.
prices_long = (
    stock_prices.rename_axis("Date")
    .reset_index()
    .melt(
        id_vars="Date",
        var_name="ticker",
        value_name="adjusted_close",
    )
    .dropna(subset=["adjusted_close"])
)

# Coverage and data-quality audit.
coverage = []

for ticker in INDIAN_STOCKS:
    if ticker not in stock_prices.columns:
        coverage.append({
            "ticker": ticker,
            "downloaded": False,
            "observations": 0,
            "first_date": None,
            "last_date": None,
            "missing_fraction_on_union_calendar": np.nan,
            "return_observations": 0,
        })
        continue

    s = stock_prices[ticker]
    r = log_returns[ticker]

    coverage.append({
        "ticker": ticker,
        "downloaded": True,
        "observations": int(s.notna().sum()),
        "first_date": str(s.first_valid_index().date()),
        "last_date": str(s.last_valid_index().date()),
        "missing_fraction_on_union_calendar": float(s.isna().mean()),
        "return_observations": int(r.notna().sum()),
    })

coverage = pd.DataFrame(coverage)

# Save all files.
stock_prices.to_csv(OUT_DIR / "india_adjusted_close_wide.csv")
log_returns.to_csv(OUT_DIR / "india_log_returns_wide.csv")
prices_long.to_csv(
    OUT_DIR / "india_adjusted_close_long.csv", index=False
)
coverage.to_csv(OUT_DIR / "india_data_coverage.csv", index=False)

if BENCHMARK in prices.columns:
    prices[[BENCHMARK]].to_csv(
        OUT_DIR / "nifty50_index_adjusted_close.csv"
    )

pd.DataFrame(failures).to_csv(
    OUT_DIR / "india_download_failures.csv", index=False
)

# Machine-readable universe and sample definition.
pd.DataFrame({
    "ticker": INDIAN_STOCKS,
    "market": "NSE",
    "sample_type": "fixed_selected_equity_sample",
    "point_in_time_constituents": False,
}).to_csv(OUT_DIR / "india_universe.csv", index=False)

print("\n=== DOWNLOAD SUMMARY ===")
print("Output directory:", OUT_DIR.resolve())
print("Requested stocks:", len(INDIAN_STOCKS))
print("Successfully downloaded stocks:", len(stock_columns))
print("Failed downloads:", len(failures))
print("Price matrix shape:", stock_prices.shape)
print("Return matrix shape:", log_returns.shape)

print("\n=== COVERAGE ===")
print(coverage.to_string(index=False))

if len(stock_columns) < len(INDIAN_STOCKS):
    print(
        "\nWARNING: Some stocks failed. Review "
        "india_download_failures.csv before modeling."
    )

if len(stock_columns) < 20:
    raise RuntimeError(
        "Fewer than 20 stocks downloaded. Do not start model evaluation."
    )

print("\nData download and audit files saved.")
