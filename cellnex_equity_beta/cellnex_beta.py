"""
Cellnex Equity Beta Estimation vs MSCI Europe

Estimates equity beta using multiple methodologies, return horizons,
and half-lives to allow triangulation of results.

Data through 31 January 2026.

Usage:
    python cellnex_beta.py                  # auto-download via yfinance
    python cellnex_beta.py --use-sample     # use sample data (if no internet)

When live data is unavailable, the script falls back to sample data
generated from a calibrated factor model. Replace with your own CSV
files in the data/ directory for production use.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

CELLNEX_TICKER = "CLNX.MC"

# MSCI Europe proxies (EUR-denominated ETFs preferred, with fallbacks)
MSCI_EUROPE_TICKERS = [
    "IMEU.AS",   # iShares Core MSCI Europe UCITS ETF (Euronext Amsterdam, EUR)
    "SMEA.PA",   # Amundi MSCI Europe UCITS ETF (Euronext Paris, EUR)
    "MEUD.PA",   # Lyxor MSCI Europe (Euronext Paris, EUR)
    "IEUR",      # iShares MSCI Europe ETF (US-listed, USD fallback)
]

START_DATE = "2016-02-01"  # ~10 years of history
END_DATE = "2026-02-01"    # to capture up to 31 Jan 2026

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"

# Return horizons
RETURN_HORIZONS = {
    "Daily": 1,
    "Weekly": 5,
    "Monthly": 21,
}

# Half-lives for exponential weighting (in years)
HALF_LIVES_YEARS = [1, 2, 5, 8]


# =============================================================================
# Data Download (yfinance)
# =============================================================================

def download_data(ticker, start, end):
    """Download adjusted close prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception:
        return None
    if data is None or data.empty:
        return None
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
    else:
        data = data["Close"]
    return data.dropna()


def get_live_data(start, end):
    """Try to download Cellnex and MSCI Europe data via yfinance."""
    print("  Downloading Cellnex...")
    cellnex = download_data(CELLNEX_TICKER, start, end)
    if cellnex is None or len(cellnex) < 100:
        return None, None, None

    print(f"  -> Cellnex: {len(cellnex)} daily observations")

    for ticker in MSCI_EUROPE_TICKERS:
        print(f"  Trying MSCI Europe proxy: {ticker}...")
        msci = download_data(ticker, start, end)
        if msci is not None and len(msci) > 252:
            print(f"  -> Using {ticker} ({len(msci)} daily observations)")
            return cellnex, msci, ticker

    return None, None, None


# =============================================================================
# Sample Data Generation (calibrated factor model)
# =============================================================================

def generate_sample_data():
    """
    Generate realistic sample data using a calibrated factor model.

    Calibrated to approximate known historical characteristics:
    - Cellnex: telecom tower infra, listed Madrid (CLNX.MC)
      - High growth 2016-2021, correction 2022-2023, sideways 2024-2025
      - Annualized vol ~25-30%
    - MSCI Europe: broad European equity index
      - Moderate growth with COVID drawdown
      - Annualized vol ~15%
    - Beta: ~0.5-0.8 range (defensive infrastructure stock)
    """
    np.random.seed(42)

    # Generate business days from Feb 2016 to Jan 2026
    dates = pd.bdate_range(start="2016-02-01", end="2026-01-31")
    n = len(dates)

    # MSCI Europe: GBM with regime-dependent drift/vol
    msci_daily_returns = np.zeros(n)
    msci_price = np.zeros(n)
    msci_price[0] = 100.0  # Normalized to 100

    # Define market regimes (approximate)
    for i in range(1, n):
        date = dates[i]
        # Base parameters
        annual_drift = 0.06
        annual_vol = 0.14

        # COVID crash: Feb-Mar 2020
        if pd.Timestamp("2020-02-20") <= date <= pd.Timestamp("2020-03-23"):
            annual_drift = -3.0
            annual_vol = 0.80
        # COVID recovery: Apr-Dec 2020
        elif pd.Timestamp("2020-03-24") <= date <= pd.Timestamp("2020-12-31"):
            annual_drift = 0.60
            annual_vol = 0.25
        # 2022 correction (war, rates)
        elif pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2022-10-01"):
            annual_drift = -0.20
            annual_vol = 0.22
        # 2023-2025 recovery
        elif pd.Timestamp("2023-01-01") <= date <= pd.Timestamp("2025-12-31"):
            annual_drift = 0.10
            annual_vol = 0.13

        dt = 1 / 252
        msci_daily_returns[i] = (annual_drift - 0.5 * annual_vol**2) * dt + \
                                 annual_vol * np.sqrt(dt) * np.random.randn()
        msci_price[i] = msci_price[i - 1] * np.exp(msci_daily_returns[i])

    # Cellnex: factor model with time-varying beta and idiosyncratic component
    cellnex_price = np.zeros(n)
    cellnex_price[0] = 14.0  # Approx IPO era price in EUR

    for i in range(1, n):
        date = dates[i]
        dt = 1 / 252

        # Time-varying beta
        beta = 0.65
        if date < pd.Timestamp("2019-01-01"):
            beta = 0.55
        elif date < pd.Timestamp("2021-01-01"):
            beta = 0.70  # Growth phase, higher beta
        elif date < pd.Timestamp("2023-01-01"):
            beta = 0.80  # Correction phase, higher beta
        else:
            beta = 0.60  # Stabilization

        # Idiosyncratic component
        idio_vol = 0.22
        if pd.Timestamp("2019-01-01") <= date <= pd.Timestamp("2021-06-30"):
            idio_drift = 0.25  # Strong growth phase
            idio_vol = 0.20
        elif pd.Timestamp("2021-07-01") <= date <= pd.Timestamp("2023-06-30"):
            idio_drift = -0.20  # Correction
            idio_vol = 0.25
        else:
            idio_drift = 0.02
            idio_vol = 0.20

        idio_return = (idio_drift - 0.5 * idio_vol**2) * dt + \
                       idio_vol * np.sqrt(dt) * np.random.randn()

        cellnex_return = beta * msci_daily_returns[i] + idio_return
        cellnex_price[i] = cellnex_price[i - 1] * np.exp(cellnex_return)

    cellnex_series = pd.Series(cellnex_price, index=dates, name="CLNX.MC")
    msci_series = pd.Series(msci_price, index=dates, name="MSCI_Europe")

    return cellnex_series, msci_series


def load_csv_data():
    """Try to load price data from CSV files in the data/ directory."""
    cellnex_path = DATA_DIR / "cellnex_prices.csv"
    msci_path = DATA_DIR / "msci_europe_prices.csv"

    if cellnex_path.exists() and msci_path.exists():
        cellnex = pd.read_csv(cellnex_path, index_col=0, parse_dates=True).squeeze()
        msci = pd.read_csv(msci_path, index_col=0, parse_dates=True).squeeze()
        return cellnex, msci
    return None, None


# =============================================================================
# Returns Calculation
# =============================================================================

def compute_returns(prices, horizon_days):
    """Compute log returns for a given horizon (non-overlapping)."""
    if horizon_days == 1:
        return np.log(prices / prices.shift(1)).dropna()
    else:
        sampled = prices.iloc[::horizon_days]
        return np.log(sampled / sampled.shift(1)).dropna()


# =============================================================================
# Beta Estimation Methods
# =============================================================================

def ols_beta(stock_returns, index_returns):
    """Standard OLS beta (full sample)."""
    aligned = pd.concat([stock_returns, index_returns], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return np.nan
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def ols_beta_window(stock_returns, index_returns, window_periods):
    """OLS beta using only the most recent N periods."""
    aligned = pd.concat([stock_returns, index_returns], axis=1, join="inner").dropna()
    if len(aligned) < max(window_periods, 10):
        return np.nan
    aligned = aligned.tail(window_periods)
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def ew_beta(stock_returns, index_returns, half_life_periods):
    """Exponentially weighted beta using WLS with decay weights."""
    aligned = pd.concat([stock_returns, index_returns], axis=1, join="inner").dropna()
    n = len(aligned)
    if n < 10:
        return np.nan

    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values

    # Weights: most recent observation has highest weight
    periods_ago = np.arange(n - 1, -1, -1, dtype=float)
    decay = np.log(2) / half_life_periods
    weights = np.exp(-decay * periods_ago)

    # Weighted least squares via normal equations
    sqrt_w = np.sqrt(weights)
    X = np.column_stack([np.ones(n), x])
    Xw = X * sqrt_w[:, np.newaxis]
    yw = y * sqrt_w
    try:
        coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return coeffs[1]  # slope = beta


# =============================================================================
# Main Estimation
# =============================================================================

def estimate_all_betas(cellnex_prices, msci_prices):
    """Estimate betas across all methodology/horizon combinations."""
    results = {}

    for horizon_name, horizon_days in RETURN_HORIZONS.items():
        stock_ret = compute_returns(cellnex_prices, horizon_days)
        index_ret = compute_returns(msci_prices, horizon_days)

        # 1) OLS - Full Sample
        beta = ols_beta(stock_ret, index_ret)
        results[("OLS (Full Sample)", horizon_name)] = beta

        # 2) OLS - 2 Year Window
        if horizon_days == 1:
            window = 504   # ~2 years of trading days
        elif horizon_days == 5:
            window = 104   # ~2 years of weeks
        else:
            window = 24    # ~2 years of months
        beta = ols_beta_window(stock_ret, index_ret, window)
        results[("OLS (2-Year Window)", horizon_name)] = beta

        # 3) Exponentially Weighted with different half-lives
        for hl_years in HALF_LIVES_YEARS:
            if horizon_days == 1:
                hl_periods = hl_years * 252
            elif horizon_days == 5:
                hl_periods = hl_years * 52
            else:
                hl_periods = hl_years * 12

            beta = ew_beta(stock_ret, index_ret, hl_periods)
            results[(f"EW (HL = {hl_years}Y)", horizon_name)] = beta

    return results


# =============================================================================
# Visualization
# =============================================================================

def create_chart(results, msci_proxy_label, cellnex_prices, msci_prices,
                 is_sample_data=False):
    """Create a single professional chart for senior management."""

    methodologies = [
        "OLS (Full Sample)",
        "OLS (2-Year Window)",
        "EW (HL = 1Y)",
        "EW (HL = 2Y)",
        "EW (HL = 5Y)",
        "EW (HL = 8Y)",
    ]
    horizons = list(RETURN_HORIZONS.keys())

    # Build matrix
    matrix = np.zeros((len(methodologies), len(horizons)))
    for i, method in enumerate(methodologies):
        for j, horizon in enumerate(horizons):
            matrix[i, j] = results.get((method, horizon), np.nan)

    all_betas = [v for v in results.values() if not np.isnan(v)]
    mean_beta = np.mean(all_betas)
    median_beta = np.median(all_betas)
    min_beta = np.min(all_betas)
    max_beta = np.max(all_betas)

    # =========================================================================
    # Figure setup
    # =========================================================================
    fig = plt.figure(figsize=(14, 10.5), facecolor="white")

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3, 2],
        height_ratios=[6, 1],
        hspace=0.12,
        wspace=0.35,
        left=0.12, right=0.95,
        top=0.84, bottom=0.08,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_dots = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[1, :])

    # =========================================================================
    # Panel 1: Annotated Heatmap
    # =========================================================================
    vmin = max(0, min_beta - 0.15)
    vmax = max_beta + 0.15
    cmap = plt.cm.YlOrRd

    im = ax_heat.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(len(methodologies)):
        for j in range(len(horizons)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "N/A"
            else:
                text = f"{val:.2f}"
            norm_val = (val - vmin) / (vmax - vmin) if not np.isnan(val) else 0.5
            text_color = "white" if norm_val > 0.65 else "black"
            ax_heat.text(j, i, text, ha="center", va="center",
                         fontsize=14, fontweight="bold", color=text_color)

    ax_heat.set_xticks(range(len(horizons)))
    ax_heat.set_xticklabels(horizons, fontsize=11, fontweight="bold")
    ax_heat.set_yticks(range(len(methodologies)))
    ax_heat.set_yticklabels(methodologies, fontsize=10)
    ax_heat.set_xlabel("Return Horizon", fontsize=11, fontweight="bold", labelpad=10)
    ax_heat.set_title("Beta Estimates by Method & Horizon", fontsize=12,
                       fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.03)
    cbar.set_label("Beta", fontsize=10)

    # =========================================================================
    # Panel 2: Dot Plot (distribution of all estimates)
    # =========================================================================
    horizon_colors = {"Daily": "#1565C0", "Weekly": "#E65100", "Monthly": "#2E7D32"}
    method_short = {
        "OLS (Full Sample)": "OLS Full",
        "OLS (2-Year Window)": "OLS 2Y",
        "EW (HL = 1Y)": "EW 1Y",
        "EW (HL = 2Y)": "EW 2Y",
        "EW (HL = 5Y)": "EW 5Y",
        "EW (HL = 8Y)": "EW 8Y",
    }

    y_positions = []
    x_values = []
    colors = []
    y_counter = 0

    for method in methodologies:
        for horizon in horizons:
            val = results.get((method, horizon), np.nan)
            if not np.isnan(val):
                y_positions.append(y_counter)
                x_values.append(val)
                colors.append(horizon_colors[horizon])
            y_counter += 1
        y_counter += 0.5

    ax_dots.scatter(x_values, y_positions, c=colors, s=110, zorder=5,
                    edgecolors="white", linewidths=1.5)

    # Median & mean lines
    ax_dots.axvline(median_beta, color="#C62828", linestyle="--", linewidth=2.0,
                    alpha=0.85, zorder=3)
    ax_dots.axvline(mean_beta, color="#6A1B9A", linestyle=":", linewidth=2.0,
                    alpha=0.85, zorder=3)

    # Y-axis labels
    y_counter = 0
    ytick_positions = []
    ytick_labels = []
    for method in methodologies:
        mid = y_counter + 1
        ytick_positions.append(mid)
        ytick_labels.append(method_short[method])
        y_counter += len(horizons) + 0.5

    ax_dots.set_yticks(ytick_positions)
    ax_dots.set_yticklabels(ytick_labels, fontsize=9)
    ax_dots.invert_yaxis()
    ax_dots.set_xlabel("Beta", fontsize=11, fontweight="bold", labelpad=10)
    ax_dots.set_title("Distribution of Estimates", fontsize=12,
                       fontweight="bold", pad=10)
    ax_dots.grid(axis="x", alpha=0.3, linestyle="-")

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=horizon_colors[h],
               markersize=9, label=h) for h in horizons
    ]
    legend_elements.append(
        Line2D([0], [0], color="#C62828", linestyle="--", linewidth=2,
               label=f"Median = {median_beta:.2f}")
    )
    legend_elements.append(
        Line2D([0], [0], color="#6A1B9A", linestyle=":", linewidth=2,
               label=f"Mean = {mean_beta:.2f}")
    )
    ax_dots.legend(handles=legend_elements, loc="lower right", fontsize=8,
                   framealpha=0.9, edgecolor="gray")

    # =========================================================================
    # Panel 3: Summary bar
    # =========================================================================
    ax_summary.axis("off")
    data_start = max(cellnex_prices.index[0], msci_prices.index[0]).strftime("%b %Y")
    data_end = min(cellnex_prices.index[-1], msci_prices.index[-1]).strftime("%b %Y")

    summary_text = (
        f"Summary  |  "
        f"Mean Beta: {mean_beta:.2f}  |  "
        f"Median Beta: {median_beta:.2f}  |  "
        f"Range: [{min_beta:.2f} \u2013 {max_beta:.2f}]  |  "
        f"Data: {data_start} to {data_end}  |  "
        f"Index: {msci_proxy_label}"
    )

    bbox = FancyBboxPatch(
        (0.01, 0.1), 0.98, 0.8,
        boxstyle="round,pad=0.01",
        facecolor="#F5F5F5", edgecolor="#BDBDBD", linewidth=1,
        transform=ax_summary.transAxes,
    )
    ax_summary.add_patch(bbox)
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#424242")

    # =========================================================================
    # Title
    # =========================================================================
    title = "Cellnex (CLNX) Equity Beta to MSCI Europe"
    subtitle = "Multi-Method Estimation for Triangulation"
    if is_sample_data:
        subtitle += "  [Sample Data -- replace with live data for production]"

    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", y=0.96)

    # =========================================================================
    # Methodology note
    # =========================================================================
    note = (
        "OLS = Ordinary Least Squares regression  |  "
        "EW = Exponentially Weighted (half-life controls decay of older observations)  |  "
        "2Y Window = most recent 2 years only"
    )
    fig.text(0.5, 0.02, note, ha="center", va="center", fontsize=7.5,
             color="#757575", style="italic")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cellnex_equity_beta.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\n  Chart saved to: {output_path}")
    plt.close(fig)

    return output_path


# =============================================================================
# Print Results Table
# =============================================================================

def print_results_table(results):
    """Print a formatted table of all beta estimates."""
    methodologies = [
        "OLS (Full Sample)",
        "OLS (2-Year Window)",
        "EW (HL = 1Y)",
        "EW (HL = 2Y)",
        "EW (HL = 5Y)",
        "EW (HL = 8Y)",
    ]
    horizons = list(RETURN_HORIZONS.keys())

    print("\n" + "=" * 65)
    print("  Cellnex Equity Beta Estimates vs MSCI Europe")
    print("=" * 65)
    print(f"  {'Methodology':<22} {'Daily':>10} {'Weekly':>10} {'Monthly':>10}")
    print("  " + "-" * 61)

    for method in methodologies:
        row = f"  {method:<22}"
        for horizon in horizons:
            val = results.get((method, horizon), np.nan)
            if np.isnan(val):
                row += f"{'N/A':>10}"
            else:
                row += f"{val:>10.3f}"
        print(row)

    print("  " + "-" * 61)
    all_betas = [v for v in results.values() if not np.isnan(v)]
    print(f"  {'Mean':>22} {np.mean(all_betas):>30.3f}")
    print(f"  {'Median':>22} {np.median(all_betas):>30.3f}")
    print(f"  {'Range':>22} {f'[{np.min(all_betas):.3f} - {np.max(all_betas):.3f}]':>30}")
    print("=" * 65)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 55)
    print("  Cellnex Equity Beta Estimation vs MSCI Europe")
    print("=" * 55)

    use_sample = "--use-sample" in sys.argv
    is_sample_data = False
    msci_label = ""

    # Step 1: Get price data
    print("\n1. Loading price data...")

    if not use_sample:
        # Try live download first
        cellnex_prices, msci_prices, msci_ticker = get_live_data(START_DATE, END_DATE)
        if cellnex_prices is not None:
            msci_label = msci_ticker
        else:
            # Try CSV files
            print("  Live download failed. Trying CSV files...")
            cellnex_prices, msci_prices = load_csv_data()
            if cellnex_prices is not None:
                msci_label = "MSCI Europe (CSV)"
            else:
                print("  No CSV data found. Falling back to sample data...")
                use_sample = True

    if use_sample:
        print("  Generating sample data (calibrated factor model)...")
        cellnex_prices, msci_prices = generate_sample_data()
        msci_label = "MSCI Europe (sample)"
        is_sample_data = True

    # Align date ranges
    common_start = max(cellnex_prices.index[0], msci_prices.index[0])
    common_end = min(cellnex_prices.index[-1], msci_prices.index[-1])
    cellnex_prices = cellnex_prices[common_start:common_end]
    msci_prices = msci_prices[common_start:common_end]

    print(f"  Date range: {common_start.strftime('%Y-%m-%d')} to "
          f"{common_end.strftime('%Y-%m-%d')}")
    print(f"  Cellnex observations:     {len(cellnex_prices):,}")
    print(f"  MSCI Europe observations: {len(msci_prices):,}")

    if is_sample_data:
        print("\n  NOTE: Using sample data generated from a calibrated factor model.")
        print("  For production results, run with live yfinance access or provide")
        print("  CSV files in the data/ directory.")

    # Step 2: Estimate betas
    print("\n2. Estimating betas across methodologies and horizons...")
    results = estimate_all_betas(cellnex_prices, msci_prices)
    print_results_table(results)

    # Step 3: Create chart
    print("\n3. Creating chart...")
    output_path = create_chart(results, msci_label, cellnex_prices, msci_prices,
                               is_sample_data=is_sample_data)

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
