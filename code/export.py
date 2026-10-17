"""Course submission price exports."""
from pathlib import Path
import json
import numpy as np

def export_asset_closing_prices(context, output_dir, start=None, end=None):
    """Export 18 asset definitions and observed closes; never fill market holidays.

    The existing pipeline uses adjusted Close. A separate Yahoo request retrieves
    unadjusted Close so the course's daily closing-price submission is explicit.
    CSV values retain calculation precision rather than rounding source data.
    """
    pd = context["pd"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tickers = list(context["ALL_ASSETS"])
    if len(tickers) != 18 or len(set(tickers)) != 18:
        raise ValueError("Expected exactly 18 distinct assets.")
    observed = context["close_local"].reindex(columns=tickers).loc[start:end]
    if observed.empty or observed.notna().sum().eq(0).any():
        raise ValueError("Adjusted closing prices do not cover all 18 assets.")
    raw = context["yf"].download(
        tickers, start=str(observed.index.min().date()),
        end=str((observed.index.max() + pd.Timedelta(days=1)).date()),
        auto_adjust=False, progress=False, threads=False,
    )
    if raw.empty or 'Close' not in raw:
        raise RuntimeError("Unadjusted closing-price download is empty or lacks Close.")
    unadjusted = raw["Close"].reindex(index=observed.index, columns=tickers)
    # Compare actual observations, not a generic weekday calendar: a local
    # market holiday is valid missing data, but a lost provider row is not.
    expected = observed.notna()
    missing = expected & (~np.isfinite(unadjusted) | unadjusted.le(0))
    if missing.any().any():
        examples = [(str(date.date()), ticker) for date, ticker in
                    missing.stack().loc[lambda s: s].index[:5]]
        raise RuntimeError(f"Unadjusted closing-price download is incomplete or invalid. Missing observed dates/assets: {examples}. Retry the download.")
    if (expected & (~np.isfinite(observed) | observed.le(0))).any().any():
        raise ValueError("Adjusted closing prices contain invalid observations.")
    # Crypto has weekend observations while the FX valuation grid is business
    # days. Use the preceding grid row (never a future rate). Existing missing
    # FX values on that grid remain missing and fail validation below.
    fx = context["fx_to_sgd"].reindex(observed.index, method='ffill')
    sgd = observed.copy()
    metadata = context["assets"].reindex(tickers)
    for ticker in tickers:
        currency = metadata.loc[ticker, "Local Currency"]
        if currency != "SGD":
            sgd[ticker] = observed[ticker] * fx[currency]
    valuation_prices = context.get('portfolio_prices_sgd')
    conversion_start = (valuation_prices.index.min() if valuation_prices is not None
                        else observed.index.min())
    conversion_required = expected.copy()
    conversion_required.loc[conversion_required.index < conversion_start] = False
    if (conversion_required & (~np.isfinite(sgd) | sgd.le(0))).any().any():
        raise RuntimeError("SGD conversion is incomplete or invalid on observed asset dates.")
    tables = {
        "asset_list.csv": metadata,
        "daily_closing_prices_local_currency.csv": unadjusted,
        "daily_adjusted_closing_prices_local_currency.csv": observed,
        "daily_adjusted_closing_prices_sgd.csv": sgd,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, encoding="utf-8-sig", index_label="Ticker" if filename == "asset_list.csv" else "Date")
    (output / "price_data_notes.json").write_text(json.dumps({
        "source": "Yahoo Finance", "asset_count": 18,
        "start": str(observed.index.min().date()), "end": str(observed.index.max().date()),
        "coverage_check": "Every adjusted local observation must have a finite positive unadjusted close. SGD conversion is required from the strategy's common valuation start; earlier unavailable FX stays missing. Genuine local-market holidays remain missing.",
        "sgd_required_from": str(conversion_start.date()),
        "per_asset_coverage": {ticker: {
            "first_observation": str(unadjusted[ticker].first_valid_index().date()),
            "last_observation": str(unadjusted[ticker].last_valid_index().date()),
            "observations": int(unadjusted[ticker].notna().sum()),
        } for ticker in tickers},
        "raw_close": "Unadjusted Close in each instrument's trading currency.",
        "adjusted_close": "Yahoo auto_adjust=True Close, used for strategy returns.",
        "missing_values": "Preserved on non-trading dates; no artificial price observations.",
        "sgd_conversion": "Adjusted local close times pipeline FX to SGD; FX may be forward-filled up to 5 business days.",
    }, indent=2), encoding="utf-8")
    return {name: output / name for name in tables}
