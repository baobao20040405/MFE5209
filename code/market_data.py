"""Market/FX/calendar preparation, preserving the research convention."""
if __name__ == '__main__':
    print(
        'market_data.py is an internal data-preparation stage. '
        'Run code/main.py to evaluate the final product.'
    )
    raise SystemExit(0)

FX_DOWNLOAD = ['SGD=X', 'EURSGD=X', 'GBPSGD=X', 'JPY=X', 'CNY=X', 'HKDSGD=X']
VIX_TICKERS = ['^VIX', '^VIX3M']
MARKET_BENCHMARK_TICKERS = ['VT']
download_tickers = PORTFOLIO_ASSETS + FX_DOWNLOAD + VIX_TICKERS + MARKET_BENCHMARK_TICKERS
raw = download_market_data(

    download_tickers,
    start=DOWNLOAD_START,
    end=DOWNLOAD_END,
    auto_adjust=True,
    progress=False,
    threads=False,
    group_by='column'
)

if raw.empty:
    raise RuntimeError('Yahoo Finance does not return the data, please check the network connection and restart.')

close_local = raw['Close'].reindex(columns=ALL_ASSETS)
volume_local = raw['Volume'].reindex(columns=ALL_ASSETS)
fx_close = raw['Close'].reindex(columns=FX_DOWNLOAD)
vix_close = raw['Close'].reindex(columns=VIX_TICKERS)
market_benchmark_close_local = raw['Close'].reindex(columns=MARKET_BENCHMARK_TICKERS)

coverage = pd.DataFrame({
    'Minimum Effective Date': close_local.apply(lambda s: s.first_valid_index()),
    'Last effective date': close_local.apply(lambda s: s.last_valid_index()),
    'Effective observations': close_local.notna().sum(),
}).join(assets[['Asset Class', 'Asset Name', 'Local Currency']])
coverage = coverage[['Asset Class', 'Asset Name', 'Local Currency', 'Minimum Effective Date', 'Last effective date', 'Effective observations']]
display(coverage)

missing = coverage.index[coverage['Effective observations'].eq(0)].tolist()
if missing:
    raise RuntimeError(f'No data available for:{missing}')

calendar_start = min(close_local.apply(lambda s: s.first_valid_index()))
calendar_end = max(close_local.apply(lambda s: s.last_valid_index()))
calendar = pd.bdate_range(calendar_start, calendar_end)

local_bday = close_local.reindex(calendar).ffill(limit=5)
volume_bday = volume_local.reindex(calendar)
fx_bday = fx_close.reindex(calendar).ffill(limit=5)
vix_bday = vix_close.reindex(calendar).ffill(limit=5)
vix_open = raw['Open'].reindex(columns=VIX_TICKERS)
vix_open_bday = vix_open.reindex(calendar).ffill(limit=5)

# Yahoo's JPYSGD=X, CNYSGD=X history occasionally returns only the latest day and is constructed using the United States dollar cross-exchange rate.
# SGD=X, JPY=X, CNY=X, CNY=X each of the corresponding SGD, JPY, and CNY numbers.
fx_to_sgd = pd.DataFrame(index=calendar)
fx_to_sgd['USD'] = fx_bday['SGD=X']
fx_to_sgd['EUR'] = fx_bday['EURSGD=X']
fx_to_sgd['GBP'] = fx_bday['GBPSGD=X']
fx_to_sgd['JPY'] = fx_bday['SGD=X'] / fx_bday['JPY=X']
fx_to_sgd['CNY'] = fx_bday['SGD=X'] / fx_bday['CNY=X']
fx_to_sgd['HKD'] = fx_bday['HKDSGD=X']

prices_sgd = pd.DataFrame(index=calendar, columns=ALL_ASSETS, dtype=float)
for ticker, row in assets.iterrows():
    ccy = row['Local Currency']
    if ccy == 'SGD':
        prices_sgd[ticker] = local_bday[ticker]
    else:
        prices_sgd[ticker] = local_bday[ticker] * fx_to_sgd[ccy]

# All assets have common areas after the available prices; they are not filled backwards or artificially prolonged.
common_start = max(prices_sgd[col].first_valid_index() for col in ALL_ASSETS)
common_end = min(prices_sgd[col].last_valid_index() for col in ALL_ASSETS)
prices_sgd = prices_sgd.loc[common_start:common_end].dropna(how='any')
dollar_volume_sgd = prices_sgd * volume_bday.reindex(prices_sgd.index)
returns = prices_sgd.pct_change(fill_method=None).dropna(how='any')

# The investmentable market benchmark is converted uniformly to the total SSD return; VT is the United States dollar ETF, ES3.SI itself traded with SGD.
market_benchmark_local_bday = market_benchmark_close_local.reindex(calendar).ffill(limit=5)
benchmark_prices_sgd = pd.DataFrame(index=prices_sgd.index, dtype=float)
benchmark_prices_sgd['ES3.SI'] = prices_sgd['ES3.SI']
benchmark_prices_sgd['VT'] = (
    market_benchmark_local_bday['VT'].reindex(prices_sgd.index)
    * fx_to_sgd['USD'].reindex(prices_sgd.index)
)
if benchmark_prices_sgd[['ES3.SI', 'VT']].isna().all().any():
    raise RuntimeError('ES3.SI or VT lacks usable SGD prices.')
benchmark_returns_sgd = benchmark_prices_sgd.pct_change(fill_method=None).dropna(how='any')

portfolio_prices_sgd = prices_sgd.copy()
portfolio_returns_for_backtest = returns.copy()
# VIX retains a combination of the independent history prior to the start, so that the full three-year score can also be used for the first transaction date.
vix_data = vix_bday.loc[:returns.index[-1]].dropna(how='any')
if vix_data.empty or not returns.index.isin(vix_data.index).all():
    raise RuntimeError('The VIX pressure indicator does not cover all of the return dates.')

if len(returns) <= LOOKBACK:
    raise RuntimeError(f'Insufficient common return observations: {len(returns)}; require more than {LOOKBACK}.')

strategy_start = returns.index[LOOKBACK]
data_summary = pd.DataFrame({
    'Value': [common_start.date(), common_end.date(), strategy_start.date(), len(prices_sgd), len(returns)]
}, index=['Common price starting point', 'Common price destination', 'Practical starting point for the strategy (including 252-day pre-heat)', 'Common price observations', 'Co-benefit observations'])
display(data_summary)


vix_open_daily = vix_open_bday['^VIX'].reindex(returns.index)
