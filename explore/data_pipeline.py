"""SORA, asset pool, situational downloads and SSD total return data.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# 1.2 Day SORA no-risk interest rate (all policies used)
#
#
# The fixed 1.40 per cent risk-free rate has been deleted. All strategies use `SORA_daily.csv` for cash gains, leverage financing costs, Sharp ratio and Sotino ratio. The original SORA value is an annual percentage; in order to avoid future information, only SORA, which was formed on the previous Singapore business day, is used for the return date. Cash gains are calculated on the basis of the actual calendar days and the ACT/365 caliber; the lock-in test period SORA is allowed to be loaded only in the final test module.
# --- Original Notebook code unit 4 ---
SORA_FILE = Path.cwd() / 'SORA_daily.csv'


def load_sora_daily(max_date):
    if not SORA_FILE.exists():
        raise FileNotFoundError(f'Missing day SORA file:{SORA_FILE}')
    frame = pd.read_csv(SORA_FILE)
    required_columns = {'Date', 'SORA'}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f'SORA files must contain columns:{sorted(required_columns)}')
    frame = frame[['Date', 'SORA']].copy()
    frame['Date'] = pd.to_datetime(frame['Date'], errors='raise')
    if frame['Date'].duplicated().any():
        duplicated = frame.loc[frame['Date'].duplicated(keep=False), 'Date'].dt.date.tolist()[:5]
        raise ValueError(f'SORA has repeated dates, for example:{duplicated}')
    frame['SORA'] = pd.to_numeric(frame['SORA'], errors='raise')
    if frame['SORA'].isna().any() or (frame['SORA'] <= -100.0).any():
        raise ValueError('SORA contains missing values or an unlikely annualized interest rate.')
    frame = frame.sort_values('Date').set_index('Date').loc[:pd.Timestamp(max_date)].copy()
    frame = frame.rename(columns={'SORA': 'SORA annualized rate (%)'})
    # SORA is the overnight interest rate; in order to avoid a dispute as to the date of publication, only the last formed SORA is used.
    frame['Annualized SORA interest rate available for retrospect (%)'] = frame['SORA annualized rate (%)'].shift(1)
    frame['Single-day risk-free rate of return (ACT/365)'] = frame['Annualized SORA interest rate available for retrospect (%)'] / 100.0 / 365.0
    frame = frame.dropna(subset=['Annualized SORA interest rate available for retrospect (%)', 'Single-day risk-free rate of return (ACT/365)'])
    if frame.empty:
        raise RuntimeError('Day SORA data is empty.')
    return frame


def align_sora_returns(index, sora_frame, allow_holdout=False, annual_spread=0.0):
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return pd.Series(dtype=float, index=index, name='Day-free rate of return')
    if index.max() >= HOLDOUT_START and not allow_holdout:
        raise RuntimeError('The reading of SORA data outside the locked sample is prohibited at the research stage.')
    if sora_frame.index.max() >= HOLDOUT_START and not allow_holdout:
        raise RuntimeError('SORA DataFrame, which was introduced during the research phase, contains lock-in data.')
    annual_rate = (
        sora_frame['Annualized SORA interest rate available for retrospect (%)'].reindex(index).ffill() / 100.0
        + float(annual_spread)
    )
    if annual_rate.isna().any():
        missing_dates = annual_rate.index[annual_rate.isna()][:5].date.tolist()
        raise RuntimeError(f'SORA cannot overwrite the policy date. Example:{missing_dates}')
    calendar_days = index.to_series().diff().dt.days.astype(float)
    calendar_days.iloc[0] = 1.0
    if (calendar_days <= 0).any():
        raise RuntimeError('The strategic dates must be strictly incremental in order to calculate SORA against ACT/365.')
    daily = annual_rate * calendar_days.to_numpy() / 365.0
    label = 'Rate of return on day financing' if annual_spread else 'Day-free rate of return'
    return pd.Series(daily.to_numpy(), index=index, name=label)


# The physical cut-off at the research stage does not leave SORA in the study of DataFrame.
sora_daily_df = load_sora_daily(RESEARCH_END)
assert sora_daily_df.index.max() < HOLDOUT_START
display(sora_daily_df.tail())
print(f'Research-period SORA: {sora_daily_df.index.min().date()} to {sora_daily_df.index.max().date()}; no fixed risk-free-rate proxy is used.')

# 1.3 Original asset pool and ETF mapping
# --- Original Notebook code unit 6--

asset_rows = [
    ('Equities', 'United States', 'S&P 500', 'SPY', 'SPDR S&P 500 ETF Trust', 'USD'),
    ('Equities', 'United States', 'NASDAQ 100', 'QQQ', 'Invesco QQQ Trust', 'USD'),
    ('Equities', 'United States', 'Russell 2000', 'IWM', 'iShares Russell 2000 ETF', 'USD'),
    ('Equities', 'Germany', 'DAX, Germany', 'EXS1.DE', 'iShares Core DAX UCITS ETF', 'EUR'),
    ('Equities', 'United Kingdom', 'FTSE 100', 'ISF.L', 'iShares Core FTSE 100 UCITS ETF', 'GBP'),
    ('Equities', 'Japan', 'TOPIX', '1308.T', 'Listed Index Fund TOPIX', 'JPY'),
    ('Equities', 'China', 'CSI 300', '510300.SS', 'Huatai-PineBridge CSI 300 ETF', 'CNY'),
    ('Equities', 'Hong Kong SAR', 'Hang Seng Index', '2800.HK', 'Tracker Fund of Hong Kong', 'HKD'),
    ('Equities', 'Singapore', 'Straits Times Index', 'ES3.SI', 'SPDR Straits Times Index ETF', 'SGD'),
    ('Bonds', 'United States', 'United States Treasury 7-10 years', 'IEF', 'iShares 7-10 Year Treasury Bond ETF', 'USD'),
    ('Bonds', 'China', 'China 5-Year Government Bond', '511010.SS', 'Government Bond ETF', 'CNY'),
    ('Commodities', 'Global', 'Energy (oil)', 'USO', 'United States Oil Fund', 'USD'),
    ('Commodities', 'Global', 'Industrial metals', 'DBB', 'Invesco DB Base Metals Fund', 'USD'),
    ('Commodities', 'Global', 'Gold', 'GLD', 'SPDR Gold Shares', 'USD'),
    ('FX', 'United States', 'United States dollar index', 'UUP', 'Invesco DB US Dollar Index Bullish Fund', 'USD'),
    ('FX', 'Japan', 'Japanese yen', 'FXY', 'Invesco CurrencyShares Japanese Yen Trust', 'USD'),
    ('FX', 'Euro Area', 'Euro', 'FXE', 'Invesco CurrencyShares Euro Trust', 'USD'),
    ('Crypto', 'Global', 'Bitcoin', 'BTC-USD', 'Bitcoin', 'USD'),
]

assets = pd.DataFrame(asset_rows, columns=['Asset Class', 'Market', 'Asset Name', 'Code', 'ETF English Name', 'Local Currency']).set_index('Code')
display(assets)

ALL_ASSETS = assets.index.tolist()
EQUITIES = assets.query("`Asset Class` == 'Equities'").index.tolist()
BONDS = assets.query("`Asset Class` == 'Bonds'").index.tolist()

print(f'Equities: {len(EQUITIES)}; bonds: {len(BONDS)}; total risky assets: {len(ALL_ASSETS)}.')


PORTFOLIO_ASSETS = ALL_ASSETS.copy()
asset_metadata = assets.copy()

print(f'Formal asset universe: {len(PORTFOLIO_ASSETS)} instruments.')

# 1.4 Reversible price, exchange rate conversion and global valuation calendar
#
#
# Yahoo Finance's ETF prices are denominated in the respective market currency. Each local currency price is multiplied by the corresponding exchange rate of “how much SDD is convertible per unit of foreign currency” and the net value of SSD assets is uniformly obtained.
#
# Inter-market break days vary, so the global valuation calendar is used from Monday to Friday, with up to five working days in advance only for the most recent prices that have occurred. That treatment would not use future prices; bitcoin would also be valued only on working days, and the cumulative change over weekends would be reflected in the next working day.
# --- Original Notebook code unit 8--

FX_DOWNLOAD = ['SGD=X', 'EURSGD=X', 'GBPSGD=X', 'JPY=X', 'CNY=X', 'HKDSGD=X']
VIX_TICKERS = ['^VIX', '^VIX3M']
MARKET_BENCHMARK_TICKERS = ['VT']
download_tickers = PORTFOLIO_ASSETS + FX_DOWNLOAD + VIX_TICKERS + MARKET_BENCHMARK_TICKERS
raw = yf.download(
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
    raise RuntimeError('ZXQ0QZSI or VT market benchmarks do not have available SGD prices.')
benchmark_returns_sgd = benchmark_prices_sgd.pct_change(fill_method=None).dropna(how='any')

portfolio_prices_sgd = prices_sgd.copy()
portfolio_returns_for_backtest = returns.copy()
# VIX retains a combination of the independent history prior to the start, so that the full three-year score can also be used for the first transaction date.
vix_data = vix_bday.loc[:returns.index[-1]].dropna(how='any')
if vix_data.empty or not returns.index.isin(vix_data.index).all():
    raise RuntimeError('The VIX pressure indicator does not cover all of the return dates.')

if len(returns) <= LOOKBACK:
    raise RuntimeError(f'Common effective benefits only{len(returns)}Shit, that\'s not enough.{LOOKBACK}Day risk estimation window.')

strategy_start = returns.index[LOOKBACK]
data_summary = pd.DataFrame({
    'Value': [common_start.date(), common_end.date(), strategy_start.date(), len(prices_sgd), len(returns)]
}, index=['Common price starting point', 'Common price destination', 'Practical starting point for the strategy (including 252-day pre-heat)', 'Common price observations', 'Co-benefit observations'])
display(data_summary)

