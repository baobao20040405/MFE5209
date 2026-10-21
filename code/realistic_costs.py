"""Frozen-path cost sensitivity; assumptions copied from explore/config.py.

Not a fresh execution simulation: signals, weights and trades remain unchanged.
FX is charged on every foreign-currency trade, with no currency cash netting.
Annual tax drags are proxies, not jurisdiction-specific tax calculations.
"""
import numpy as np
import pandas as pd

COMMISSION = dict(zip(['United States', 'Germany', 'United Kingdom', 'Japan',
    'China', 'Hong Kong SAR', 'Singapore', 'Global', 'Euro Area'],
    [.0002, .0008, .0005, .0005, .0003, .0008, .0006, .0004, .0004]))
SPREAD = dict(zip(COMMISSION, [.0003, .0008, .0008, .0006, .001, .0008, .0008, .0005, .0005]))
FX = {'SGD': 0., 'USD': .001, 'EUR': .0015, 'GBP': .0015,
      'JPY': .0015, 'CNY': .002, 'HKD': .0015}
TAX = dict(zip(['SPY', 'QQQ', 'IWM', 'EXS1.DE', 'ISF.L', '1308.T',
    '510300.SS', '2800.HK', 'ES3.SI', 'IEF', '511010.SS', 'USO', 'DBB',
    'GLD', 'UUP', 'FXY', 'FXE', 'BTC-USD'],
    [.0036, .0018, .0036, .0038, 0., .003, .0025, 0., 0., .0105,
     .002, 0., 0., 0., .0015, .0015, .0015, 0.]))
# VT uses the US equity distribution-tax proxy; this ETF was not in explore's 18 assets.
TAX['VT'] = .0036


def cost_schedule(metadata):
    frame = metadata[['Market', 'Local Currency']].copy()
    frame['Commission'] = frame['Market'].map(COMMISSION)
    frame['Spread and Slippage'] = frame['Market'].map(SPREAD)
    frame['FX Conversion'] = frame['Local Currency'].map(FX)
    if 'BTC-USD' in frame.index:
        frame.loc['BTC-USD', ['Commission', 'Spread and Slippage']] = .0015
    frame['Annual Tax Drag'] = frame.index.to_series().map(TAX)
    if frame.isna().any().any():
        raise ValueError('Incomplete realistic-cost assumptions.')
    frame['Static One-Way Rate'] = frame[['Commission', 'Spread and Slippage', 'FX Conversion']].sum(axis=1)
    return frame


def replay_costs(result, context, dates, capital, schedule):
    """Reprice frozen executed weights, starting scenario NAV at test capital.

    Opening inherited positions are not charged again. Order notionals use
    pre-cost, post-market-return NAV. Lagged volume/volatility avoids lookahead.
    Missing ADV on an actual trade is an error, never zero impact.
    """
    tickers = list(result['end_of_day_weights'].columns)
    rates = schedule.loc[tickers]
    delta = result['trade_deltas'].reindex(index=dates, columns=tickers).fillna(0.)
    held = result['pre_return_weights'].reindex(index=dates, columns=tickers)
    old_fee = result['transaction_cost'].reindex(dates).fillna(0.)
    gross = (1 + result['returns'].reindex(dates)) / (1 - old_fee) - 1
    prices = context['prices_sgd'].copy()
    prices['VT'] = context['benchmark_prices_sgd']['VT']
    volumes = context['raw']['Volume'].reindex(index=prices.index, columns=tickers)
    # Exclude the current session; do not use forward-filled volume as observations.
    adv = (prices[tickers] * volumes).shift(1).rolling(20, min_periods=5).median().reindex(dates)
    vol = prices[tickers].pct_change(fill_method=None).shift(1).rolling(20, min_periods=5).std().reindex(dates)
    nav, rows, net = float(capital), [], []
    for date in dates:
        change = delta.loc[date].abs()
        active = change > 1e-12
        if ((adv.loc[date, active] <= 0) | ~np.isfinite(adv.loc[date, active])
                | ~np.isfinite(vol.loc[date, active])).any():
            raise ValueError(f'Missing lagged liquidity inputs for realistic-cost trades on {date.date()}.')
        pre_cost_nav = nav * (1 + gross.loc[date])
        unit_impact = pd.Series(0., index=tickers)
        unit_impact.loc[active] = (.10 * vol.loc[date, active] * np.sqrt(
            change[active] * pre_cost_nav / adv.loc[date, active])).clip(upper=.01)
        row = {key: float(change @ rates[key]) for key in
               ['Commission', 'Spread and Slippage', 'FX Conversion']}
        row['Market Impact'] = float(change @ unit_impact)
        row['Tax Drag'] = float(held.loc[date] @ rates['Annual Tax Drag']) / 252
        fee = sum(row[key] for key in row if key != 'Tax Drag')
        value = (1 + gross.loc[date] - row['Tax Drag']) * (1 - fee) - 1
        if not np.isfinite(value) or value <= -1 or fee >= 1:
            raise ValueError('Invalid realistic-cost return.')
        row['Transaction Cost'] = fee
        rows.append(row); net.append(value); nav *= 1 + value
    output = dict(result)
    output['returns'] = pd.Series(net, index=dates, name=result['name'])
    components = pd.DataFrame(rows, index=dates)
    output['transaction_cost'] = components['Transaction Cost']
    return output, components
