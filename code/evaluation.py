"""Final-product functions extracted from the completed research.

Loaded by the product runner into an explicit shared execution context.
"""

def drawdown_series(r):
    r = pd.Series(r, dtype=float).dropna()
    wealth = (1.0 + r).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth / running_peak - 1.0


def compute_metrics(result, start=None, end=None):
    r = result['returns'].loc[start:end].dropna()
    turnover = result['turnover'].loc[start:end]
    trade_legs = result.get('trade_legs', pd.Series(dtype=float)).loc[start:end]
    weights = result['weights'].loc[start:end]
    n = len(r)
    if n < 2:
        return pd.Series(dtype=float)
    years = n / TRADING_DAYS
    total_return = float((1.0 + r).prod() - 1.0)
    ending_value_sgd = INITIAL_CAPITAL_SGD * (1.0 + total_return)
    cumulative_pnl_sgd = ending_value_sgd - INITIAL_CAPITAL_SGD
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    annual_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    if 'risk_free_daily' not in result:
        raise RuntimeError(f"{result.get('name', 'Policy')}Lack of SORA daily risk-free rate of return.")
    rf_daily_series = result['risk_free_daily'].reindex(r.index).ffill()
    if rf_daily_series.isna().any():
        raise RuntimeError(f"{result.get('name', 'Policy')}SORA coverage is incomplete.")
    excess = r - rf_daily_series
    annual_rf = float((1.0 + rf_daily_series).prod() ** (1.0 / years) - 1.0)
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = (annual_return - annual_rf) / downside if downside > 0 else np.nan
    max_dd = drawdown_series(r).min()
    calmar = annual_return / abs(max_dd) if max_dd < 0 else np.nan
    annual_traded_notional = turnover.sum() / years
    annual_turnover = 0.5 * annual_traded_notional
    gross_exposure = result.get('gross_exposure', pd.Series(dtype=float)).loc[start:end]
    avg_gross = (gross_exposure.mean() if not gross_exposure.empty
                 else (weights.abs().sum(axis=1).mean() if not weights.empty else np.nan))

    return pd.Series({
        'Initial funding (SGD)': INITIAL_CAPITAL_SGD, 'End of period net value (SGD)': ending_value_sgd,
        'Cumulative gains and losses (SGD)': cumulative_pnl_sgd,
        'Cumulative Return': total_return, 'Annualized Return': annual_return, 'Annualized Volatility': annual_vol,
        'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino, 'Maximum Drawdown': max_dd,
        'Calmar Ratio': calmar, 'Positive-Day Ratio': (r > 0).mean(), 'Annual One-Way Turnover': annual_turnover,
        'Annual Two-Way Traded Notional': annual_traded_notional,
        'Number of Rebalances': len(turnover), 'Number of Trade Legs': int(trade_legs.sum()),
        'Average Gross Exposure': avg_gross,
    })

def drawdown_event_statistics(return_series):
    """Statistics of independent evacuations; one event begins with a break in historical peaks and ends with a return to peaks."""
    r = pd.Series(return_series, dtype=float).dropna()
    if r.empty:
        return {
            'Number of evacuations': 0, 'Number of retreats repaired': 0, 'Unrecovered evacuations at the end of the period': 0,
            'Average retreat range': np.nan, 'Average withdrawal duration (trade date)': np.nan,
            'Maximum withdrawal duration (trade date)': np.nan,
        }

    wealth = (1.0 + r).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_peak - 1.0
    underwater = drawdown < -1e-12
    events = []
    start_position = None

    for position, date in enumerate(drawdown.index):
        if bool(underwater.iloc[position]) and start_position is None:
            start_position = position
        recovery = (
            start_position is not None
            and not bool(underwater.iloc[position])
            and position > start_position
        )
        if recovery:
            episode = drawdown.iloc[start_position:position]
            events.append({
                'Start date': drawdown.index[start_position], 'End Date': date,
                'Maximum Drawdown': float(episode.min()),
                'Duration (trade date)': int(position - start_position + 1),
                'Rehabilitation': True,
            })
            start_position = None

    if start_position is not None:
        episode = drawdown.iloc[start_position:]
        events.append({
            'Start date': drawdown.index[start_position], 'End Date': drawdown.index[-1],
            'Maximum Drawdown': float(episode.min()),
            'Duration (trade date)': int(len(drawdown) - start_position),
            'Rehabilitation': False,
        })

    event_table = pd.DataFrame(events)
    if event_table.empty:
        return {
            'Number of evacuations': 0, 'Number of retreats repaired': 0, 'Unrecovered evacuations at the end of the period': 0,
            'Average retreat range': 0.0, 'Average withdrawal duration (trade date)': 0.0,
            'Maximum withdrawal duration (trade date)': 0,
        }
    return {
        'Number of evacuations': int(len(event_table)),
        'Number of retreats repaired': int(event_table['Rehabilitation'].sum()),
        'Unrecovered evacuations at the end of the period': int((~event_table['Rehabilitation']).sum()),
        'Average retreat range': float(event_table['Maximum Drawdown'].mean()),
        'Average withdrawal duration (trade date)': float(event_table['Duration (trade date)'].mean()),
        'Maximum withdrawal duration (trade date)': int(event_table['Duration (trade date)'].max()),
    }


def build_fifo_trade_ledger(result, metric_start, metric_end, initial_capital=None):
    """Reconstruct FIFO units from post-fee closing NAV and actual weights.

    Allocate the engine's recorded costs, not a second estimate of costs.
    Small sales financing fees are retained for reconciliation but flagged so
    they do not inflate ordinary completed-trade statistics.
    """
    initial_capital = INITIAL_CAPITAL_SGD if initial_capital is None else float(initial_capital)
    dates = result['returns'].dropna().loc[metric_start:metric_end].index
    lots = {ticker: [] for ticker in result.get('trade_assets', PORTFOLIO_ASSETS)}
    assets = list(lots)
    prices = result.get('instrument_prices', portfolio_prices_sgd).reindex(columns=assets)
    opening = prices.shift(1)
    before = result['pre_return_weights'].reindex(index=dates, columns=assets).fillna(0.0)
    closing = result['end_of_day_weights'].reindex(index=dates, columns=assets).fillna(0.0)
    deltas = result['trade_deltas'].reindex(index=dates, columns=assets).fillna(0.0)
    rates = pd.Series(result.get('asset_cost_rate', COST), index=assets, dtype=float)
    effective = result.get('effective_asset_cost_rate')
    fees = result.get('transaction_cost')
    nav, rows, lot_number, max_error = float(initial_capital), [], 0, 0.0
    positions = {date: i for i, date in enumerate(dates)}
    if len(dates):
        first = dates[0]
        for ticker in assets:
            weight = float(before.loc[first, ticker])
            if weight < -1e-10:
                raise ValueError('FIFO ledger supports long-only holdings.')
            if weight > 1e-10:
                price = float(opening.loc[first, ticker])
                if not np.isfinite(price) or price <= 0:
                    raise ValueError(f'Missing opening price for held asset {ticker}.')
                lot_number += 1
                lots[ticker].append({'Batch Number': lot_number, 'Date of purchase': first,
                    'Purchase price': price, 'Remaining share': weight * nav / price,
                    'Unit buy-in cost': 0.0, 'Inheritance at the beginning of the period': True})
    for date in dates:
        day_rates = effective.loc[date].reindex(assets).fillna(0.0) if effective is not None and date in effective.index else rates
        cost_weights = deltas.loc[date].abs() * day_rates
        fee_fraction = float(fees.get(date, 0.0)) if fees is not None else float(cost_weights.sum())
        if not 0 <= fee_fraction < 1:
            raise ValueError('Invalid recorded transaction-cost fraction.')
        nav_after = nav * (1.0 + float(result['returns'].loc[date]))
        nav_before = nav_after / (1.0 - fee_fraction)
        total_fee = nav_before - nav_after
        if total_fee > 1e-8 and cost_weights.sum() <= 0:
            raise ValueError('Recorded costs cannot be allocated to executed trades.')
        allocated = cost_weights * (total_fee / cost_weights.sum()) if cost_weights.sum() > 0 else cost_weights * 0
        for ticker in assets:
            available = sum(lot['Remaining share'] for lot in lots[ticker])
            target_value = float(closing.loc[date, ticker]) * nav_after
            if target_value < -1e-8:
                raise ValueError('FIFO ledger supports long-only holdings.')
            if available <= 1e-12 and abs(target_value) <= 1e-8 and allocated[ticker] <= 1e-8:
                continue
            price = float(prices.loc[date, ticker])
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f'Missing execution price for held asset {ticker} on {date}.')
            target = max(0.0, target_value / price)
            change = target - available
            fee = float(allocated[ticker])
            tolerance = max(1e-12, abs(nav_after) * 1e-12 / price)
            if change < -tolerance:
                selling = -change
                exit_cost_per_unit = fee / selling
                fee_funding_only = float(deltas.loc[date, ticker]) >= -1e-10
                while selling > tolerance and lots[ticker]:
                    lot = lots[ticker][0]
                    matched = min(selling, lot['Remaining share'])
                    invested = matched * (lot['Purchase price'] + lot['Unit buy-in cost'])
                    profit = matched * (price - exit_cost_per_unit) - invested
                    rows.append({'Policy': result['name'], 'Assets': ticker,
                        'Date of purchase': lot['Date of purchase'], 'Date of sale': date,
                        'Warehousing period (trade date)': positions[date] - positions[lot['Date of purchase']],
                        'Purchase price (SGD)': lot['Purchase price'], 'Sale price (SGD)': price,
                        'Match share': matched, 'Input amount (SGD)': invested,
                        'Net profit (SGD)': profit, 'Single net gain': profit / invested if invested > 0 else np.nan,
                        'Inheritance at the beginning of the period': bool(lot['Inheritance at the beginning of the period']),
                        'Fee funding only': fee_funding_only})
                    lot['Remaining share'] -= matched
                    selling -= matched
                    if lot['Remaining share'] <= tolerance:
                        lots[ticker].pop(0)
            elif change > tolerance:
                lot_number += 1
                lots[ticker].append({'Batch Number': lot_number, 'Date of purchase': date,
                    'Purchase price': price, 'Remaining share': change,
                    'Unit buy-in cost': fee / change,
                    'Inheritance at the beginning of the period': False})
            elif fee > 1e-8:
                # A tiny intended buy can exactly offset its fee-financing sale.
                if available <= 0:
                    raise ValueError('Cannot attach trading costs to an empty position.')
                for lot in lots[ticker]:
                    lot['Unit buy-in cost'] += fee / available
            error = abs(sum(lot['Remaining share'] for lot in lots[ticker]) * price - target_value)
            max_error = max(max_error, error)
            if error > max(1e-7, abs(nav_after) * 1e-9):
                raise RuntimeError(f'FIFO holdings do not reconcile for {ticker} on {date}.')
        nav = nav_after
    open_lots = [lot for group in lots.values() for lot in group if lot['Remaining share'] > 1e-12]
    diagnostics = {'Number of unprocessed authorizations at end of period': len(open_lots),
        'Unsettled assets at end of period': sum(bool(group) for group in lots.values()),
        'Maximum FIFO holding reconciliation error (SGD)': max_error}
    return pd.DataFrame(rows), diagnostics


def completed_trade_statistics(ledger, diagnostics):
    if not ledger.empty and 'Fee funding only' in ledger:
        ledger = ledger.loc[~ledger['Fee funding only']]
    if ledger.empty:
        return {
            'The odds.': np.nan, 'Total number of transactions completed': 0,
            'Average holding period (trade date)': np.nan,
            'Average profit for single transactions (SGD)': np.nan,
            'Average rate of return on individual transactions': np.nan,
            **diagnostics,
        }
    return {
        'The odds.': float((ledger['Net profit (SGD)'] > 0.0).mean()),
        'Total number of transactions completed': int(len(ledger)),
        'Average holding period (trade date)': float(ledger['Warehousing period (trade date)'].mean()),
        'Average profit for single transactions (SGD)': float(ledger['Net profit (SGD)'].mean()),
        'Average rate of return on individual transactions': float(ledger['Single net gain'].mean()),
        **diagnostics,
    }


def build_course_required_metrics(result_map, start, end):
    records, ledgers = [], {}
    for strategy_name, result in result_map.items():
        base = compute_metrics(result, start, end)
        ledger, diagnostics = build_fifo_trade_ledger(result, start, end)
        ledgers[strategy_name] = ledger
        trade_stats = completed_trade_statistics(ledger, diagnostics)
        drawdown_stats = drawdown_event_statistics(result['returns'].loc[start:end])
        records.append({
            'Policy': strategy_name,
            'Cumulative gains and losses (SGD)': base['Cumulative gains and losses (SGD)'],
            'Annualized Return': base['Annualized Return'],
            'Sharpe Ratio': base['Sharpe Ratio'],
            'Maximum Drawdown': base['Maximum Drawdown'],
            'The odds.': trade_stats['The odds.'],
            'Annualized Volatility': base['Annualized Volatility'],
            'Number of evacuations': drawdown_stats['Number of evacuations'],
            'Number of retreats repaired': drawdown_stats['Number of retreats repaired'],
            'Unrecovered evacuations at the end of the period': drawdown_stats['Unrecovered evacuations at the end of the period'],
            'Average withdrawal duration (trade date)': drawdown_stats['Average withdrawal duration (trade date)'],
            'Maximum withdrawal duration (trade date)': drawdown_stats['Maximum withdrawal duration (trade date)'],
            'Total number of transactions completed': trade_stats['Total number of transactions completed'],
            'Average holding period (trade date)': trade_stats['Average holding period (trade date)'],
            'Average profit for single transactions (SGD)': trade_stats['Average profit for single transactions (SGD)'],
            'Average rate of return on individual transactions': trade_stats['Average rate of return on individual transactions'],
            'Number of unprocessed authorizations at end of period': trade_stats['Number of unprocessed authorizations at end of period'],
            'Unsettled assets at end of period': trade_stats['Unsettled assets at end of period'],
            'Number of Rebalances': base['Number of Rebalances'],
            'Number of Trade Legs': base['Number of Trade Legs'],
        })
    return pd.DataFrame(records).set_index('Policy'), ledgers
