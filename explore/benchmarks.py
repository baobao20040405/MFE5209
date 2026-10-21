"""Currency P&L, transaction versus market benchmark.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# Part VI Summary of performance versus market comparison
#
# 6.1 Supplementary display of monetary amounts P&L and trading activities
#
#
# This section does not create new strategies or recalculate another set of benefits. It converts the same line of net proceeds used for this part of the harmonized indicator into net end-of-period, cumulative and daily gains and losses on the assumption of initial funding** SGD 1,000,000** and separates the number of “ports of asset transactions” from the “number of portfolio transfers”.
#
# The harmonized table of indicators is the official calibre; this section is a supplementary view of monetary amounts and transactions, showing all 12 formal strategies.
# --- Original Notebook code unit 69-
def build_pnl_activity_summary(result_map, start, end):
    records = []
    pnl_paths = {}
    for strategy_name, result in result_map.items():
        r = result['returns'].loc[start:end].dropna()
        if r.empty:
            continue
        nav = INITIAL_CAPITAL_SGD * (1.0 + r).cumprod()
        daily_pnl = nav.diff()
        daily_pnl.iloc[0] = nav.iloc[0] - INITIAL_CAPITAL_SGD
        turnover = result['turnover'].loc[start:end]
        trade_legs = result.get('trade_legs', pd.Series(dtype=float)).loc[start:end]
        records.append({
            'Policy': strategy_name,
            'Initial funding (SGD)': INITIAL_CAPITAL_SGD,
            'End of period net value (SGD)': float(nav.iloc[-1]),
            'Cumulative gains and losses (SGD)': float(nav.iloc[-1] - INITIAL_CAPITAL_SGD),
            'Average daily surplus/loss (SGD)': float(daily_pnl.mean()),
            'Maximum single-day gain/loss (SGD)': float(daily_pnl.min()),
            'Positive-Day Ratio': float((r > 0.0).mean()),
            'Number of Rebalances': int(len(turnover)),
            'Number of Trade Legs': int(trade_legs.sum()),
        })
        pnl_paths[strategy_name] = nav - INITIAL_CAPITAL_SGD
    return pd.DataFrame(records).set_index('Policy'), pd.DataFrame(pnl_paths)


research_pnl_summary, research_pnl_paths = build_pnl_activity_summary(
    results, strategy_start, RESEARCH_END
)
holdout_pnl_summary, holdout_pnl_paths = build_pnl_activity_summary(
    holdout_results, HOLDOUT_START, actual_holdout_end
)


def format_pnl_activity_table(table):
    shown = table.copy()
    for column in ['Initial funding (SGD)', 'End of period net value (SGD)', 'Cumulative gains and losses (SGD)',
                   'Average daily surplus/loss (SGD)', 'Maximum single-day gain/loss (SGD)']:
        shown[column] = shown[column].map(lambda x: f'S${x:,.2f}')
    shown['Positive-Day Ratio'] = shown['Positive-Day Ratio'].map(lambda x: f'{x:.2%}')
    return shown


print(f'Study the complete range:{strategy_start.date()}to{RESEARCH_END.date()}')
display(format_pnl_activity_table(research_pnl_summary))
print(f'Lock test range:{HOLDOUT_START.date()}to{actual_holdout_end.date()}')
display(format_pnl_activity_table(holdout_pnl_summary))

fig, ax = plt.subplots(figsize=(16, 8))
for strategy_name in final_strategy_names if 'final_strategy_names' in globals() else list(results):
    research_style = HIGHLIGHT_COLORS.get(strategy_name, MUTED_COLOR)
    linewidth = 3.4 if strategy_name == SUBMISSION_STRATEGY else (2.8 if strategy_name == BASELINE_STRATEGY else 1.35)
    alpha = 1.0 if strategy_name in {BASELINE_STRATEGY, SUBMISSION_STRATEGY} else 0.58
    ax.plot(research_pnl_paths.index, research_pnl_paths[strategy_name], linestyle='-',
            color=research_style, linewidth=linewidth, alpha=alpha,
            label=strategy_label(strategy_name))
ax.axhline(0.0, color='#777777', linewidth=0.8)
ax.set_title(f'All {len(results)} strategies: study complete inter-district cumulative P&L (initial SGD 1.0 million)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative P&L (SGD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'S${value:,.2f}'))
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(16, 7))
for strategy_name in holdout_pnl_paths.columns:
    color = HIGHLIGHT_COLORS.get(strategy_name, MUTED_COLOR)
    linewidth = 3.4 if strategy_name == SUBMISSION_STRATEGY else (2.8 if strategy_name == BASELINE_STRATEGY else 1.35)
    alpha = 1.0 if strategy_name in {BASELINE_STRATEGY, SUBMISSION_STRATEGY} else 0.58
    ax.plot(holdout_pnl_paths.index, holdout_pnl_paths[strategy_name], linestyle='-',
            color=color, linewidth=linewidth, alpha=alpha, label=strategy_label(strategy_name))
ax.axhline(0.0, color='#777777', linewidth=0.8)
ax.set_title(f'All {len(results)} strategies: Accumulated P&L (initial SGD 1.0 million)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative P&L (SGD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'S${value:,.2f}'))
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()

print('Note: This section uses the same net income as the harmonized indicator table; the amount of P&L is only expressed as the SDD of the rate of return.')
print('Note: The proportion of positive returns on a daily basis is not a trade-off; the number of legs of an asset transaction is not a number of times a combination of a warehouse.')

# 6.2 Indicators of transaction and evacuation incidents as required by the course
#
#
# The four transactional indicators listed in this section and previously not strictly covered, as well as the frequency of evacuations, do not change any strategy, weight, signal or return returns.
#
# 6.2.1 Indicator calibre
#
# -** Definition of transaction**: each increase forms an independent hold-up batch; first-in-first-out (FIFO) matches the first batch. A single sale that matches more than one purchase order will result in more than one completed transaction.
# - **The trade-off**: net profit is greater than 0 per cent of the completed FIFO transaction batch, after deducting the cost of buying and selling the unilateral transaction.
# - **The total number of transactions**: the number of FIFO transactions that have been completed at the level of the return zone; the unsold batch is shown separately as “the number of unsettled shipments at the end of the period”, the non-transaction rate and the single profit.
# - ** Average holding period**: average number of trading days from purchase to date of sale of completed batches.
# - ** Average profit for individual transactions**: average net profit for completed transactions, expressed in SSD amounts and relative capital investment return.
# - ** Recovery event**: Net value starts when it first breaks its historical peak and ends when it returns to or exceeds its previous peak; withdrawals that remain unrecovered at the end of the evaluation are counted separately as unrecovered events.
# - ** Sub-sector starting batch**: “Inheritance batch” is established at the starting point of the zone at the closing price and at the unified beginning of the period, if the assets are held before the beginning of the evaluation sub-area. The hold-up period is an in-house observational holding period and does not pretend to be a full historical hold-up period.
#
# The turnover here is different from the “proportion of positive returns on a daily basis”; the total number of transactions is different from the number of combinations and legs of asset transactions. Qualitative indicators apply to comparison of transactions, but there may be more uncompleted lead times for long-term, continuous-held asset allocation strategies, which must be explained at the same time as unsettled amounts at the end of the period.
#
# >FIFO statistics only describe the number of asset batches achieved in the rebalancing process, do not represent Alpha for independent trading signals and should not be used as the main basis for recommended strategies.
# --- Original Notebook code unit 71-
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


def build_fifo_trade_ledger(result, metric_start, metric_end, initial_capital=INITIAL_CAPITAL_SGD):
    """Reconstruct FIFO units from post-fee closing NAV and actual weights.

    Allocate the engine's recorded costs, not a second estimate of costs.
    Small sales financing fees are retained for reconciliation but flagged so
    they do not inflate ordinary completed-trade statistics.
    """
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


def format_course_required_metrics(table):
    shown = table.copy()
    for column in ['Annualized Return', 'Maximum Drawdown', 'The odds.', 'Annualized Volatility', 'Average rate of return on individual transactions']:
        shown[column] = shown[column].map(lambda value: '' if pd.isna(value) else f'{value:.2%}')
    shown['Sharpe Ratio'] = shown['Sharpe Ratio'].map(lambda value: f'{value:.2f}')
    for column in ['Cumulative gains and losses (SGD)', 'Average profit for single transactions (SGD)']:
        shown[column] = shown[column].map(lambda value: '' if pd.isna(value) else f'S${value:,.2f}')
    for column in ['Average withdrawal duration (trade date)', 'Maximum withdrawal duration (trade date)',
                   'Average holding period (trade date)']:
        shown[column] = shown[column].map(lambda value: '' if pd.isna(value) else f'{value:.2f}')
    integer_columns = [
        'Number of evacuations', 'Number of retreats repaired', 'Unrecovered evacuations at the end of the period',
        'Total number of transactions completed', 'Number of unprocessed authorizations at end of period', 'Unsettled assets at end of period',
        'Number of Rebalances', 'Number of Trade Legs',
    ]
    for column in integer_columns:
        shown[column] = shown[column].astype(int)
    return shown


research_course_metrics, research_trade_ledgers = build_course_required_metrics(
    results, strategy_start, RESEARCH_END
)
holdout_course_metrics, holdout_trade_ledgers = build_course_required_metrics(
    holdout_results, HOLDOUT_START, actual_holdout_end
)

print(f'Study full inter-territory content indicators:{strategy_start.date()}to{RESEARCH_END.date()}')
display(format_course_required_metrics(research_course_metrics))
print(f'Lock test inter-temporal content indicators:{HOLDOUT_START.date()}to{actual_holdout_end.date()}')
display(format_course_required_metrics(holdout_course_metrics))

# The details of the main policy are kept as auditable DataFrame and are not written in local files.
baseline_name = SUBMISSION_STRATEGY
baseline_research_trade_ledger = research_trade_ledgers[baseline_name]
baseline_holdout_trade_ledger = holdout_trade_ledgers[baseline_name]
print('Submitting a list of transactions completed by FIFO between candidate strategy studies (first 20 lines):')
display(baseline_research_trade_ledger.head(20).style.format({
    'Purchase price (SGD)': '{:,.2f}', 'Sale price (SGD)': '{:,.2f}',
    'Match share': '{:,.2f}', 'Input amount (SGD)': 'S${:,.2f}',
    'Net profit (SGD)': 'S${:,.2f}', 'Single net gain': '{:.2%}',
}))

if research_course_metrics['Total number of transactions completed'].sum() <= 0:
    raise RuntimeError('The transaction book does not identify any completed transactions. Please check FIFO reconstruction logic.')
if research_course_metrics['Number of evacuations'].sum() <= 0:
    raise RuntimeError('No evacuations have been identified. Please check the evacuation logic.')
print('The indicators for missing items have been added: the success of transactions, the total number of transactions, the average holding period, the average profit of individual transactions and the frequency of evacuations.')

# 6.3 Market benchmark: Singapore equities versus global equities
#
#
# This section uses two SSD total returns that can be invested in ETF as the market benchmark and maintains exactly the same trading date and valuation currency with all official strategies:
#
# - **ES3.SI**: SPDR Times Index ETF, representing the Singapore stock market;
# - ** VT**: Vanguard Total World Stock ETF, representing the global stock market.
#
# ES3.SI directly uses the SPD compound price; VT's United States dollar compound price is converted to SSD at the USSD/SPD exchange rate for the same period. Alpha and Beta used daily earnings estimates after the deduction of the daily SORA, and Alpha used the same cut-off distance of 252 days for arithmetical anaesthesia; the information ratio was based on the active daily gain of the strategic reduction baseline; and the monthly gain calculation was used for increases and decreases in catch rates. The duration of the study was more stringently closed to 2026-07-31, with the locked test range being evaluated separately only after the completion of the locked test module.
#
# > Interpret limits: This strategy is a multi-asset portfolio, with stocks in Beta significantly below 1; Alpha, as compared to the estimate of the single equity index, includes differences between asset allocation and risk levels and cannot be fully interpreted as Alpha. ES3.SI and VT are retained as external market benchmarks; 18 rights to assets etc., 60/40, all-asset classic risk parities and random Top10 are used to test strategic structure and momentum screening contributions.
#
# The annual revenue portion retains all 12 strategies and two benchmarks, DataFrame; to ensure the readability of the classroom projection, the column chart is in the form of a single grouping of a single axis. The end year is incomplete and is clearly marked with an asterisk.
# --- Original Notebook code unit 73-
MARKET_BENCHMARK_NAMES = {
    'ES3.SI': 'Singapore Stock Market (ES3.SI)',
    'VT': 'Global stock market (VT)',
}
SUBMISSION_STRATEGY = DYNAMIC_ATTACK_H20_NAME


def annualized_compound_return(return_series):
    return_series = pd.Series(return_series, dtype=float).dropna()
    if len(return_series) < 2:
        return np.nan
    return float((1.0 + return_series).prod() ** (TRADING_DAYS / len(return_series)) - 1.0)


def benchmark_relative_metrics(result, benchmark_return, start=None, end=None):
    strategy_return = result['returns'].loc[start:end].rename('Strategic gains')
    benchmark_return = benchmark_return.loc[start:end].rename('Baseline proceeds')
    risk_free = result['risk_free_daily'].loc[start:end].rename('SORA')
    aligned = pd.concat([strategy_return, benchmark_return, risk_free], axis=1).dropna()
    if len(aligned) < 20:
        return pd.Series(dtype=float)

    strategy_excess = aligned['Strategic gains'] - aligned['SORA']
    benchmark_excess = aligned['Baseline proceeds'] - aligned['SORA']
    benchmark_variance = float(benchmark_excess.var(ddof=1))
    beta = (float(strategy_excess.cov(benchmark_excess)) / benchmark_variance
            if benchmark_variance > 0 else np.nan)
    daily_alpha = float(strategy_excess.mean() - beta * benchmark_excess.mean())
    annual_alpha = daily_alpha * TRADING_DAYS

    active_return = aligned['Strategic gains'] - aligned['Baseline proceeds']
    tracking_error = float(active_return.std(ddof=1) * np.sqrt(TRADING_DAYS))
    information_ratio = (float(active_return.mean() / active_return.std(ddof=1)
                               * np.sqrt(TRADING_DAYS))
                         if active_return.std(ddof=1) > 0 else np.nan)

    monthly = (1.0 + aligned[['Strategic gains', 'Baseline proceeds']]).resample('ME').prod() - 1.0
    up_months = monthly['Baseline proceeds'] > 0.0
    down_months = monthly['Baseline proceeds'] < 0.0
    upside_capture = (float(monthly.loc[up_months, 'Strategic gains'].mean()
                            / monthly.loc[up_months, 'Baseline proceeds'].mean())
                      if up_months.any() else np.nan)
    downside_capture = (float(monthly.loc[down_months, 'Strategic gains'].mean()
                              / monthly.loc[down_months, 'Baseline proceeds'].mean())
                        if down_months.any() else np.nan)

    strategy_total = float((1.0 + aligned['Strategic gains']).prod() - 1.0)
    benchmark_total = float((1.0 + aligned['Baseline proceeds']).prod() - 1.0)
    strategy_annual = annualized_compound_return(aligned['Strategic gains'])
    benchmark_annual = annualized_compound_return(aligned['Baseline proceeds'])
    return pd.Series({
        'Common trading day': len(aligned),
        'Strategy cumulative rate of return': strategy_total,
        'Baseline cumulative rate of return': benchmark_total,
        'Cumulative excess gains': strategy_total - benchmark_total,
        'Strategy annualized rate of return': strategy_annual,
        'Base yearized rate of return': benchmark_annual,
        'Annualized positive returns': strategy_annual - benchmark_annual,
        'Alpha (calculation)': annual_alpha,
        'Beta': beta,
        'Information ratio': information_ratio,
        'Yearly tracking error': tracking_error,
        'Increase catch rate': upside_capture,
        'Lower catch rate': downside_capture,
        'Alpha/Beta used SORA': True,
    })


# The full formal strategy is to be retained as an ordinary DataFrame that can continue to be filtered.
market_benchmark_records = []
for strategy_name, result in results.items():
    for benchmark_ticker, benchmark_label in MARKET_BENCHMARK_NAMES.items():
        metric = benchmark_relative_metrics(
            result,
            benchmark_returns_sgd[benchmark_ticker],
            strategy_start,
            RESEARCH_END,
        )
        record = metric.to_dict()
        record.update({
            'Policy': strategy_name,
            'Market benchmarks': benchmark_label,
            'Comparison': 'Study the complete range.',
        })
        market_benchmark_records.append(record)

market_benchmark_metrics = pd.DataFrame(market_benchmark_records).set_index(
    ['Market benchmarks', 'Policy']
).sort_index()

# Towards final submission: presentation of recommended strategy sub-species, validation period, full research period and lock-in test period.
submission_benchmark_records = []
submission_periods = {
    'In Sample': (None, IN_SAMPLE_END, results[SUBMISSION_STRATEGY]),
    'Validation interval': (OOS_START, RESEARCH_END, results[SUBMISSION_STRATEGY]),
    'Study the complete range.': (None, RESEARCH_END, results[SUBMISSION_STRATEGY]),
    'Lock Test Range': (HOLDOUT_START, actual_holdout_end, holdout_results[SUBMISSION_STRATEGY]),
}
for period_name, (start, end, result) in submission_periods.items():
    for benchmark_ticker, benchmark_label in MARKET_BENCHMARK_NAMES.items():
        metric = benchmark_relative_metrics(
            result, benchmark_returns_sgd[benchmark_ticker], start, end
        )
        record = metric.to_dict()
        record.update({'Period': period_name, 'Market benchmarks': benchmark_label})
        submission_benchmark_records.append(record)

submission_benchmark_comparison = pd.DataFrame(submission_benchmark_records).set_index(
    ['Period', 'Market benchmarks']
)


def format_benchmark_table(frame):
    formatted = frame.copy()
    percent_columns = [
        'Strategy cumulative rate of return', 'Baseline cumulative rate of return', 'Cumulative excess gains',
        'Strategy annualized rate of return', 'Base yearized rate of return', 'Annualized positive returns',
        'Alpha (calculation)', 'Yearly tracking error', 'Increase catch rate', 'Lower catch rate',
    ]
    for column in percent_columns:
        formatted[column] = formatted[column].map(lambda value: f'{value:.2%}')
    for column in ['Beta', 'Information ratio']:
        formatted[column] = formatted[column].map(lambda value: f'{value:.2f}')
    formatted['Common trading day'] = formatted['Common trading day'].astype(int)
    return formatted


display(format_benchmark_table(submission_benchmark_comparison))

# Cumulative inter-research benefits: recommended strategies use product main coarse lines and all sequences use solid lines.
benchmark_plot_returns = pd.concat({
    SUBMISSION_STRATEGY: results[SUBMISSION_STRATEGY]['returns'],
    MARKET_BENCHMARK_NAMES['ES3.SI']: benchmark_returns_sgd['ES3.SI'],
    MARKET_BENCHMARK_NAMES['VT']: benchmark_returns_sgd['VT'],
}, axis=1).loc[strategy_start:RESEARCH_END].dropna(how='any')
benchmark_cumulative = (1.0 + benchmark_plot_returns).cumprod() - 1.0

benchmark_plot_style = {
    SUBMISSION_STRATEGY: {'color': STRATEGY_COLORS[SUBMISSION_STRATEGY], 'linewidth': 3.6, 'linestyle': '-'},
    MARKET_BENCHMARK_NAMES['ES3.SI']: {'color': MARKET_BENCHMARK_COLORS['ES3.SI'], 'linewidth': 2.4, 'linestyle': '-'},
    MARKET_BENCHMARK_NAMES['VT']: {'color': MARKET_BENCHMARK_COLORS['VT'], 'linewidth': 2.4, 'linestyle': '-'},
}
fig, ax = plt.subplots(figsize=(15, 8))
for column in benchmark_cumulative:
    ax.plot(benchmark_cumulative.index, benchmark_cumulative[column],
            label=column, **benchmark_plot_style[column])
ax.axvline(IN_SAMPLE_END, color='#AAB2BD', linestyle='-', linewidth=1.0,
           label='The end point in the sample')
ax.axhline(0.0, color='#777777', linewidth=0.8)
ax.set_title('Recommended strategies and accumulated gains from market benchmarks (total SSD return, reduced strategic transaction costs)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
style_axis(ax)
place_legend(ax)
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()


# Annual returns on all 12 strategies and two market benchmarks; 2015 and 2026 are part of the year, with clear indications.
annual_strategy_returns = pd.DataFrame({
    strategy_name: (1.0 + result['returns'].loc[:RESEARCH_END]).groupby(
        result['returns'].loc[:RESEARCH_END].index.year
    ).prod() - 1.0
    for strategy_name, result in results.items()
})
annual_benchmark_returns = (1.0 + benchmark_returns_sgd.loc[
    strategy_start:RESEARCH_END, ['ES3.SI', 'VT']
]).groupby(benchmark_returns_sgd.loc[strategy_start:RESEARCH_END].index.year).prod() - 1.0
annual_benchmark_returns = annual_benchmark_returns.rename(columns=MARKET_BENCHMARK_NAMES)
annual_returns_all = pd.concat(
    [annual_strategy_returns, annual_benchmark_returns], axis=1
).sort_index()
annual_returns_all.index.name = 'Year'

annual_returns_display = annual_returns_all.T.copy()
annual_returns_display.columns = [
    f'{year}(part of year)' if year in {annual_returns_all.index.min(), annual_returns_all.index.max()}
    else str(year)
    for year in annual_returns_all.index
]
display(annual_returns_display.map(lambda value: '' if pd.isna(value) else f'{value:.2%}'))

# The focus strategy and the two market benchmarks are presented in the same column chart.
annual_plot_names = FOCUS_STRATEGIES + list(MARKET_BENCHMARK_NAMES.values())
annual_plot_colors = {
    **HIGHLIGHT_COLORS,
    MARKET_BENCHMARK_NAMES['ES3.SI']: MARKET_BENCHMARK_COLORS['ES3.SI'],
    MARKET_BENCHMARK_NAMES['VT']: MARKET_BENCHMARK_COLORS['VT'],
}
annual_plot_frame = annual_returns_all[annual_plot_names]
year_labels = [
    f'{year}*' if year in {annual_plot_frame.index.min(), annual_plot_frame.index.max()}
    else str(year)
    for year in annual_plot_frame.index
]

fig, ax = plt.subplots(figsize=(18, 9))
x_positions = np.arange(len(annual_plot_frame.index), dtype=float)
group_width = 0.86
bar_width = group_width / len(annual_plot_names)
for series_position, series_name in enumerate(annual_plot_names):
    is_benchmark = series_name in MARKET_BENCHMARK_NAMES.values()
    offset = (series_position - (len(annual_plot_names) - 1) / 2) * bar_width
    ax.bar(
        x_positions + offset,
        annual_plot_frame[series_name].to_numpy(dtype=float),
        width=bar_width * 0.92,
        color=annual_plot_colors[series_name],
        alpha=0.92 if is_benchmark else 0.78,
        edgecolor='#222222' if is_benchmark else 'none',
        linewidth=0.8,
        hatch='//' if is_benchmark else None,
        label=strategy_label(series_name),
    )

ax.axhline(0.0, color='#555555', linewidth=0.9)
ax.set_xticks(x_positions)
ax.set_xticklabels(year_labels, rotation=0)
ax.set_xlabel('Year (* partial year)')
ax.set_ylabel('Annual rate of return')
ax.set_title('Focus strategy and market base year returns (SDD total return; same axis)',
             fontsize=16, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
style_axis(ax, axis='y')
place_legend(ax)
fig.tight_layout(rect=[0, 0.16, 1, 1])
plt.show()


# The withdrawal of the same set of priority strategies and market benchmarks is also shown in a chart.
focus_benchmark_returns = pd.concat({
    **{name: results[name]['returns'] for name in FOCUS_STRATEGIES},
    MARKET_BENCHMARK_NAMES['ES3.SI']: benchmark_returns_sgd['ES3.SI'],
    MARKET_BENCHMARK_NAMES['VT']: benchmark_returns_sgd['VT'],
}, axis=1).loc[strategy_start:RESEARCH_END].dropna(how='any')
focus_benchmark_wealth = (1.0 + focus_benchmark_returns).cumprod()
focus_benchmark_drawdowns = focus_benchmark_wealth.div(
    focus_benchmark_wealth.cummax()
) - 1.0

fig, ax = plt.subplots(figsize=(18, 9))
for series_name in focus_benchmark_drawdowns.columns:
    is_submission = series_name == SUBMISSION_STRATEGY
    is_base = series_name == BASELINE_STRATEGY
    is_benchmark = series_name in MARKET_BENCHMARK_NAMES.values()
    series_label = strategy_label(series_name) if series_name in STRATEGY_LABELS else series_name
    if is_submission or is_base:
        fill_color = annual_plot_colors[series_name] if is_submission else '#8C9299'
        fill_alpha = 0.52 if is_submission else 0.22
        ax.fill_between(
            focus_benchmark_drawdowns.index,
            focus_benchmark_drawdowns[series_name].to_numpy(dtype=float), 0.0,
            color=fill_color, linewidth=0, edgecolor='none', alpha=fill_alpha,
            label=series_label, zorder=4 if is_submission else 1,
        )
    else:
        ax.plot(
            focus_benchmark_drawdowns.index,
            focus_benchmark_drawdowns[series_name],
            color=annual_plot_colors[series_name], linewidth=1.7 if is_benchmark else 1.3,
            alpha=0.80 if is_benchmark else 0.62,
            label=series_label, zorder=3 if is_benchmark else 2,
        )

ax.axvline(IN_SAMPLE_END, color='#AAB2BD', linestyle='-', linewidth=1.0,
           label='The end point in the sample')
ax.axhline(0.0, color='#555555', linewidth=0.8)
ax.set_title('Focus policy and market benchmark retreat curve (SGD total return; same axis)',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Fall back.')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0.16, 1, 1])
plt.show()

if annual_returns_all.index.max() != RESEARCH_END.year:
    raise RuntimeError('Annual returns do not cover the final year of the study.')

if benchmark_plot_returns.index.max() >= HOLDOUT_START:
    raise RuntimeError('The market benchmark study map read the lock-in test range.')
if not {'ES3.SI', 'VT'}.issubset(benchmark_returns_sgd.columns):
    raise RuntimeError('Market baseline returns data are incomplete.')
print('Market benchmarking is complete: Alpha/Beta has deducted SORA, and all research period indicators and graphics stop before lock-in test set.')

# 6.4 Consolidated summary of all 12 formal strategies
#
#
# This section is the only formal consolidated portal. The table of indicators, the complete study of the cumulative inter-area income map, the retraction map and the cumulative inter-validation income map are all read out and therefore contain the same 12 strategies. The previous drawing section defines only functions and does not produce another set of formal results.
# --- Original Notebook code module 75-
FINAL_EXPECTED_STRATEGY_COUNT = 13
final_strategy_names = list(results.keys())
if len(final_strategy_names) != FINAL_EXPECTED_STRATEGY_COUNT:
    raise RuntimeError(
        f'The number of formal strategies should read{FINAL_EXPECTED_STRATEGY_COUNT}, actual{len(final_strategy_names)}。'
    )
if len(set(final_strategy_names)) != len(final_strategy_names):
    raise RuntimeError('There is a duplication of official policy names.')

sora_audit_records = []
for scope_name, result_map in [('Area of study', results), ('Lock Test Range', holdout_results)]:
    for strategy_name, result in result_map.items():
        if 'risk_free_daily' not in result or 'funding_daily' not in result:
            raise RuntimeError(f'{scope_name}Yes.{strategy_name}No single SORA is used.')
        strategy_index = result['returns'].dropna().index
        rf_series = result['risk_free_daily'].reindex(strategy_index)
        funding_series = result['funding_daily'].reindex(strategy_index)
        if rf_series.isna().any() or funding_series.isna().any():
            raise RuntimeError(f'{scope_name}Yes.{strategy_name}There is a SORA coverage gap.')
        sora_audit_records.append({
            'Scope': scope_name, 'Policy': strategy_name,
            'Number of days of trading in proceeds': len(strategy_index), 'SORA covers transaction days': int(rf_series.notna().sum()),
            'Finance interest rate covers transaction days': int(funding_series.notna().sum()),
        })
sora_usage_audit = pd.DataFrame(sora_audit_records).set_index(['Scope', 'Policy'])
print(
    f'SORA uniform inspection passes: study space{len(results)}/{len(results)}Item,'
    f'Lock Test Range{len(holdout_results)}/{len(holdout_results)}The policies are fully covered.'
)

strategy_returns = pd.concat(
    [results[name]['returns'].rename(name) for name in final_strategy_names], axis=1
).dropna(how='any')
if strategy_returns.index.max() >= HOLDOUT_START:
    raise RuntimeError('A uniform summary of data leaks from locked tests.')

metrics_by_period = pd.concat({
    period: pd.DataFrame({
        name: compute_metrics(results[name], start, end)
        for name in final_strategy_names
    }).T
    for period, (start, end) in periods.items()
}, names=['Period', 'Policy'])

# Each test period is ranked independently from small to large according to the Sharp ratio.
sorted_period_frames = []
for period_name in metrics_by_period.index.get_level_values('Period').unique():
    period_frame = metrics_by_period.xs(period_name, level='Period').sort_values(
        'Sharpe Ratio', ascending=True, kind='mergesort'
    )
    period_frame.insert(
        0, 'Shape Sorting Number (low to high)', np.arange(1, len(period_frame) + 1)
    )
    period_frame['Period'] = period_name
    sorted_period_frames.append(period_frame.reset_index())

metrics_by_period_sorted = pd.concat(sorted_period_frames, ignore_index=True)
metrics_by_period_sorted = metrics_by_period_sorted.set_index(['Period', 'Shape Sorting Number (low to high)', 'Policy'])

# Normal DataFrame is displayed without background gradients, heat attempts or colour encoding.
formatted_metrics = metrics_by_period_sorted.copy()
for column in ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown',
               'Positive-Day Ratio', 'Annual One-Way Turnover', 'Annual Two-Way Traded Notional', 'Average Gross Exposure']:
    formatted_metrics[column] = formatted_metrics[column].map(lambda value: f'{value:.2%}')
for column in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
    formatted_metrics[column] = formatted_metrics[column].map(lambda value: f'{value:.2f}')
for column in ['Number of Rebalances', 'Number of Trade Legs']:
    formatted_metrics[column] = formatted_metrics[column].map(lambda value: f'{value:.0f}')

print(f'Number of harmonized aggregate strategies:{len(final_strategy_names)}')
print(f'Study interval:{strategy_returns.index.min().date()}to{strategy_returns.index.max().date()}')
print('Samples, certification areas and complete study areas are sorted from small to large, respectively; colour gradients are not used for the indicator tables.')
display(formatted_metrics)

final_cumulative, final_strategy_color_map = plot_all_strategy_cumulative(strategy_returns)
final_drawdowns = plot_all_strategy_drawdowns(strategy_returns, final_strategy_color_map)
final_oos_cumulative = plot_all_strategy_oos(strategy_returns, final_strategy_color_map)

assert list(final_cumulative.columns) == final_strategy_names
assert list(final_drawdowns.columns) == final_strategy_names
assert list(final_oos_cumulative.columns) == final_strategy_names
print(f'Uniform checks were carried out: a table of indicators, a map of cumulative gains, a map of fallbacks and a map of certification zones contained the same set of {len(results)} formal strategies.')

# 6.5 Strategies to defeat markets: a statistical test
#
#
# This section examines whether the strategy defeats the market at four levels: absolute gain, risk adjustment gain, back protection and statistical visibility. The market benchmark remains ** ES3.SI (Singapore equities)** and ** VT (global equities)**, using all SSD returns and aligning them with the strategy with the common trading date.
#
# - Alpha and Beta: the return of excess earnings from the use of the strategy and the relative date of SORA, and the misprocessing of differences and sequences by the Newey-West (HAC) standard;
# - Information ratio: average active daily earnings divided by tracking errors and annualization;
# - Upper- and lower-line capture rates: calculated on a monthly basis;
# - Bootstream: an estimated 95 per cent confidence interval of annualized active earnings using a sample of 20 trading days of mobile blocks;
# - Comprehensive determination: Only annualized active returns are positive, information ratios are positive, and Alpha is positive and significant will be marked as “a statistically significant defeat”; if proceeds are ahead but Alpha is not significant, they are marked as “proceeds ahead but insufficient”.
#
# These tests are ex post performance evaluations, do not participate in vouchers, risk budgets or reconciliations, and therefore do not pass benchmark future gains into strategies. Locks the test range to read only `holdout_results` that was previously independently generated.
# --- Original Notebook code unit 77 ---
import statsmodels.api as sm

OUTPERFORMANCE_BOOTSTRAP_SEED = 20260908
OUTPERFORMANCE_BOOTSTRAP_SAMPLES = 2000
OUTPERFORMANCE_BLOCK_LENGTH = 20


def maximum_drawdown(return_series):
    wealth = (1.0 + pd.Series(return_series, dtype=float).dropna()).cumprod()
    if wealth.empty:
        return np.nan
    running_peak = wealth.cummax().clip(lower=1.0)
    return float((wealth / running_peak - 1.0).min())


def moving_block_active_return_ci(
    active_returns, samples=OUTPERFORMANCE_BOOTSTRAP_SAMPLES,
    block_length=OUTPERFORMANCE_BLOCK_LENGTH, seed=OUTPERFORMANCE_BOOTSTRAP_SEED,
):
    """Move block Bootstream: Keep short-term sequences associated and return 95% of the annualized active yield."""
    values = pd.Series(active_returns, dtype=float).dropna().to_numpy()
    n = len(values)
    if n < max(60, block_length * 2):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(n / block_length))
    starts = np.arange(n)
    estimates = np.empty(samples, dtype=float)
    offsets = np.arange(block_length)
    for sample_id in range(samples):
        selected_starts = rng.choice(starts, size=block_count, replace=True)
        positions = (selected_starts[:, None] + offsets[None, :]) % n
        resampled = values[positions.ravel()[:n]]
        estimates[sample_id] = resampled.mean() * TRADING_DAYS
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def market_outperformance_test(result, benchmark_return, start=None, end=None):
    strategy = result['returns'].loc[start:end].rename('Strategic gains')
    benchmark = benchmark_return.loc[start:end].rename('Baseline proceeds')
    risk_free = result['risk_free_daily'].loc[start:end].rename('SORA')
    aligned = pd.concat([strategy, benchmark, risk_free], axis=1).dropna()
    if len(aligned) < 20:
        return pd.Series(dtype=float)

    strategy_excess = aligned['Strategic gains'] - aligned['SORA']
    benchmark_excess = aligned['Baseline proceeds'] - aligned['SORA']
    regression_x = sm.add_constant(benchmark_excess.rename('Market excess earnings'))
    max_lags = max(1, int(4 * (len(aligned) / 100) ** (2 / 9)))
    regression = sm.OLS(strategy_excess, regression_x).fit(
        cov_type='HAC', cov_kwds={'maxlags': max_lags}
    )
    daily_alpha = float(regression.params['const'])
    annual_alpha = daily_alpha * TRADING_DAYS
    alpha_t = float(regression.tvalues['const'])
    alpha_p = float(regression.pvalues['const'])
    beta = float(regression.params['Market excess earnings'])
    alpha_ci_daily = regression.conf_int(alpha=0.05).loc['const']

    active = aligned['Strategic gains'] - aligned['Baseline proceeds']
    tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    information_ratio = (
        float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if active.std(ddof=1) > 0 else np.nan
    )
    bootstrap_low, bootstrap_high = moving_block_active_return_ci(active)

    monthly = (1.0 + aligned[['Strategic gains', 'Baseline proceeds']]).resample('ME').prod() - 1.0
    up = monthly['Baseline proceeds'] > 0
    down = monthly['Baseline proceeds'] < 0
    upside_capture = (
        float(monthly.loc[up, 'Strategic gains'].mean() / monthly.loc[up, 'Baseline proceeds'].mean())
        if up.any() else np.nan
    )
    downside_capture = (
        float(monthly.loc[down, 'Strategic gains'].mean() / monthly.loc[down, 'Baseline proceeds'].mean())
        if down.any() else np.nan
    )

    strategy_total = float((1.0 + aligned['Strategic gains']).prod() - 1.0)
    benchmark_total = float((1.0 + aligned['Baseline proceeds']).prod() - 1.0)
    strategy_annual = annualized_compound_return(aligned['Strategic gains'])
    benchmark_annual = annualized_compound_return(aligned['Baseline proceeds'])
    strategy_sharpe = float(
        strategy_excess.mean() / strategy_excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    benchmark_sharpe = float(
        benchmark_excess.mean() / benchmark_excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    strategy_mdd = maximum_drawdown(aligned['Strategic gains'])
    benchmark_mdd = maximum_drawdown(aligned['Baseline proceeds'])

    return pd.Series({
        'Common trading day': len(aligned),
        'Strategy cumulative rate of return': strategy_total,
        'Baseline cumulative rate of return': benchmark_total,
        'Strategy annualized rate of return': strategy_annual,
        'Base yearized rate of return': benchmark_annual,
        'Annualized positive returns': strategy_annual - benchmark_annual,
        'Strategy Sharp Ratio': strategy_sharpe,
        'Base Sharp ratio': benchmark_sharpe,
        'Maximum retreat of strategy': strategy_mdd,
        'Maximum benchmark retreat': benchmark_mdd,
        'Alpha': annual_alpha,
        'Alpha age 95% lower limit': float(alpha_ci_daily.iloc[0] * TRADING_DAYS),
        'Alpha \' s 95% upper limit': float(alpha_ci_daily.iloc[1] * TRADING_DAYS),
        'Alpha t (HAC)': alpha_t,
        'Alpha p (HAC)': alpha_p,
        'Beta': beta,
        'Information ratio': information_ratio,
        'Yearly tracking error': tracking_error,
        'Bootstream 95% floor': bootstrap_low,
        'Bootstrap 95 per cent cap': bootstrap_high,
        'Increase catch rate': upside_capture,
        'Lower catch rate': downside_capture,
    })


def outperformance_verdict(row):
    return_lead = row['Annualized positive returns'] > 0
    risk_adjusted_lead = row['Information ratio'] > 0
    significant_alpha = row['Alpha'] > 0 and row['Alpha p (HAC)'] < 0.05
    bootstrap_positive = row['Bootstream 95% floor'] > 0
    if return_lead and risk_adjusted_lead and significant_alpha and bootstrap_positive:
        return 'A statistically significant defeat of the market.'
    if return_lead and risk_adjusted_lead:
        return 'Gains ahead, but lack of statistical evidence'
    return 'Not beat the market.'


# Full formal strategy: study of the whole zone versus two market benchmark tests.
outperformance_records = []
for strategy_name, result in results.items():
    for benchmark_ticker, benchmark_label in MARKET_BENCHMARK_NAMES.items():
        metric = market_outperformance_test(
            result, benchmark_returns_sgd[benchmark_ticker], strategy_start, RESEARCH_END
        )
        record = metric.to_dict()
        record.update({'Policy': strategy_name, 'Market benchmarks': benchmark_label})
        outperformance_records.append(record)

market_outperformance_results = pd.DataFrame(outperformance_records).set_index(
    ['Market benchmarks', 'Policy']
)
market_outperformance_results['Consolidated determination'] = market_outperformance_results.apply(
    outperformance_verdict, axis=1
)


# Recommended policy: Tests within the sample, during validation, throughout research and during lock-in testing.
submission_test_records = []
submission_test_periods = {
    'In Sample': (None, IN_SAMPLE_END, results[SUBMISSION_STRATEGY]),
    'Validation interval': (OOS_START, RESEARCH_END, results[SUBMISSION_STRATEGY]),
    'Study the complete range.': (None, RESEARCH_END, results[SUBMISSION_STRATEGY]),
    'Lock Test Range': (HOLDOUT_START, actual_holdout_end, holdout_results[SUBMISSION_STRATEGY]),
}
for period_name, (start, end, result) in submission_test_periods.items():
    for benchmark_ticker, benchmark_label in MARKET_BENCHMARK_NAMES.items():
        metric = market_outperformance_test(
            result, benchmark_returns_sgd[benchmark_ticker], start, end
        )
        record = metric.to_dict()
        record.update({'Period': period_name, 'Market benchmarks': benchmark_label})
        submission_test_records.append(record)

submission_market_test = pd.DataFrame(submission_test_records).set_index(
    ['Period', 'Market benchmarks']
)
submission_market_test['Consolidated determination'] = submission_market_test.apply(
    outperformance_verdict, axis=1
)


def format_outperformance_table(frame):
    formatted = frame.copy()
    percent_columns = [
        'Strategy cumulative rate of return', 'Baseline cumulative rate of return', 'Strategy annualized rate of return', 'Base yearized rate of return',
        'Annualized positive returns', 'Maximum retreat of strategy', 'Maximum benchmark retreat', 'Alpha',
        'Alpha age 95% lower limit', 'Alpha \' s 95% upper limit', 'Yearly tracking error',
        'Bootstream 95% floor', 'Bootstrap 95 per cent cap',
        'Increase catch rate', 'Lower catch rate',
    ]
    for column in percent_columns:
        formatted[column] = formatted[column].map(
            lambda value: '—' if pd.isna(value) else f'{value:.2%}'
        )
    for column in ['Strategy Sharp Ratio', 'Base Sharp ratio', 'Alpha t (HAC)', 'Beta', 'Information ratio']:
        formatted[column] = formatted[column].map(
            lambda value: '—' if pd.isna(value) else f'{value:.2f}'
        )
    formatted['Alpha p (HAC)'] = formatted['Alpha p (HAC)'].map(
        lambda value: '—' if pd.isna(value) else f'{value:.2f}'
    )
    formatted['Common trading day'] = formatted['Common trading day'].fillna(0).astype(int)
    return formatted


display_columns = [
    'Strategy cumulative rate of return', 'Baseline cumulative rate of return', 'Annualized positive returns',
    'Strategy Sharp Ratio', 'Base Sharp ratio', 'Maximum retreat of strategy', 'Maximum benchmark retreat',
    'Alpha', 'Alpha t (HAC)', 'Alpha p (HAC)', 'Beta', 'Information ratio',
    'Bootstream 95% floor', 'Bootstrap 95 per cent cap',
    'Increase catch rate', 'Lower catch rate', 'Consolidated determination',
]
print('The proposed strategy, interdivisional market victory test:')
display(format_outperformance_table(submission_market_test)[display_columns])

print('The market victory test for all formal strategy studies (high to low information ratios):')
display(
    format_outperformance_table(
        market_outperformance_results.sort_values('Information ratio', ascending=False)
    )[display_columns]
)


# The rolling test only uses each date and 252 prior common trading days, without affecting strategic transaction decisions.
rolling_source = pd.concat({
    SUBMISSION_STRATEGY: results[SUBMISSION_STRATEGY]['returns'],
    MARKET_BENCHMARK_NAMES['ES3.SI']: benchmark_returns_sgd['ES3.SI'],
    MARKET_BENCHMARK_NAMES['VT']: benchmark_returns_sgd['VT'],
}, axis=1).loc[strategy_start:RESEARCH_END].dropna()

rolling_active_return = pd.DataFrame(index=rolling_source.index)
rolling_information_ratio = pd.DataFrame(index=rolling_source.index)
for benchmark_label in [MARKET_BENCHMARK_NAMES['ES3.SI'], MARKET_BENCHMARK_NAMES['VT']]:
    active = rolling_source[SUBMISSION_STRATEGY] - rolling_source[benchmark_label]
    rolling_active_return[benchmark_label] = active.rolling(TRADING_DAYS).mean() * TRADING_DAYS
    rolling_information_ratio[benchmark_label] = (
        active.rolling(TRADING_DAYS).mean()
        / active.rolling(TRADING_DAYS).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

rolling_colors = {
    MARKET_BENCHMARK_NAMES['ES3.SI']: MARKET_BENCHMARK_COLORS['ES3.SI'],
    MARKET_BENCHMARK_NAMES['VT']: MARKET_BENCHMARK_COLORS['VT'],
}
rolling_styles = {
    MARKET_BENCHMARK_NAMES['ES3.SI']: '-',
    MARKET_BENCHMARK_NAMES['VT']: '-',
}
fig, ax = plt.subplots(figsize=(14, 6))
for benchmark_label in rolling_active_return.columns:
    ax.plot(
        rolling_active_return.index, rolling_active_return[benchmark_label],
        color=rolling_colors[benchmark_label], linestyle=rolling_styles[benchmark_label],
        linewidth=2.0, label=f'Relative{benchmark_label}',
    )
ax.axhline(0, color='#777777', linewidth=1.0)
ax.axvline(IN_SAMPLE_END, color='#999999', linestyle='-', linewidth=1.2, label='The end point in the sample')
style_axis(ax)
place_legend(ax)
ax.set_title('Recommended policy 252, rolling year positive returns')
ax.set_ylabel('Active proceeds')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.set_xlabel('Date')
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
for benchmark_label in rolling_information_ratio.columns:
    ax.plot(
        rolling_information_ratio.index, rolling_information_ratio[benchmark_label],
        color=rolling_colors[benchmark_label], linestyle=rolling_styles[benchmark_label],
        linewidth=2.0, label=f'Relative{benchmark_label}',
    )
ax.axhline(0, color='#777777', linewidth=1.0)
ax.axvline(IN_SAMPLE_END, color='#999999', linestyle='-', linewidth=1.2, label='The end point in the sample')
style_axis(ax)
place_legend(ax)
ax.set_title('Recommended policy 252-day rolling information ratio')
ax.set_ylabel('Information ratio')
ax.set_xlabel('Date')
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()


# Data segregation and integrity of results.
assert results[SUBMISSION_STRATEGY]['returns'].index.max() < HOLDOUT_START
assert rolling_source.index.max() < HOLDOUT_START
assert set(market_outperformance_results.index.get_level_values('Market benchmarks')) == set(MARKET_BENCHMARK_NAMES.values())
print('Test complete: The test for the research period did not read the locked data outside the sample; the locked range was only evaluated separately from the existing holdout results.')

