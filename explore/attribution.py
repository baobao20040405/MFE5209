"""Assets, strategy modules, factors, exchange rate and trade wear and tear.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# Part VIII
#
# 8.1 Credit calibration to daily reconciliation
#
#
# This section starts with precise day-to-day attribution of the final submission strategy and ends with a single horizontal attribution of the three core strategies; no re-selection of parameters. The dynamic filter is fixed to **Top10** and the attribute range is up to the day before the outer starting point of the locked sample. All transaction signals continue to use data from the previous transaction date and earlier; attribution is an ex post interpretation tool and does not enter into transaction decision-making.
#
# First, the daily weights are re-established and the formal master returns net proceeds are broken down into gross asset gains, SORA cash gains and a single unilateral transaction cost of 0.05 per cent. Sub-market commissions, swaps, price differential slips, market shocks and withholding taxes are shown only in section 4.5 independently of the actual cost scenario and are not used in conjunction with formal attributions. The daily split must be closely reconciled with the net strategic gain.
# --- Original Notebook code unit 92--
ATTRIBUTION_STRATEGY = SUBMISSION_STRATEGY
attribution_result = results[ATTRIBUTION_STRATEGY]
ATTRIBUTION_START = strategy_start
ATTRIBUTION_END = RESEARCH_END


def build_daily_attribution_ledger(result, start, end):
    """Rebuild the weights before the transaction and generate daily contributions to the accounts; no data after the study area is read."""
    daily_return = result['returns'].loc[start:end].dropna()
    if daily_return.empty or daily_return.index.max() >= HOLDOUT_START:
        raise RuntimeError('Performance attribution is empty or beyond the point of departure of the lock test.')

    asset_columns = list(result['weights'].columns)
    if 'pre_return_weights' not in result or 'end_of_day_weights' not in result:
        raise RuntimeError('The results of the strategy lack real daily hold and cannot be accurately attributed.')
    pre_return_weights = result['pre_return_weights'].reindex(
        index=daily_return.index, columns=asset_columns, fill_value=0.0
    ).fillna(0.0)
    trade_deltas = result.get('trade_deltas', pd.DataFrame()).reindex(
        index=daily_return.index, columns=asset_columns, fill_value=0.0
    ).fillna(0.0)
    stored_cost = result['transaction_cost'].reindex(daily_return.index).fillna(0.0)
    stored_tax = result.get('withholding_tax_drag', pd.Series(0.0, index=daily_return.index)).reindex(
        daily_return.index
    ).fillna(0.0)
    stored_components = result.get('cost_components', pd.DataFrame()).reindex(
        index=daily_return.index,
        columns=['Commissions and transaction costs', 'FX Conversion Costs', 'Bid-Ask Spread and Slippage', 'Market Impact', 'Obsessed taxes'],
        fill_value=0.0,
    ).fillna(0.0)
    # The official master survey uses the unified unilateral 0.05 per cent. Full transaction costs if no real cost breakdown
    # The rest of the realistic cost components remain zero.
    missing_component_total = stored_components[
        ['Commissions and transaction costs', 'FX Conversion Costs', 'Bid-Ask Spread and Slippage', 'Market Impact']
    ].sum(axis=1).eq(0.0) & stored_cost.gt(0.0)
    stored_components.loc[missing_component_total, 'Commissions and transaction costs'] = stored_cost.loc[
        missing_component_total
    ]
    cash_return = result['risk_free_daily'].reindex(daily_return.index)
    if cash_return.isna().any():
        raise RuntimeError('The SORA-day gain during the performance attribution period was incomplete.')

    rows, weight_rows, trade_rows = [], [], []

    for date in daily_return.index:
        current_weights = pre_return_weights.loc[date].astype(float)
        delta = trade_deltas.loc[date].astype(float)
        if delta.abs().sum() > 1e-12:
            trade_rows.append(pd.Series(delta, name=date))

        day_asset_return = portfolio_returns_for_backtest.reindex(
            index=[date], columns=asset_columns
        ).iloc[0].fillna(0.0)
        cash_weight = float(result.get('cash_weight', pd.Series(dtype=float)).get(
            date, 1.0 - current_weights.sum()
        ))
        asset_contribution = current_weights * day_asset_return
        cash_contribution = cash_weight * float(cash_return.loc[date])
        gross_return = float(asset_contribution.sum() + cash_contribution)

        tax_drag = float(stored_tax.loc[date])
        after_tax_gross = gross_return - tax_drag
        cost_rate = float(stored_cost.loc[date])
        exact_cost_drag = (1.0 + after_tax_gross) * cost_rate
        raw_transaction_components = stored_components.loc[date, [
            'Commissions and transaction costs', 'FX Conversion Costs', 'Bid-Ask Spread and Slippage', 'Market Impact'
        ]]
        raw_total = float(raw_transaction_components.sum())
        if raw_total > 0:
            exact_transaction_components = exact_cost_drag * raw_transaction_components / raw_total
        else:
            exact_transaction_components = raw_transaction_components * 0.0

        row = asset_contribution.to_dict()
        row.update({
            'SORA Cash': cash_contribution,
            'Commissions and transaction costs': -float(exact_transaction_components['Commissions and transaction costs']),
            'FX Conversion Costs': -float(exact_transaction_components['FX Conversion Costs']),
            'Bid-Ask Spread and Slippage': -float(exact_transaction_components['Bid-Ask Spread and Slippage']),
            'Market Impact': -float(exact_transaction_components['Market Impact']),
            'Obsessed taxes': -tax_drag,
            'Other costed': 0.0,
        })
        rows.append(pd.Series(row, name=date))
        weight_rows.append(pd.Series({**current_weights.to_dict(), 'Cash': cash_weight}, name=date))

        reconstructed_net = after_tax_gross - exact_cost_drag
        if not np.isclose(reconstructed_net, float(daily_return.loc[date]), atol=1e-11, rtol=1e-9):
            raise RuntimeError(f'{date.date()}The attribution of proceeds was not reconciled with the net strategic gain.')
        if 1.0 + gross_return <= 0:
            raise RuntimeError(f'{date.date()}The combined gross proceeds resulted in a negative net value.')

    daily_components = pd.DataFrame(rows).fillna(0.0)
    pre_return_weights = pd.DataFrame(weight_rows).fillna(0.0)
    trades = pd.DataFrame(trade_rows).reindex(columns=asset_columns).fillna(0.0)
    reconciliation = daily_components.sum(axis=1) - daily_return
    if float(reconciliation.abs().max()) > 1e-10:
        raise RuntimeError('The day-to-day gains are attributed to errors in the reconciliation that go beyond tolerance.')
    return daily_components, pre_return_weights, trades, reconciliation


def geometrically_link_contributions(daily_components, total_returns):
    """Link the daily computational contribution to the total cumulative gain point."""
    aligned_components = daily_components.reindex(total_returns.index).fillna(0.0)
    total_returns = total_returns.reindex(aligned_components.index).astype(float)
    if not np.allclose(aligned_components.sum(axis=1), total_returns, atol=1e-10, rtol=1e-9):
        raise RuntimeError('Day-to-day contributions before geometric links cannot be reconciled with net gains.')
    wealth_after = (1.0 + total_returns).cumprod()
    wealth_before = wealth_after.shift(1, fill_value=1.0)
    terminal_wealth = float(wealth_after.iloc[-1])
    # W t-W_XQ1QXZZZ=W_X{t-1}r t, so that the cumulative income from the beginning of the period can be strictly increased by a net link.
    linked = aligned_components.mul(wealth_before, axis=0).sum(axis=0)
    if not np.isclose(float(linked.sum()), terminal_wealth - 1.0, atol=1e-9):
        raise RuntimeError('Accumulated revenue contributions cannot be reconciled with the cumulative benefit of the strategy.')
    return linked


daily_attribution, attribution_weights_daily, attribution_trades, _daily_reconciliation = build_daily_attribution_ledger(
    attribution_result, ATTRIBUTION_START, ATTRIBUTION_END
)
attribution_reconciliation = daily_attribution.sum(axis=1) - attribution_result['returns'].loc[
    ATTRIBUTION_START:ATTRIBUTION_END
]
attribution_audit = pd.DataFrame({
    'Value': [
        TOP_N, len(daily_attribution), attribution_reconciliation.abs().max(),
        attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END].index.max(),
    ]
}, index=['Number of momentum filters', 'Number of transactions attributed', 'Maximum daily reconciliation error', 'Deadline for attribution'])
display(attribution_audit)

# 8.2 Contribution of income from asset and class levels
#
#
# The contribution of assets is calculated using a daily weighting before the transaction and links the contribution after the compound gain to the end of the study. Positive is an increase in the final net value and negative is a drag on the final net value. The aggregation of categories and asset detail use the same set of lower-level contributions, so that the two tables can be reconciled with each other.
# --- Original Notebook code module 94-
linked_asset_contribution = geometrically_link_contributions(
    daily_attribution,
    attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END],
)

asset_contribution_table = pd.DataFrame({
    'Cumulative yield contribution (percentage points)': linked_asset_contribution,
    'End-of-cycle wealth contribution (SGD)': linked_asset_contribution * INITIAL_CAPITAL_SGD,
})
asset_contribution_table['Asset Name'] = [
    asset_metadata.loc[index, 'Asset Name'] if index in asset_metadata.index else index
    for index in asset_contribution_table.index
]
asset_contribution_table['Asset Class'] = [
    asset_metadata.loc[index, 'Asset Class'] if index in asset_metadata.index else (
        'Cash' if index == 'SORA Cash' else 'Trading Friction'
    )
    for index in asset_contribution_table.index
]
asset_contribution_table = asset_contribution_table.sort_values(
    'Cumulative yield contribution (percentage points)', ascending=False
)

class_contribution_table = asset_contribution_table.groupby('Asset Class', sort=False)[
    ['Cumulative yield contribution (percentage points)', 'End-of-cycle wealth contribution (SGD)']
].sum().sort_values('Cumulative yield contribution (percentage points)', ascending=False)
strategy_total_return = float(
    (1.0 + attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END]).prod() - 1.0
)
if not np.isclose(class_contribution_table['Cumulative yield contribution (percentage points)'].sum(), strategy_total_return, atol=1e-9):
    raise RuntimeError('The contribution of category income was not reconciled with the cumulative benefit of the strategy.')

display(asset_contribution_table.style.format({
    'Cumulative yield contribution (percentage points)': '{:+.2%}', 'End-of-cycle wealth contribution (SGD)': '{:+,.0f}',
}))
display(class_contribution_table.style.format({
    'Cumulative yield contribution (percentage points)': '{:+.2%}', 'End-of-cycle wealth contribution (SGD)': '{:+,.0f}',
}))

def cumulative_linked_contribution_path(daily_components, total_returns):
    """Converts the daily computational contribution into a contribution path that can be aggregated on a day-to-day basis and at the end of the period consistent with the accumulated gain."""
    aligned_returns = total_returns.reindex(daily_components.index)
    wealth_before_return = (1.0 + aligned_returns).cumprod().shift(1).fillna(1.0)
    return daily_components.mul(wealth_before_return, axis=0).cumsum()


def plot_signed_stacked_contribution(frame, title, color_map, label_map=None):
    """is stacked from a zero-axis to a negative contribution to avoid a negative value being misread by the normal stacked area map."""
    label_map = label_map or {}
    columns = frame.columns.tolist()
    colors = [color_map.get(column, '#A7ADB4') for column in columns]
    labels = [label_map.get(column, column) for column in columns]
    positive = frame.clip(lower=0.0)
    negative = frame.clip(upper=0.0)
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.stackplot(
        frame.index, *[positive[column].to_numpy(dtype=float) for column in columns],
        labels=labels, colors=colors, alpha=0.90,
    )
    ax.stackplot(
        frame.index, *[negative[column].to_numpy(dtype=float) for column in columns],
        labels=['_nolegend_'] * len(columns), colors=colors, alpha=0.90,
    )
    ax.axhline(0.0, color='#4F5963', linewidth=0.9)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative proceeds contribution')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
    style_axis(ax, axis='y')
    place_legend(ax, item_count=len(columns))
    fig.tight_layout(rect=[0, 0.16 if len(columns) > 4 else 0, 1, 1])
    plt.show()


attribution_period_returns = attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END]
asset_contribution_path = cumulative_linked_contribution_path(
    daily_attribution, attribution_period_returns
)
if not np.allclose(
    asset_contribution_path.iloc[-1].reindex(linked_asset_contribution.index),
    linked_asset_contribution, atol=1e-9,
):
    raise RuntimeError('The end value of the asset \' s cumulative contribution path was not reconciled with the attribution statement.')

contribution_class_lookup = {
    column: (asset_metadata.loc[column, 'Asset Class'] if column in asset_metadata.index
             else ('Cash' if column == 'SORA Cash' else 'Trading Friction'))
    for column in asset_contribution_path.columns
}
class_contribution_path = pd.DataFrame(index=asset_contribution_path.index)
for asset_class in dict.fromkeys(contribution_class_lookup.values()):
    members = [column for column, mapped_class in contribution_class_lookup.items()
               if mapped_class == asset_class]
    class_contribution_path[asset_class] = asset_contribution_path[members].sum(axis=1)

plot_signed_stacked_contribution(
    class_contribution_path,
    f'{strategy_label(ATTRIBUTION_STRATEGY)}: Accumulated sources of income contribution (by broad class of assets)',
    ASSET_CLASS_COLORS,
)

asset_palette = {
    column: plt.matplotlib.colors.to_hex(plt.cm.tab20(index / max(1, len(asset_contribution_path.columns) - 1)))
    for index, column in enumerate(asset_contribution_path.columns)
}
asset_label_map = {
    column: (asset_metadata.loc[column, 'Asset Name'] if column in asset_metadata.index else column)
    for column in asset_contribution_path.columns
}
plot_signed_stacked_contribution(
    asset_contribution_path,
    f'{strategy_label(ATTRIBUTION_STRATEGY)}: Accumulated sources of contribution to proceeds (target assets)',
    asset_palette,
    asset_label_map,
)

# 8.3 Strategy module incremental contribution
#
#
# This table compares the marginal changes in the module according to the fixed construction order: the equal rights baseline screens trends agilely balanced intelligence and sound mind agitation. The incrementality is the cumulative yield difference between counter-fact strategies that allows for an initial reconciliation but is affected by the order in which the modules are added, and therefore is not interpreted as the sole causal contribution.
# --- Original Notebook code unit 96-
ATTRIBUTION_MOMENTUM_EQUAL_NAME = 'Dedicated: 12-1 month kinetic Top10, etc. Rights'


def target_momentum_12_1_equal(history):
    selected, score = select_top10(history)
    target = pd.Series(0.0, index=ALL_ASSETS, dtype=float)
    target.loc[selected] = 1.0 / len(selected)
    return target, selected, score


attribution_momentum_equal_result = run_dynamic_backtest(
    ATTRIBUTION_MOMENTUM_EQUAL_NAME, target_momentum_12_1_equal,
    covariance_function=None, end_date=RESEARCH_END,
)

module_sequence = [
    ('GMA-EW', results[STRATEGY_NAMES['equal_weight']]),
    ('Trends filter', attribution_momentum_equal_result),
    ('Agile balance.', results[STRATEGY_NAMES['ewma_erc']]),
    ('GMI-AHRB', results[FULLY_DYNAMIC_BUDGET_NAME]),
    ('GMI-ADG', results[DYNAMIC_ATTACK_NAME]),
]
module_rows = []
previous_total = 0.0
for order, (module_name, module_result) in enumerate(module_sequence):
    period_return = module_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END].dropna()
    total = float((1.0 + period_return).prod() - 1.0)
    increment = total if order == 0 else total - previous_total
    module_rows.append({
        'Module Order': order + 1, 'Module': module_name,
        'Cumulative gains from this phase of the strategy': total, 'A step-by-step incremental contribution': increment,
    })
    previous_total = total

module_attribution_table = pd.DataFrame(module_rows).set_index('Module Order')
if not np.isclose(
    module_attribution_table['A step-by-step incremental contribution'].sum(),
    module_attribution_table['Cumulative gains from this phase of the strategy'].iloc[-1], atol=1e-12,
):
    raise RuntimeError('The policy module increment failed to start the reconciliation.')
display(module_attribution_table.style.format({
    'Cumulative gains from this phase of the strategy': '{:.2%}', 'A step-by-step incremental contribution': '{:+.2%}',
}))

fig, ax = plt.subplots(figsize=(12, 5.5))
module_values = module_attribution_table['A step-by-step incremental contribution']
ax.bar(
    module_attribution_table['Module'], module_values,
    color=np.where(module_values >= 0, '#2F7D59', '#C94C40')
)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Psychic attack: incremental contribution of the strategy module over layers')
ax.set_ylabel('Cumulative gain increase (percentage points)')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.tick_params(axis='x', rotation=20)
plt.show()

# 8.4 Contribution of child gain and residual Alpha
#
#
# The current four-factor framework is followed: global equities, US medium-term national debt, commodity baskets and US dollars. The return was based on SORA as a risk-free gain, using the HAC standard wrongly. In addition to Beta, this section further disassemblys the daily strategic gains into SORA, four factors, regression distance Alpha and unexplained disability, and contributes to the cumulative gains by linking them to the same geometry as the asset attribution.
#
# This is an ex post interpretation and does not participate in weight calculations; the “remnant Alpha” indicates that the average benefit for which the existing factor cannot be explained is not automatically equivalent to a sustainable projection capability.
# --- Original Notebook code unit 98--
def factor_performance_attribution(result, start, end):
    strategy = result['returns'].loc[start:end].rename('Policy')
    rf = result['risk_free_daily'].loc[start:end].rename('SORA')
    factors = pd.DataFrame({
        'Global equities': benchmark_returns_sgd['VT'].loc[start:end],
        'Medium-term national debt': returns['IEF'].loc[start:end],
        'Commodity basket': returns[['USO', 'DBB', 'GLD']].loc[start:end].mean(axis=1),
        'United States dollars': returns['UUP'].loc[start:end],
    })
    aligned = pd.concat([strategy, rf, factors], axis=1).dropna()
    y = aligned['Policy'] - aligned['SORA']
    factor_excess = aligned[factors.columns].sub(aligned['SORA'], axis=0)
    design = sm.add_constant(factor_excess)
    max_lags = max(1, int(4 * (len(aligned) / 100) ** (2 / 9)))
    model = sm.OLS(y, design).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})

    daily = pd.DataFrame(index=aligned.index)
    daily['SORA'] = aligned['SORA']
    for factor_name in factors.columns:
        daily[factor_name] = float(model.params[factor_name]) * factor_excess[factor_name]
    daily['Alpha Residual (Intersection)'] = float(model.params['const'])
    daily['No explanation for disability'] = aligned['Policy'] - daily.sum(axis=1)
    if not np.allclose(daily.sum(axis=1), aligned['Policy'], atol=1e-11, rtol=1e-9):
        raise RuntimeError('It is not possible to reconcile with the strategic gain due to the daily break-up of the children.')

    linked = geometrically_link_contributions(daily, aligned['Policy'])
    annual_arithmetic = daily.mean() * TRADING_DAYS
    summary = pd.DataFrame({
        'Beta/factor': [
            np.nan if name in ['SORA', 'No explanation for disability'] else (
                float(model.params['const']) if name == 'Alpha Residual (Intersection)'
                else float(model.params[name])
            ) for name in daily.columns
        ],
        'Contribution of annual calculus benefits': annual_arithmetic,
        'Cumulative yield contribution (percentage points)': linked,
    }, index=daily.columns)
    summary.loc['Alpha Residual (Intersection)', 'HAC t'] = float(model.tvalues['const'])
    summary.loc['Alpha Residual (Intersection)', 'HAC p'] = float(model.pvalues['const'])
    return summary, daily, model


factor_attribution_table, factor_daily_contribution, factor_model = factor_performance_attribution(
    attribution_result, ATTRIBUTION_START, ATTRIBUTION_END
)
display(factor_attribution_table.style.format({
    'Beta/factor': '{:.2f}', 'Contribution of annual calculus benefits': '{:+.2%}',
    'Cumulative yield contribution (percentage points)': '{:+.2%}', 'HAC t': '{:.2f}', 'HAC p': '{:.2f}',
}))
print(f'Four factors back to R squared:{factor_model.rsquared:.2f}')

fig, ax = plt.subplots(figsize=(11, 5.5))
factor_plot = factor_attribution_table.sort_values('Cumulative yield contribution (percentage points)')
ax.barh(
    factor_plot.index, factor_plot['Cumulative yield contribution (percentage points)'],
    color=np.where(factor_plot['Cumulative yield contribution (percentage points)'] >= 0, '#5B6FA8', '#C94C40')
)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title(f'{ATTRIBUTION_STRATEGY}: factor cumulative gain contribution and residual Alpha')
ax.set_xlabel('Cumulative yield contribution (percentage points)')
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
plt.show()

# 8.5 Trade-to-net gains wear waterfalls Figure
#
#
# This section uses a stand-alone copy of the actual cost of the final strategy to link gross asset gains and SORA gains to real net gains.
# Actual costs include sub-market commissions, exchange of currency, market differentials and slip points, transactions based on a 20-day delay and
# Market shocks at volatility rates, as well as annualized withholding taxes, drag agents. It does not cover the harmonization of 0.05 per cent of official strategic results;
# All parameters are transparent scenario assumptions and do not create a record of transactions by book.
# --- Original Notebook code unit 100--
realistic_attribution_result = realistic_submission_result
realistic_daily_attribution, _, _, realistic_reconciliation_error = build_daily_attribution_ledger(
    realistic_attribution_result, ATTRIBUTION_START, ATTRIBUTION_END
)
asset_columns_for_attribution = [
    column for column in realistic_daily_attribution.columns
    if column in realistic_attribution_result['weights'].columns
]
wear_daily = pd.DataFrame(index=realistic_daily_attribution.index)
wear_daily['Gross proceeds from assets'] = realistic_daily_attribution[asset_columns_for_attribution].sum(axis=1)
wear_daily['SORA cash proceeds'] = realistic_daily_attribution['SORA Cash']
wear_daily['Commissions and transaction costs'] = realistic_daily_attribution['Commissions and transaction costs']
wear_daily['FX Conversion Costs'] = realistic_daily_attribution['FX Conversion Costs']
wear_daily['Bid-Ask Spread and Slippage'] = realistic_daily_attribution['Bid-Ask Spread and Slippage']
wear_daily['Market Impact'] = realistic_daily_attribution['Market Impact']
wear_daily['Obsessed taxes'] = realistic_daily_attribution['Obsessed taxes']
wear_daily['Other costed'] = realistic_daily_attribution['Other costed']

wear_linked = geometrically_link_contributions(
    wear_daily,
    realistic_attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END],
)
wear_attribution_table = pd.DataFrame({
    'Cumulative yield contribution (percentage points)': wear_linked,
    'End-of-cycle wealth contribution (SGD)': wear_linked * INITIAL_CAPITAL_SGD,
})
wear_attribution_table.loc['Final net gain', 'Cumulative yield contribution (percentage points)'] = wear_linked.sum()
wear_attribution_table.loc['Final net gain', 'End-of-cycle wealth contribution (SGD)'] = (
    wear_linked.sum() * INITIAL_CAPITAL_SGD
)
display(wear_attribution_table.style.format({
    'Cumulative yield contribution (percentage points)': '{:+.2%}', 'End-of-cycle wealth contribution (SGD)': '{:+,.0f}',
}))

friction_coverage = pd.DataFrame({
    'Treatment': [
        'A copy of the actual cost is accounted for by market.', 'A copy of the actual cost is recorded in the currency of the transaction',
        'A copy of the actual cost is recorded on the market', 'Actual cost copies are entered using square root models',
        'A copy of the actual cost is presented on a daily basis by the annual agent of the asset', 'SORA cash gains accounted for; no borrowing for current non-leveraging strategy',
    ],
    'Model description and limitations': [
        'Formal comparisons remain uniform at 0.05 per cent; copies of stand-alone realistic costs are used in the Falls map',
        'The reality scenario uses a split-rate manual rate that does not pretend to be a real exchange of history.',
        'The reality scenario uses a market-based fixed agent, Yahoo has no historical order book',
        'Use of square root impact models for delayed ADV and volatility in realistic scenarios',
        'Use of asset-grade fixed annual tax negatives in realistic scenarios',
        'Current strategy is unlevered and cashless, so no borrowing takes place; future leverage needs to be added to SORA floating financing Bad',
    ],
    'This falls caliber.': ['Sub-market rates', 'Earmark rate', 'Submarket agent', 'Delay ADV square root model', 'Asset-grade agent', 'Not applicable'],
}, index=['Commissions/transactions agents', 'FX Conversion Costs', 'Real trade price difference and slide point', 'Market Impact', 'Taxes and withholding taxes', 'Financing Cost'])
display(friction_coverage)

waterfall_items = ['Gross proceeds from assets', 'SORA cash proceeds', 'Commissions and transaction costs', 'FX Conversion Costs', 'Bid-Ask Spread and Slippage', 'Market Impact', 'Obsessed taxes', 'Other costed']
waterfall_labels = [
    'Gross proceeds from assets', 'SORA cash proceeds', 'Commissions and transaction costs', 'FX Conversion Costs',
    'Bid-Ask Spread and Slippage', 'Market Impact', 'Obsessed taxes', 'Other costed',
]
increments = wear_linked.reindex(waterfall_items).fillna(0.0)
starts = increments.cumsum().shift(1, fill_value=0.0)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    waterfall_labels, increments, bottom=starts,
    color=np.where(increments >= 0, '#2F7D59', '#C94C40'), edgecolor='white'
)
ax.bar(
    ['Final net gain'], [increments.sum()],
    color='#264653', edgecolor='white'
)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title(f'{ATTRIBUTION_STRATEGY}: Gross proceeds from real cost scenario to net gain falls Figure')
ax.set_ylabel('Cumulative yield contribution (percentage points)')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.tick_params(axis='x', rotation=18)
plt.show()

assert TOP_N == 10, 'The official momentum filter must be Top10.'
assert daily_attribution.index.max() < HOLDOUT_START, 'Performance attribution does not allow access to locked test data.'
assert np.isclose(
    wear_attribution_table.loc['Final net gain', 'Cumulative yield contribution (percentage points)'],
    float((1.0 + realistic_attribution_result['returns'].loc[
        ATTRIBUTION_START:ATTRIBUTION_END
    ]).prod() - 1.0), atol=1e-9,
), 'Real cost wear waterfalls were not reconciled with real net gains.'
assert realistic_reconciliation_error.abs().max() <= 1e-10
print('Performance attribution is completed: assets, categories, modules, factors and transaction wear and tear have been exported; daily reconciliation with accumulated gains is performed.')

# 8.6 Extension attribution: Bitcoin, gold, country, momentum and dynamic budget
#
#
# The base 4 factor model is used to provide a stable, interpretable reference; this section further disassembly the proceeds previously classified as residual Alpha. Extension models added:
#
# 1. **bitcoin, gold, energy, industrial metal factors**: avoid mixing different commodity risks into a basket;
# ** Relative national and regional factors**: the United States, Europe, Japan, China/Hong Kong and Singapore use “regional equity gains - global equity gains” to reduce duplication with global stocks Beta;
# 3. **Multi-asset kinetic style factor**: use of 12-1 months of earnings to collect as of signal date, close to execute next transaction date, 1/3 more, 1/3 less;
# 4. ** Non-linear stock selection factor**: test of Treynor-Mazuy junction selection using global equity excess yield squares minus historical expansion averages;
# 5. ** Dynamic risk budget incremental factor**: the incremental interaction of internal budget rules is measured using a fully dynamic two-tiered risk budget strategy less the benefit of the static two-tiered risk budget strategy over the same period.
#
# Regional, dynamic and dynamic budget factors are explanatory agents. The extended regression may be co-linear, so that both the correlation and condition numbers are exported; the factor contribution is a statistical interpretation and is not equivalent to a defined causal link.
# --- Original Notebook code unit 102-
def build_delayed_multi_asset_momentum_factor(return_frame, start, end):
    """Constructs a multi-assets motor multiple factor before cost without future information."""
    frame = return_frame.loc[:end, ALL_ASSETS].copy()
    dates = frame.loc[start:end].index
    long_assets, short_assets = [], []
    pending_signal = None
    records = []
    start_position = frame.index.get_loc(dates[0])
    group_size = max(1, len(ALL_ASSETS) // 3)

    for date in dates:
        position = frame.index.get_loc(date)

        day_return = frame.loc[date]
        factor_return = 0.0
        if long_assets and short_assets:
            factor_return = float(
                day_return[long_assets].mean() - day_return[short_assets].mean()
            )
        records.append((date, factor_return))

        # Orders formed on the closing date of the previous transaction and executed on the closing date,
        # The new multi-empty combination begins to yield benefits between the next closing and closing slots.
        if pending_signal is not None:
            if not pending_signal['Signal Date'] < date:
                raise RuntimeError('The kinetic style factor is active on the signal day.')
            long_assets = pending_signal['Multiple']
            short_assets = pending_signal['Empty']
            pending_signal = None

        if (position - start_position) % CHECK_INTERVAL == 0 and pending_signal is None:
            history = frame.iloc[position - LOOKBACK + 1:position + 1]
            if len(history) != LOOKBACK or history.index.max() != date:
                raise RuntimeError('The momentum style factor historical window does not strictly close to the signal day closing or insufficient length.')
            score = momentum_12_1(history)
            pending_signal = {
                'Signal Date': date,
                'Multiple': score.nlargest(group_size).index.tolist(),
                'Empty': score.nsmallest(group_size).index.tolist(),
            }

    factor = pd.Series(dict(records), name='Multi-asset momentum', dtype=float)
    if factor.index.max() >= HOLDOUT_START:
        raise RuntimeError('The kinetic style factor reads the locking out data.')
    return factor


def build_expanded_attribution_factors(result, start, end):
    strategy_index = result['returns'].loc[start:end].dropna().index
    rf = result['risk_free_daily'].reindex(strategy_index)
    if rf.isna().any():
        raise RuntimeError('Extension factor attribution SORA coverage is incomplete.')

    world = benchmark_returns_sgd['VT'].reindex(strategy_index)
    us_equity = returns[['SPY', 'QQQ', 'IWM']].reindex(strategy_index).mean(axis=1)
    europe_equity = returns[['EXS1.DE', 'ISF.L']].reindex(strategy_index).mean(axis=1)
    japan_equity = returns['1308.T'].reindex(strategy_index)
    greater_china = returns[['510300.SS', '2800.HK']].reindex(strategy_index).mean(axis=1)
    singapore_equity = returns['ES3.SI'].reindex(strategy_index)

    factor_frame = pd.DataFrame(index=strategy_index)
    factor_frame['Global equities'] = world - rf
    factor_frame['United States relative global'] = us_equity - world
    factor_frame['Europe relative global'] = europe_equity - world
    factor_frame['Japan relative global'] = japan_equity - world
    factor_frame['Hong Kong, China, relative global'] = greater_china - world
    factor_frame['Singapore relative global'] = singapore_equity - world
    factor_frame['US Medium Term National Debt'] = returns['IEF'].reindex(strategy_index) - rf
    factor_frame['Gold'] = returns['GLD'].reindex(strategy_index) - rf
    factor_frame['Energy'] = returns['USO'].reindex(strategy_index) - rf
    factor_frame['Industrial metals'] = returns['DBB'].reindex(strategy_index) - rf
    factor_frame['United States dollars'] = returns['UUP'].reindex(strategy_index) - rf
    factor_frame['Bitcoin.'] = returns['BTC-USD'].reindex(strategy_index) - rf
    factor_frame['Multi-asset momentum'] = build_delayed_multi_asset_momentum_factor(
        returns, start, end
    ).reindex(strategy_index)

    # Only the averages of current market returns and expansions up to the previous day are used and future averages are not read.
    market_square = factor_frame['Global equities'].pow(2)
    historical_square_mean = market_square.expanding(min_periods=2).mean().shift(1)
    factor_frame['Non-linear stock selection'] = (
        market_square - historical_square_mean
    ).fillna(0.0)

    dynamic_return = results[SUBMISSION_STRATEGY]['returns'].reindex(strategy_index)
    static_return = results[HIERARCHICAL_BUDGET_NAME]['returns'].reindex(strategy_index)
    factor_frame['Dynamic risk budget increment'] = dynamic_return - static_return

    if factor_frame.isna().any().any():
        missing = factor_frame.columns[factor_frame.isna().any()].tolist()
        raise RuntimeError(f'Extension factor is missing:{missing}')
    return factor_frame, rf


expanded_factor_returns, expanded_factor_rf = build_expanded_attribution_factors(
    attribution_result, ATTRIBUTION_START, ATTRIBUTION_END
)

expanded_factor_definition = pd.DataFrame({
    'Type': [
        'Market', 'Country/area', 'Country/area', 'Country/area', 'Country/area', 'Country/area',
        'Asset class', 'Asset class', 'Asset class', 'Asset class', 'Asset class', 'Asset class',
        'Style', 'Non-linear', 'Internal policy interaction',
    ],
    'Definition Summary': [
        'VT－SORA', 'United States stock basket-VT', 'European stock basket - VT', 'TOPIX－VT',
        'China/Hong Kong stock basket-VT', 'ES3.SI－VT', 'IEF－SORA', 'GLD－SORA',
        'USO－SORA', 'DBB－SORA', 'UUP－SORA', 'BTC-USD－SORA',
        '12-1 months of winners and losers behind schedule', 'Global Equities Excessed Earnings Zero minus Historical Growth Average',
        'Full dynamic double budget gains -- static double budget gains',
    ],
}, index=expanded_factor_returns.columns)
display(expanded_factor_definition)

# 8.6.1 Extension factor contribution, residual Alpha and identification diagnosis
#
#
# The return is still based on the daily excess gains of the strategy relative to SORA as a variable and using the HAC standard error. The proposed daily income is broken down into SORA, extension factors, cut-off alpha and unexplained disability, and the net opening link is used to contribute to the aggregate cumulative income. The “remnant Alpha” in the table below is the cut-off after the control of these extension factors, rather than the unconditional guarantee of future sustainable gains.
# --- Original Notebook code unit 104 ---
def expanded_factor_performance_attribution(result, factor_frame, rf, start, end):
    strategy = result['returns'].loc[start:end].rename('Policy').reindex(factor_frame.index)
    aligned = pd.concat([strategy, rf.rename('SORA'), factor_frame], axis=1).dropna()
    y = aligned['Policy'] - aligned['SORA']
    x = aligned[factor_frame.columns]
    design = sm.add_constant(x)
    max_lags = max(1, int(4 * (len(aligned) / 100) ** (2 / 9)))
    model = sm.OLS(y, design).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})

    daily = pd.DataFrame(index=aligned.index)
    daily['SORA'] = aligned['SORA']
    for factor_name in factor_frame.columns:
        daily[factor_name] = float(model.params[factor_name]) * aligned[factor_name]
    daily['Alpha Residual (Intersection)'] = float(model.params['const'])
    daily['No explanation for disability'] = aligned['Policy'] - daily.sum(axis=1)
    if not np.allclose(daily.sum(axis=1), aligned['Policy'], atol=1e-11, rtol=1e-9):
        raise RuntimeError('The daily contribution of the extended factor was not reconciled with the strategic gain.')

    linked = geometrically_link_contributions(daily, aligned['Policy'])
    annual_arithmetic = daily.mean() * TRADING_DAYS
    rows = []
    for name in daily.columns:
        if name == 'SORA' or name == 'No explanation for disability':
            coefficient = t_value = p_value = np.nan
        elif name == 'Alpha Residual (Intersection)':
            coefficient = float(model.params['const'])
            t_value = float(model.tvalues['const'])
            p_value = float(model.pvalues['const'])
        else:
            coefficient = float(model.params[name])
            t_value = float(model.tvalues[name])
            p_value = float(model.pvalues[name])
        rows.append({
            'Attribution': name, 'Beta/factor': coefficient,
            'Contribution of annual calculus benefits': annual_arithmetic[name],
            'Cumulative yield contribution (percentage points)': linked[name],
            'HAC t': t_value, 'HAC p': p_value,
        })
    summary = pd.DataFrame(rows).set_index('Attribution')

    standardized = (x - x.mean()) / x.std(ddof=1).replace(0.0, np.nan)
    standardized = standardized.dropna(axis=1, how='any')
    condition_number = float(np.linalg.cond(standardized.values))
    correlation = x.corr()
    off_diagonal = correlation.where(~np.eye(len(correlation), dtype=bool))
    maximum_correlation = float(off_diagonal.abs().max().max())
    diagnostics = pd.Series({
        'Effective trading date': len(aligned), 'Number of extended factors': len(factor_frame.columns),
        'Return R Square': float(model.rsquared), 'Adjusted R Square': float(model.rsquared_adj),
        'Number of conditions (after standardization)': condition_number,
        'Maximum factor absolute correlation factor': maximum_correlation,
        'The legacy of Alpha\'s annual calculus contribution': float(model.params['const'] * TRADING_DAYS),
        'Residual Alpha HAC t': float(model.tvalues['const']),
        'Residual Alpha HAC p': float(model.pvalues['const']),
    })
    return summary, daily, model, diagnostics, correlation


(
    expanded_factor_attribution_table,
    expanded_factor_daily_contribution,
    expanded_factor_model,
    expanded_factor_diagnostics,
    expanded_factor_correlation,
) = expanded_factor_performance_attribution(
    attribution_result, expanded_factor_returns, expanded_factor_rf,
    ATTRIBUTION_START, ATTRIBUTION_END,
)

display(expanded_factor_attribution_table.style.format({
    'Beta/factor': '{:.2f}', 'Contribution of annual calculus benefits': '{:+.2%}',
    'Cumulative yield contribution (percentage points)': '{:+.2%}',
    'HAC t': '{:.2f}', 'HAC p': '{:.2f}',
}))
display(expanded_factor_diagnostics.to_frame('Value').style.format({
    'Value': '{:.2f}',
}))

basic_alpha = float(factor_model.params['const'] * TRADING_DAYS)
expanded_alpha = float(expanded_factor_model.params['const'] * TRADING_DAYS)
model_comparison = pd.DataFrame({
    'Base 4 factor model': [
        factor_model.rsquared, basic_alpha,
        factor_model.tvalues['const'], factor_model.pvalues['const'],
    ],
    'Extension Factor Model': [
        expanded_factor_model.rsquared, expanded_alpha,
        expanded_factor_model.tvalues['const'], expanded_factor_model.pvalues['const'],
    ],
}, index=['Return R Square', 'The legacy of Alpha\'s annual calculus contribution', 'Alpha HAC t', 'Alpha HAC p'])
model_comparison['Extension model changes'] = (
    model_comparison['Extension Factor Model'] - model_comparison['Base 4 factor model']
)
display(model_comparison.style.format('{:.2f}'))

plot_table = expanded_factor_attribution_table.drop(index='No explanation for disability').sort_values(
    'Cumulative yield contribution (percentage points)'
)
fig, ax = plt.subplots(figsize=(13, 8))
colors = np.where(plot_table['Cumulative yield contribution (percentage points)'] >= 0, '#4C72B0', '#C94C40')
ax.barh(plot_table.index, plot_table['Cumulative yield contribution (percentage points)'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title(f'{ATTRIBUTION_STRATEGY}: Contribution of the cumulative benefits of the extended factor')
ax.set_xlabel('Cumulative yield contribution (percentage points)')
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
plt.tight_layout()
plt.show()

assert expanded_factor_returns.index.max() < HOLDOUT_START, 'The extended factor reads lock test data.'
assert np.isclose(
    expanded_factor_attribution_table['Cumulative yield contribution (percentage points)'].sum(),
    strategy_total_return, atol=1e-9,
), 'The cumulative contribution of the expansion factor was not reconciled with the strategy \' s net gain.'
print('Extension factor attribution is complete: cumulative contribution is reconciled with strategic net gain and off-species is locked.')

# 8.7 Accurate division of local asset gains from exchange rate gains
#
#
# The previous section explains the contribution of assets in terms of total SGD returns, but no distinction has been made between the local market performance of assets and exchange rate changes. This section uses constant equations
#
# \[
# 1+r^{SGD}_{i,t}=(1+r^{local}_{i,t})(1+r^{FX}_{i,t})
# \]
#
# The SSD proceeds for each asset are accurately broken down into “local asset gains, exchange rate gains, local income x exchange rate gains interaction items”. The inclusion of SORA cash proceeds and transaction costs is closely reconciled with the strategic net proceeds on a daily and cumulative basis. The split is accounting attribution and is not interpreted as a duplicate Alpha contribution.
# --- Original Notebook code unit 106--
def build_currency_attribution(daily_ledger, pre_return_weights, result, start, end):
    dates = result['returns'].loc[start:end].dropna().index
    asset_columns = list(result['weights'].columns)
    # The actual valuation calendar must be aligned before the proceeds are calculated. A common sample would remove dates that exceeded 5 days ' unquoted bid;
    # If the earnings are calculated first in the full workday calendar, China’s first gain after a long vacation will erroneously span only one day.
    valuation_index = portfolio_returns_for_backtest.index
    local_price_frame = local_bday.reindex(
        index=valuation_index, columns=asset_columns
    )
    local_return_frame = local_price_frame.pct_change(fill_method=None).reindex(dates)
    fx_return_frame = fx_to_sgd.reindex(valuation_index).pct_change(fill_method=None).reindex(dates)

    local_daily = pd.DataFrame(0.0, index=dates, columns=asset_columns)
    fx_daily = pd.DataFrame(0.0, index=dates, columns=asset_columns)
    interaction_daily = pd.DataFrame(0.0, index=dates, columns=asset_columns)
    reconstructed_asset_return = pd.DataFrame(0.0, index=dates, columns=asset_columns)

    for ticker in asset_columns:
        currency = asset_metadata.loc[ticker, 'Local Currency']
        local_r = local_return_frame[ticker].fillna(0.0)
        if currency == 'SGD':
            fx_r = pd.Series(0.0, index=dates)
        else:
            fx_r = fx_return_frame[currency].fillna(0.0)
        weights = pre_return_weights[ticker].reindex(dates).fillna(0.0)
        local_daily[ticker] = weights * local_r
        fx_daily[ticker] = weights * fx_r
        interaction_daily[ticker] = weights * local_r * fx_r
        reconstructed_asset_return[ticker] = local_r + fx_r + local_r * fx_r

    observed_asset_return = portfolio_returns_for_backtest.reindex(
        index=dates, columns=asset_columns
    ).fillna(0.0)
    active_mask = pre_return_weights.reindex(index=dates, columns=asset_columns).fillna(0.0).abs() > 1e-14
    pricing_error = (reconstructed_asset_return - observed_asset_return).where(active_mask, 0.0)
    maximum_pricing_error = float(pricing_error.abs().max().max())
    if maximum_pricing_error > 1e-10:
        largest = pricing_error.abs().stack().nlargest(8)
        raise RuntimeError(
            f'Local gains and exchange rate gains cannot re-construct SSD asset gains; maximum error ={maximum_pricing_error:.12g}；'
            f'Maximum error ={largest.to_dict()}'
        )

    components = pd.DataFrame(index=dates)
    components['Local asset price gains'] = local_daily.sum(axis=1)
    components['Exchange rate conversion gains'] = fx_daily.sum(axis=1)
    components['Local income x exchange rate interaction'] = interaction_daily.sum(axis=1)
    for column in ['SORA Cash', 'Commissions and transaction costs', 'FX Conversion Costs', 'Bid-Ask Spread and Slippage', 'Market Impact', 'Obsessed taxes', 'Other costed']:
        components[column] = daily_ledger[column].reindex(dates).fillna(0.0)
    strategy_return = result['returns'].reindex(dates)
    if not np.allclose(components.sum(axis=1), strategy_return, atol=1e-10, rtol=1e-9):
        raise RuntimeError('Currency attribution failed to reconcile with the daily net gains of the strategy.')
    return components, local_daily, fx_daily, interaction_daily, pricing_error


(
    currency_daily_components,
    local_asset_daily_contribution,
    fx_asset_daily_contribution,
    local_fx_interaction_daily,
    currency_pricing_error,
) = build_currency_attribution(
    daily_attribution, attribution_weights_daily, attribution_result,
    ATTRIBUTION_START, ATTRIBUTION_END,
)

currency_linked = geometrically_link_contributions(
    currency_daily_components,
    attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END],
)
currency_attribution_table = pd.DataFrame({
    'Cumulative yield contribution (percentage points)': currency_linked,
    'End-of-cycle wealth contribution (SGD)': currency_linked * INITIAL_CAPITAL_SGD,
})
currency_attribution_table.loc['Final net gain'] = [
    currency_linked.sum(), currency_linked.sum() * INITIAL_CAPITAL_SGD,
]

def link_partial_components(partial_components, total_returns):
    # Linking a group of incomplete contributions with a grouping of assets at the beginning of the grouping period; this group is not required to be alone equal to a group of proceeds.
    total_returns = total_returns.reindex(partial_components.index).astype(float)
    wealth_before = (1.0 + total_returns).cumprod().shift(1, fill_value=1.0)
    return partial_components.mul(wealth_before, axis=0).sum(axis=0)


asset_currency_detail = pd.DataFrame({
    'Local market contributions': link_partial_components(
        local_asset_daily_contribution,
        attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END],
    ),
    'Contribution of exchange rates': link_partial_components(
        fx_asset_daily_contribution,
        attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END],
    ),
    'Interactive contributions': link_partial_components(
        local_fx_interaction_daily,
        attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END],
    ),
})
asset_currency_detail['Total SGD asset contribution'] = asset_currency_detail.sum(axis=1)
asset_currency_detail['Local Currency'] = asset_metadata.loc[asset_currency_detail.index, 'Local Currency']
asset_currency_detail['Asset Name'] = asset_metadata.loc[asset_currency_detail.index, 'Asset Name']
asset_currency_detail = asset_currency_detail.sort_values('Total SGD asset contribution', ascending=False)

display(currency_attribution_table.style.format({
    'Cumulative yield contribution (percentage points)': '{:+.2%}', 'End-of-cycle wealth contribution (SGD)': '{:+,.0f}',
}))
display(asset_currency_detail.style.format({
    'Local market contributions': '{:+.2%}', 'Contribution of exchange rates': '{:+.2%}',
    'Interactive contributions': '{:+.2%}', 'Total SGD asset contribution': '{:+.2%}',
}))

plot_currency = currency_attribution_table.drop(index='Final net gain').sort_values('Cumulative yield contribution (percentage points)')
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.barh(
    plot_currency.index, plot_currency['Cumulative yield contribution (percentage points)'],
    color=np.where(plot_currency['Cumulative yield contribution (percentage points)'] >= 0, '#3B7EA1', '#C94C40'),
)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title(f'{ATTRIBUTION_STRATEGY}Local markets, exchange rates and cost contributions')
ax.set_xlabel('Cumulative yield contribution (percentage points)')
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
plt.tight_layout()
plt.show()

assert currency_daily_components.index.max() < HOLDOUT_START
assert np.isclose(currency_linked.sum(), strategy_total_return, atol=1e-9)
print('Local assets are attributed to the exchange rate: the SSD constant income equation, the daily net gain and the accumulated net gain have been reconciled.')

# 8.8 Distinction between external market factors and internal strategic rules
#
#
# “Dynamic risk budget increments” are strategic internal rules and are not tradable external market factors. In order to avoid increased mechanical alignment by adding internal variables, this section compares sequentially:
#
# 1. ** External market and style factors**: markets, regions, bonds, commodities, United States dollars, bitcoin and multi-asset momentum after one day;
# ** External factor + non-linear market status**: accession to non-linear items that are structured only by the global stock market history;
# 3. ** External factor + state + internal strategy rules**: dynamic risk budget increments are only added at the end.
#
# Therefore, the first and second models should be given priority when judging whether there are remnants of Alpha in the strategy; the third model is mainly used to explain the strategy mechanisms.
# --- Original Notebook code unit 108-
def hac_factor_model(result, factor_frame, rf, start, end):
    strategy = result['returns'].loc[start:end].rename('Policy')
    aligned = pd.concat([
        strategy, rf.rename('SORA'), factor_frame,
    ], axis=1).dropna()
    y = aligned['Policy'] - aligned['SORA']
    x = sm.add_constant(aligned[factor_frame.columns])
    max_lags = max(1, int(4 * (len(aligned) / 100) ** (2 / 9)))
    model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    return model, aligned


internal_factor = 'Dynamic risk budget increment'
timing_factor = 'Non-linear stock selection'
external_columns = [
    column for column in expanded_factor_returns.columns
    if column not in [internal_factor, timing_factor]
]
factor_model_sets = {
    'External markets and style factors': external_columns,
    'External factor + non-linear state': external_columns + [timing_factor],
    'External factor + state + internal rules': external_columns + [timing_factor, internal_factor],
}

factor_layer_models = {}
factor_layer_records = []
for model_name, columns in factor_model_sets.items():
    model, aligned = hac_factor_model(
        attribution_result, expanded_factor_returns[columns], expanded_factor_rf,
        ATTRIBUTION_START, ATTRIBUTION_END,
    )
    factor_layer_models[model_name] = model
    factor_layer_records.append({
        'Model': model_name,
        'Number of factors': len(columns),
        'R Square': float(model.rsquared),
        'Adjusted R Square': float(model.rsquared_adj),
        'Alphaization of remnants': float(model.params['const'] * TRADING_DAYS),
        'Alpha HAC t': float(model.tvalues['const']),
        'Alpha HAC p': float(model.pvalues['const']),
    })

factor_layer_comparison = pd.DataFrame(factor_layer_records).set_index('Model')
display(factor_layer_comparison.style.format({
    'R Square': '{:.2f}', 'Adjusted R Square': '{:.2f}',
    'Alphaization of remnants': '{:+.2%}', 'Alpha HAC t': '{:.2f}', 'Alpha HAC p': '{:.2f}',
}))

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(factor_layer_comparison.index, factor_layer_comparison['R Square'], color=['#7A9E9F', '#4C78A8', '#2F4B7C'])
ax.set_title('Layer by Layer Add R Square after Interpretation Variable')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=15)
style_axis(ax, axis='y')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(factor_layer_comparison.index, factor_layer_comparison['Alphaization of remnants'], color=['#7A9E9F', '#4C78A8', '#2F4B7C'])
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Alpha (Agenomics) from layer-by-layer models')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.tick_params(axis='x', rotation=15)
style_axis(ax, axis='y')
fig.tight_layout()
plt.show()

print('Principle of interpretation: External models are used to determine market and style exposure; internal rule models are used only for mechanism interpretation and not as primary evidence of Alpha \' s visibility.')

# 8.9 Two-year rolling Alpha and Beta stability
#
#
# The return of the whole sample will mean that the different stages of the market are on average combined. This section uses 504 trading day rolling windows to observe the stability of global exposures to stocks, bonds, gold, United States dollars, bitcoin and momentum on each date only using data from that date and earlier. A rolling cut-off is only a conditional description and cannot be interpreted as realizing future gains.
# --- Original Notebook code unit 110-
from statsmodels.regression.rolling import RollingOLS

ROLLING_FACTOR_COLUMNS = ['Global equities', 'US Medium Term National Debt', 'Gold', 'United States dollars', 'Bitcoin.', 'Multi-asset momentum']
rolling_strategy = attribution_result['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END].rename('Policy')
rolling_frame = pd.concat([
    rolling_strategy,
    expanded_factor_rf.rename('SORA'),
    expanded_factor_returns[ROLLING_FACTOR_COLUMNS],
], axis=1).dropna()
rolling_y = rolling_frame['Policy'] - rolling_frame['SORA']
rolling_x = sm.add_constant(rolling_frame[ROLLING_FACTOR_COLUMNS])
rolling_fit = RollingOLS(
    rolling_y, rolling_x, window=504, min_nobs=378,
).fit(params_only=True)
rolling_parameters = rolling_fit.params.dropna(how='all')
rolling_parameters['Scroll Alpha Annualization'] = rolling_parameters['const'] * TRADING_DAYS

rolling_parameter_summary = pd.DataFrame({
    'Latest value [v]': rolling_parameters.iloc[-1],
    'Medium': rolling_parameters.median(),
    'Ten percent.': rolling_parameters.quantile(0.10),
    '90% part': rolling_parameters.quantile(0.90),
})
display(rolling_parameter_summary.style.format('{:+.2f}'))

fig, ax = plt.subplots(figsize=(14, 6))
rolling_parameters[ROLLING_FACTOR_COLUMNS].plot(ax=ax, linewidth=1.4, style='-')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('504 trading day rolling factor Beta')
ax.set_ylabel('Beta')
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0.16, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(14, 5.5))
rolling_parameters['Scroll Alpha Annualization'].plot(ax=ax, color='#9C3D54', linewidth=1.8, linestyle='-')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('504 trading day rolling Alpha (conditional description only)')
ax.set_ylabel('Alpha')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
style_axis(ax)
fig.tight_layout()
plt.show()

assert rolling_parameters.index.max() < HOLDOUT_START
print('Rolling attribution complete: Only 504 transaction days backwards are used at each point of time without reading off-species locking data.')

# 8.10 Shapley at Core Investment Model III
#
#
# The elimination of a fixed order gives the module ' s interactive proceeds to all subsequent modules. This section revisits all eight combinations of the following three modules under the same asset pool, frequency of inspection on the same 15-day date, collection execution on the same transaction date and the same cost assumption:
#
# - Carbide Top10 screening;
# - EWMA strict risk parity;
# - A fully dynamic double-tier risk budget.
#
# An average marginal contribution (Shapley value) is then taken for all accession sequences. It answers “what is the average increment of the modules to the cumulative proceeds in the unified anti-facts agreement” and is not a statistical factor, Alpha, nor is it equivalent to the causal conclusion.
# This section only divides the three investment modules of the drive screening, EWMA strict risk parity and a fully dynamic double-tier risk budget. Dynamic offensives, no-trading zones and 50 per cent partial implementations of the final-submission strategy do not enter the Shapley Alliance; these implementation impacts are explained by a fixed sequence of increments and trade wear and tear.
# --- Original Notebook code unit 112-
from itertools import combinations
from math import factorial

SHAPLEY_MODULES = ['Driver Top10 Filter', 'EWMA strict risk parity', 'A fully dynamic two-tier risk budget']


def run_module_coalition(enabled_modules, start=ATTRIBUTION_START, end=ATTRIBUTION_END):
    enabled = set(enabled_modules)
    frame = returns.loc[:end, ALL_ASSETS]
    dates = frame.loc[start:end].index
    rf = align_sora_returns(dates, sora_daily_df)
    current = pd.Series(0.0, index=ALL_ASSETS, dtype=float)
    pending = None
    records = []
    first_location = frame.index.get_loc(dates[0])

    for date in dates:
        location = frame.index.get_loc(date)
        day = frame.loc[date]
        cash_weight = 1.0 - float(current.sum())
        gross = float(current @ day) + cash_weight * float(rf.loc[date])
        if 1.0 + gross <= 0:
            raise RuntimeError('Shapley\'s counter-fact net is not positive.')
        current = current * (1.0 + day) / (1.0 + gross)

        cost_rate = 0.0
        if pending is not None:
            if not pending['Signal Date'] < date:
                raise RuntimeError('Shapley\'s counter-fact signal is closed.')
            target = pending['Target weights']
            turnover = float((target - current).abs().sum())
            cost_rate = COST * turnover
            current = target.copy()
            pending = None

        net = (1.0 - cost_rate) * (1.0 + gross) - 1.0
        records.append((date, net))

        if (location - first_location) % CHECK_INTERVAL == 0:
            history = frame.iloc[location - LOOKBACK + 1:location + 1]
            if len(history) != LOOKBACK or history.index.max() != date:
                raise RuntimeError('The Shapley Anti-Fact History window does not strictly close to signal day closing or insufficient length.')
            score = momentum_12_1(history)
            if 'Driver Top10 Filter' in enabled:
                selected = score.nlargest(TOP_N).index.tolist()
            else:
                selected = ALL_ASSETS.copy()

            if 'A fully dynamic two-tier risk budget' in enabled:
                budget = fully_dynamic_budget(history, selected, score)
            else:
                budget = pd.Series(1.0 / len(selected), index=selected)

            if 'EWMA strict risk parity' in enabled:
                selected_weight = strict_risk_budget(
                    ewma_covariance(history[selected]), budget,
                )
            else:
                selected_weight = budget / budget.sum()
            target = selected_weight.reindex(ALL_ASSETS, fill_value=0.0)
            target = target / target.sum()
            pending = {'Signal Date': date, 'Target weights': target}

    series = pd.Series(dict(records), name='Strategic gains', dtype=float)
    if series.index.max() >= HOLDOUT_START:
        raise RuntimeError('Shapley read out-of-species lock data against the facts.')
    return series


coalition_returns = {}
coalition_values = {}
for size in range(len(SHAPLEY_MODULES) + 1):
    for subset in combinations(SHAPLEY_MODULES, size):
        key = frozenset(subset)
        series = run_module_coalition(key)
        coalition_returns[key] = series
        coalition_values[key] = float((1.0 + series).prod() - 1.0)

module_shapley = {}
n_modules = len(SHAPLEY_MODULES)
for module in SHAPLEY_MODULES:
    contribution = 0.0
    other_modules = [item for item in SHAPLEY_MODULES if item != module]
    for size in range(len(other_modules) + 1):
        for subset in combinations(other_modules, size):
            subset_key = frozenset(subset)
            weight = factorial(size) * factorial(n_modules - size - 1) / factorial(n_modules)
            contribution += weight * (
                coalition_values[subset_key | {module}] - coalition_values[subset_key]
            )
    module_shapley[module] = contribution

empty_key = frozenset()
full_key = frozenset(SHAPLEY_MODULES)
shapley_table = pd.DataFrame({
    'Cumulative gain increase': pd.Series(module_shapley),
})
shapley_table.loc['Cumulative gain from no module baseline', 'Cumulative gain increase'] = coalition_values[empty_key]
shapley_table.loc['Cumulative benefits of a complete combination of three modules', 'Cumulative gain increase'] = coalition_values[full_key]

coalition_table = pd.DataFrame([
    {
        'Enable module': '＋'.join([m for m in SHAPLEY_MODULES if m in key]) if key else 'No module',
        'Number of modules': len(key),
        'Cumulative Return': value,
    }
    for key, value in coalition_values.items()
]).sort_values(['Number of modules', 'Enable module']).set_index('Enable module')

display(coalition_table.style.format({'Cumulative Return': '{:.2%}'}))
display(shapley_table.style.format({'Cumulative gain increase': '{:+.2%}'}))

fig, ax = plt.subplots(figsize=(10, 5.5))
shapley_plot = pd.Series(module_shapley).sort_values()
ax.barh(shapley_plot.index, shapley_plot.values, color=np.where(shapley_plot.values >= 0, '#2F7D59', '#C94C40'))
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Cumulative increase in proceeds from the Shapley module under the unified anti-fact agreement')
ax.set_xlabel('Cumulative gain increase (percentage points)')
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
plt.tight_layout()
plt.show()

assert np.isclose(
    sum(module_shapley.values()),
    coalition_values[full_key] - coalition_values[empty_key],
    atol=1e-10,
), 'The Shapley module contribution cannot be reconciled with the full combination of the incremental base.'
print('The Shapley module is attributed to the fact that all eight counterfacts use the same execution, cost and research areas, and the contribution can be strictly aggregated.')

# Shapley's complete alliance should only respond to the core investment model and should not be confused with a submission strategy containing dynamic offensive and classification enforcement rules.
shapley_scope_audit = pd.DataFrame({
    'Cumulative Return': [
        coalition_values[full_key],
        float((1.0 + results[SUBMISSION_STRATEGY]['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END]).prod() - 1.0),
    ],
    'Include dynamic offensive and classification execution': [False, True],
}, index=['Shapley Three Module Full Alliance', 'Final Submission Policy'])
display(shapley_scope_audit.style.format({'Cumulative Return': '{:.2%}'}))
print('Synopsis: Shapley only explains the core investment model; the final enforcement layer is explained by a fixed sequence of increments and trade wear and tear.')

# 8.12 Unified horizontal attribution of the three core strategies
#
# The above details are attributed to the retention of the day-to-day audit depth of the final submission strategy; this section is then calculated in the same study compartments and functions as the three core strategies:
#
# - Contribution of cumulative proceeds to asset classes;
# - Gross proceeds of assets, SORA and losses due to transactions;
# - The return of the 4-factor Beta, the residual Alpha and its prominence.
#
# This makes it possible to make a direct judgement as to the type of asset from which the offensive pellets were increased, how much trade losses were incurred and whether the existing market, Beta, was only amplified. All attributions are up to 2026-07-31 without reading the locking test set.
# --- Original Notebook code unit 115-
core_category_records = []
core_friction_records = []
core_factor_records = []

for strategy_name in CORE_COMPARISON_STRATEGIES:
    strategy_result = results[strategy_name]
    ledger, daily_weights, trades, reconciliation = build_daily_attribution_ledger(
        strategy_result, ATTRIBUTION_START, ATTRIBUTION_END
    )
    strategy_period_returns = strategy_result['returns'].loc[
        ATTRIBUTION_START:ATTRIBUTION_END
    ].dropna()
    linked = geometrically_link_contributions(ledger, strategy_period_returns)

    for item_name, contribution in linked.items():
        if item_name in asset_metadata.index:
            contribution_class = asset_metadata.loc[item_name, 'Asset Class']
        elif item_name == 'SORA Cash':
            contribution_class = 'Cash'
        else:
            contribution_class = 'Trading Friction'
        core_category_records.append({
            'Policy': strategy_name,
            'Asset Class': contribution_class,
            'Cumulative proceeds contribution': float(contribution),
        })

    asset_columns = [column for column in ledger.columns if column in strategy_result['weights'].columns]
    wear_daily_local = pd.DataFrame(index=ledger.index)
    wear_daily_local['Gross proceeds from assets'] = ledger[asset_columns].sum(axis=1)
    wear_daily_local['SORA cash proceeds'] = ledger['SORA Cash']
    wear_daily_local['Traded wear and tear'] = ledger[
        ['Commissions and transaction costs', 'FX Conversion Costs', 'Bid-Ask Spread and Slippage', 'Market Impact', 'Obsessed taxes', 'Other costed']
    ].sum(axis=1)
    wear_linked_local = geometrically_link_contributions(
        wear_daily_local, strategy_period_returns
    )
    metric = compute_metrics(strategy_result, ATTRIBUTION_START, ATTRIBUTION_END)
    core_friction_records.append({
        'Policy': strategy_name,
        'Contribution of gross proceeds from assets': float(wear_linked_local['Gross proceeds from assets']),
        'SORA Cash Contribution': float(wear_linked_local['SORA cash proceeds']),
        'Traded wear and tear': float(wear_linked_local['Traded wear and tear']),
        'Final net accrued gain': float(wear_linked_local.sum()),
        'Annual One-Way Turnover': float(metric['Annual One-Way Turnover']),
        'Maximum daily reconciliation error': float(reconciliation.abs().max()),
    })

    factor_table, factor_daily, factor_model_local = factor_performance_attribution(
        strategy_result, ATTRIBUTION_START, ATTRIBUTION_END
    )
    core_factor_records.append({
        'Policy': strategy_name,
        'Global stocks Beta': float(factor_model_local.params['Global equities']),
        'Medium-term national debt Beta': float(factor_model_local.params['Medium-term national debt']),
        'Commodities Beta': float(factor_model_local.params['Commodity basket']),
        'US$ Beta': float(factor_model_local.params['United States dollars']),
        'Alphaization of remnants': float(factor_model_local.params['const'] * TRADING_DAYS),
        'Alpha HAC t': float(factor_model_local.tvalues['const']),
        'Alpha HAC p': float(factor_model_local.pvalues['const']),
        'Return R Square': float(factor_model_local.rsquared),
    })

core_category_contribution = (
    pd.DataFrame(core_category_records)
    .groupby(['Policy', 'Asset Class'])['Cumulative proceeds contribution'].sum()
    .unstack('Asset Class').fillna(0.0)
    .reindex(CORE_COMPARISON_STRATEGIES)
)
core_friction_comparison = pd.DataFrame(core_friction_records).set_index('Policy')
core_factor_comparison = pd.DataFrame(core_factor_records).set_index('Policy')

display(core_category_contribution.style.format('{:+.2%}'))
display(core_friction_comparison.style.format({
    'Contribution of gross proceeds from assets': '{:+.2%}', 'SORA Cash Contribution': '{:+.2%}',
    'Traded wear and tear': '{:+.2%}', 'Final net accrued gain': '{:+.2%}',
    'Annual One-Way Turnover': '{:.2%}', 'Maximum daily reconciliation error': '{:.2e}',
}))
display(core_factor_comparison.style.format({
    'Global stocks Beta': '{:.2f}', 'Medium-term national debt Beta': '{:.2f}',
    'Commodities Beta': '{:.2f}', 'US$ Beta': '{:.2f}',
    'Alphaization of remnants': '{:+.2%}', 'Alpha HAC t': '{:.2f}',
    'Alpha HAC p': '{:.2f}', 'Return R Square': '{:.2f}',
}))

core_category_plot = core_category_contribution.rename(index=strategy_label)
fig, ax = plt.subplots(figsize=(12, 6))
core_category_plot.T.plot(
    kind='bar', ax=ax,
    color=[STRATEGY_COLORS[name] for name in core_category_contribution.index],
)
ax.axhline(0.0, color='black', linewidth=0.8)
ax.set_title('Three core strategies: cumulative benefit contribution from asset classes')
ax.set_xlabel('Asset class')
ax.set_ylabel('Cumulative yield contribution (percentage points)')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.tick_params(axis='x', rotation=20)
style_axis(ax, axis='y')
place_legend(ax)
fig.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()

core_factor_comparison = core_factor_comparison.reindex(CORE_COMPARISON_STRATEGIES)
factor_plot = core_factor_comparison[['Alphaization of remnants']].copy()
factor_plot.index = [strategy_label(name) for name in factor_plot.index]
fig, ax = plt.subplots(figsize=(10, 6))
alpha_values = factor_plot['Alphaization of remnants'].to_numpy(dtype=float)
alpha_bars = ax.bar(
    factor_plot.index, alpha_values,
    color=[STRATEGY_COLORS[name] for name in core_factor_comparison.index],
)
ax.bar_label(alpha_bars, labels=[f'{value:.2%}' for value in alpha_values],
             padding=4, fontsize=10)
ax.axhline(0.0, color='black', linewidth=0.8)
ax.set_title('Three core strategies: Alpha from the 4-fact model')
ax.set_xlabel('Policy')
ax.set_ylabel('Alpha')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.tick_params(axis='x', rotation=18)
style_axis(ax, axis='y')
fig.tight_layout()
plt.show()

if core_friction_comparison['Maximum daily reconciliation error'].max() > 1e-10:
    raise RuntimeError('The three core strategies are attributed horizontally to daily reconciliation errors.')
if any(results[name]['returns'].loc[ATTRIBUTION_START:ATTRIBUTION_END].index.max() >= HOLDOUT_START
       for name in CORE_COMPARISON_STRATEGIES):
    raise RuntimeError('The three core strategies are attributed horizontally to the reading of the lock test set.')
print('The three core strategy categories of contribution, trade wear and tear, and the 4-factor Alpha horizontal attribution were completed.')

