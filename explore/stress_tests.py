"""Pressure tests, external validation of samples, Bootstream and VaR/ES.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# Part V Pressure testing and external validation of samples
#
# 5.1 Basic stress tests
#
#
# Pressure tests are used only to compare strategic vulnerabilities and not to reverse adjustment parameters to enforce positive gains.
# --- Original Notebook code module 43-
stress_windows = {
    'Impact of the 2020 epidemic': ('2020-02-19', '2020-03-23'),
    '2022 stock, double-kill.': ('2022-01-03', '2022-12-30'),
}

stress_records = []
for scenario, (start, end) in stress_windows.items():
    period_returns = strategy_returns.loc[start:end, FOCUS_STRATEGIES]
    for strategy in period_returns.columns:
        r = period_returns[strategy].dropna()
        if len(r) < 2:
            continue
        stress_records.append({
            'Pressure scenario': scenario,
            'Policy': strategy,
            'Start date': r.index.min().date(),
            'End Date': r.index.max().date(),
            'Cumulative Return': (1.0 + r).prod() - 1.0,
            'Maximum Drawdown': drawdown_series(r).min(),
            'Annualized Volatility': r.std(ddof=1) * np.sqrt(TRADING_DAYS),
            'Worst day of earnings': r.min(),
        })

stress_test = pd.DataFrame(stress_records).set_index(['Pressure scenario', 'Policy'])
stress_display = stress_test.copy()
for column in ['Cumulative Return', 'Maximum Drawdown', 'Annualized Volatility', 'Worst day of earnings']:
    stress_display[column] = stress_display[column].map(lambda x: f'{x:.2%}')
display(stress_display)

# The cumulative benefits of pressure windows: three product strategies are the same, and the pressure phase is marked with a light rectangle.
stress_curve_returns = strategy_returns.loc[:RESEARCH_END, CORE_COMPARISON_STRATEGIES].dropna()
stress_curves = (1.0 + stress_curve_returns).cumprod() - 1.0
fig, ax = plt.subplots(figsize=(15, 7))
for name in CORE_COMPARISON_STRATEGIES:
    ax.plot(stress_curves.index, stress_curves[name], color=STRATEGY_COLORS[name],
            linewidth=3.2 if name == SUBMISSION_STRATEGY else 2.3,
            linestyle='-', label=strategy_label(name))
span_colors = ['#F4B183', '#9DC3E6']
for idx, (scenario, (start, end)) in enumerate(stress_windows.items()):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    ax.axvspan(start_ts, end_ts, color=span_colors[idx % len(span_colors)], alpha=0.18)
    ax.text(start_ts + (end_ts - start_ts) / 2, 0.98, scenario,
            transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9)
ax.axhline(0.0, color='#555555', linewidth=0.8)
ax.set_title('Cumulative gains from the core product strategy and historical stress phase')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0, 0.86, 1])
plt.show()

# 5.2 Single- and double-layer dynamic risk budget diagnosis
# --- Original Notebook code module 45-
dynamic_result = results[DYNAMIC_BUDGET_NAME]
latest_date = dynamic_result['weights'].index[-1]
latest_trigger = dynamic_result['triggers'].loc[dynamic_result['triggers']['Execution Date'].eq(latest_date)].iloc[-1]
latest_signal_date = pd.Timestamp(latest_trigger['Signal Date'])
position = returns.index.get_loc(latest_signal_date)
history = returns.iloc[position - LOOKBACK + 1:position + 1]
selected, score = select_top10(history)
target_budget = dynamic_risk_budget(history, selected, score)
covariance = ewma_covariance(history[selected])
latest_weights = dynamic_result['weights'].loc[latest_date, selected]
actual_rc = risk_contribution_share(latest_weights, covariance)

dynamic_diagnostics = pd.DataFrame({
    'Target risk budget': target_budget,
    'Actual risk contribution': actual_rc,
    'Target weights': latest_weights,
    'Motion Score': score[selected],
})
dynamic_diagnostics.index.name = 'Assets'
dynamic_display = dynamic_diagnostics.copy()
for column in dynamic_display.columns:
    dynamic_display[column] = dynamic_display[column].map(lambda x: f'{x:.2%}')
display(dynamic_display)

max_error = float((actual_rc - target_budget).abs().max())
print(f'Maximum risk contribution error in dynamic risk budget:{max_error:.8%}')
assert max_error <= 1e-4

hierarchical_result = results[HIERARCHICAL_BUDGET_NAME]
hierarchical_date = hierarchical_result['weights'].index[-1]
hierarchical_trigger = hierarchical_result['triggers'].loc[hierarchical_result['triggers']['Execution Date'].eq(hierarchical_date)].iloc[-1]
hierarchical_signal_date = pd.Timestamp(hierarchical_trigger['Signal Date'])
hierarchical_position = returns.index.get_loc(hierarchical_signal_date)
hierarchical_history = returns.iloc[hierarchical_position - LOOKBACK + 1:hierarchical_position + 1]
hierarchical_selected, hierarchical_score = select_top10(hierarchical_history)
hierarchical_budget = hierarchical_dynamic_risk_budget(
    hierarchical_history, hierarchical_selected, hierarchical_score
)
hierarchical_covariance = ewma_covariance(hierarchical_history[hierarchical_selected])
hierarchical_weights = hierarchical_result['weights'].loc[hierarchical_date, hierarchical_selected]
hierarchical_actual_rc = risk_contribution_share(hierarchical_weights, hierarchical_covariance)

hierarchical_diagnostics = pd.DataFrame({
    'Asset Class': assets.loc[hierarchical_selected, 'Asset Class'],
    'Target risk budget': hierarchical_budget,
    'Actual risk contribution': hierarchical_actual_rc,
    'Target weights': hierarchical_weights,
    'Motion Score': hierarchical_score[hierarchical_selected],
})
hierarchical_diagnostics.index.name = 'Assets'
hierarchical_display = hierarchical_diagnostics.copy()
for column in ['Target risk budget', 'Actual risk contribution', 'Target weights', 'Motion Score']:
    hierarchical_display[column] = hierarchical_display[column].map(lambda x: f'{x:.2%}')
display(hierarchical_display)

hierarchical_class_summary = hierarchical_diagnostics.groupby('Asset Class')[
    ['Target risk budget', 'Actual risk contribution', 'Target weights']
].sum()
hierarchical_class_display = hierarchical_class_summary.copy()
for column in hierarchical_class_display.columns:
    hierarchical_class_display[column] = hierarchical_class_display[column].map(lambda x: f'{x:.2%}')
display(hierarchical_class_display)

hierarchical_error = float((hierarchical_actual_rc - hierarchical_budget).abs().max())
print(f'Bulk budget double risk contribution error:{hierarchical_error:.8%}')
assert hierarchical_error <= 1e-4


garch_fit_table = pd.DataFrame(GARCH_FIT_DIAGNOSTICS)
garch_fit_summary = pd.DataFrame({
    'Value': [len(garch_fit_table), int((garch_fit_table['Depress marks'] != 0).sum())]
}, index=['GRCH Total proposed number', 'Number of Non-Zero Depression Marks'])
display(garch_fit_summary)

# 5.3 Asset integration of the core strategy
#
#
# This section examines only two strategies:
#
# 1. **Commercial confirmation of momentum plus strict risk budget for EWMA**;
# 2. **Top10+EWMA strict risk parity**.
#
# “overall assets” means that the assets are combined with positive weights at least once in a full historical return, and does not mean that a single turnkey holds all assets. Each strategy still chooses Top10 at any set point. The results are shown only in DataFrame and are not written in local files.
# --- Original Notebook code unit 47-
def build_asset_inclusion_summary(backtest_results):
    records = []
    for strategy_name, result in backtest_results.items():
        weights_history = result['weights'].copy().reindex(columns=ALL_ASSETS, fill_value=0.0)
        total_rebalances = len(weights_history)
        if 'end_of_day_weights' not in result or result['end_of_day_weights'].empty:
            raise RuntimeError(f'{strategy_name}There is a lack of real warehouse records at the end of the period.')
        latest_weights = result['end_of_day_weights'].loc[:RESEARCH_END].iloc[-1].reindex(
            ALL_ASSETS, fill_value=0.0
        )
        for ticker in ALL_ASSETS:
            positive_weights = weights_history[ticker][weights_history[ticker] > 1e-10]
            selected_count = int(positive_weights.count())
            records.append({
                'Policy': strategy_name,
                'Code': ticker,
                'Asset Class': assets.loc[ticker, 'Asset Class'],
                'Asset Name': assets.loc[ticker, 'Asset Name'],
                'Total number of transfers': total_rebalances,
                'Number of selections': selected_count,
                'Selection Ratio': selected_count / total_rebalances if total_rebalances else np.nan,
                'Date of first selection': positive_weights.index.min() if selected_count else pd.NaT,
                'Final selection date': positive_weights.index.max() if selected_count else pd.NaT,
                'Average weight on selection': positive_weights.mean() if selected_count else 0.0,
                'Maximum weight of history': positive_weights.max() if selected_count else 0.0,
                'Current holdings': bool(latest_weights[ticker] > 1e-10),
                'Current weight': float(latest_weights[ticker]),
            })
    return pd.DataFrame(records)


INCLUSION_ANALYSIS_STRATEGIES = [
    STRATEGY_NAMES['ewma_erc'],
    *CORE_COMPARISON_STRATEGIES,
]
focused_results = {name: results[name] for name in INCLUSION_ANALYSIS_STRATEGIES}
asset_inclusion_summary = build_asset_inclusion_summary(focused_results)

coverage_records = []
for strategy_name in INCLUSION_ANALYSIS_STRATEGIES:
    strategy_rows = asset_inclusion_summary.loc[
        asset_inclusion_summary['Policy'].eq(strategy_name)
    ]
    selected_assets = strategy_rows.loc[strategy_rows['Number of selections'] > 0, 'Code'].tolist()
    missing_assets = [ticker for ticker in ALL_ASSETS if ticker not in selected_assets]
    coverage_records.append({
        'Policy': strategy_name,
        'Number of asset pools': len(ALL_ASSETS),
        'Number of assets selected at least once in history': len(selected_assets),
        'Whether all assets are covered': len(missing_assets) == 0,
        'Assets never selected': 'None' if not missing_assets else ', '.join(missing_assets),
    })
coverage_check = pd.DataFrame(coverage_records).set_index('Policy')
display(coverage_check)

comparison = asset_inclusion_summary.pivot(
    index=['Asset Class', 'Code', 'Asset Name'], columns='Policy', values='Selection Ratio'
).fillna(0.0)
comparison = comparison.rename(columns={
    STRATEGY_NAMES['ewma_erc']: 'Trend agility: proportion selected',
    FULLY_DYNAMIC_BUDGET_NAME: 'Full Dynamic Double Layer EWMA: Selection Ratio',
    CORE_ATTACK_NAME: 'Fixed 70 + 30 per cent offensive: selection ratio',
    DYNAMIC_ATTACK_NAME: 'Dynamic 0-30% offensive: selected ratio',
    DYNAMIC_ATTACK_H20_NAME: 'Dynamic 0-30% offensive + 20-day confirmation: selected ratio',
})
comparison = comparison.map(lambda x: f'{x:.1%}')
display(comparison)

current_comparison = asset_inclusion_summary.pivot(
    index=['Asset Class', 'Code', 'Asset Name'], columns='Policy', values='Current weight'
).fillna(0.0)
current_comparison = current_comparison.rename(columns={
    STRATEGY_NAMES['ewma_erc']: 'Trend agility: current weight',
    FULLY_DYNAMIC_BUDGET_NAME: 'Full Dynamic Double Layer EWMA: Current weight',
    CORE_ATTACK_NAME: 'Fixed 70 + 30% offensive: current weight',
    DYNAMIC_ATTACK_NAME: 'Dynamic 0-30% offensive: current weight',
    DYNAMIC_ATTACK_H20_NAME: 'Dynamic 0-30% offensive + 20-day confirmation: current weight',
})
current_comparison = current_comparison.loc[current_comparison.max(axis=1).sort_values(ascending=False).index]
current_comparison = current_comparison.map(lambda x: f'{x:.2%}')
display(current_comparison)

# 5.4 Asset price trends and relevance
#
#
# This section uses the strategy to convert the actual intra-temporal price to the Singapore dollar (SGD). For comparison purposes, assets are consolidated at a common starting point of 100 and price movements are shown using a logarithmic axis; the correlation of assets is calculated using the Pearson-day rate of return of SGD for the same period. Both results are displayed in the same visualized window and are not written in the local picture file.
# --- Original Notebook code module 49 ---
# Asset price trends and correlations: shown in Notebook only, no pictures saved
visual_prices = prices_sgd.loc[(prices_sgd.index >= strategy_start) & (prices_sgd.index <= RESEARCH_END), ALL_ASSETS].dropna(how='any')
if visual_prices.empty:
    raise ValueError('There is no complete asset price data in the actual range.')

normalized_prices = visual_prices.div(visual_prices.iloc[0]).mul(100)
visual_returns = visual_prices.pct_change(fill_method=None).dropna(how='any')
asset_correlation = visual_returns.corr(method='pearson')

# Sets a high-diverse colour for all assets; all price lines are used uniformly.
asset_colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(ALL_ASSETS)))
asset_name_map = assets['Asset Name'].to_dict()

fig, price_ax = plt.subplots(figsize=(18, 9))
for idx, ticker in enumerate(ALL_ASSETS):
    price_ax.plot(
        normalized_prices.index,
        normalized_prices[ticker],
        color=asset_colors[idx],
        linestyle='-',
        linewidth=1.6,
        alpha=0.92,
        label=f"{ticker}  {asset_name_map.get(ticker, ticker)}"
    )
price_ax.axhline(100, color='#555555', linewidth=0.9, linestyle='-', alpha=0.7)
price_ax.set_yscale('log')
price_ax.set_title(
    f"Total asset SGD price movements (Cast of integration = 100, logarithmic axis) \n"
    f"{normalized_prices.index[0].date()}to{normalized_prices.index[-1].date()}",
    fontsize=15, fontweight='bold'
)
price_ax.set_ylabel('Normalized price ( logarithmic)')
price_ax.set_xlabel('Date')
style_axis(price_ax)
place_legend(price_ax)
fig.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()

fig, corr_ax = plt.subplots(figsize=(15, 13))
heatmap = corr_ax.imshow(
    asset_correlation.values,
    cmap='RdBu_r', vmin=-1, vmax=1,
    interpolation='nearest', aspect='equal'
)
corr_ax.set_title(
    'Total asset SGD-day rate of return (Pearson)',
    fontsize=15, fontweight='bold', pad=14
)
corr_ax.set_xticks(np.arange(len(ALL_ASSETS)))
corr_ax.set_yticks(np.arange(len(ALL_ASSETS)))
corr_ax.set_xticklabels(ALL_ASSETS, rotation=55, ha='right', fontsize=10)
corr_ax.set_yticklabels(ALL_ASSETS, fontsize=10)
corr_ax.set_xticks(np.arange(-0.5, len(ALL_ASSETS), 1), minor=True)
corr_ax.set_yticks(np.arange(-0.5, len(ALL_ASSETS), 1), minor=True)
corr_ax.grid(which='minor', color='white', linestyle='-', linewidth=0.35, alpha=0.55)
corr_ax.tick_params(which='minor', bottom=False, left=False)

# Marks two decimals directly in the matrix; dark cells are readable using white words.
for row_index in range(len(ALL_ASSETS)):
    for column_index in range(len(ALL_ASSETS)):
        value = float(asset_correlation.iloc[row_index, column_index])
        text_color = 'white' if abs(value) >= 0.55 else '#263238'
        corr_ax.text(
            column_index, row_index, f'{value:.2f}',
            ha='center', va='center', fontsize=10, color=text_color,
        )

colorbar = fig.colorbar(heatmap, ax=corr_ax, fraction=0.032, pad=0.025)
colorbar.set_label('Pearson correlation coefficient', rotation=270, labelpad=18)
colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2f}'))
colorbar.ax.tick_params(labelsize=10)
fig.tight_layout()
plt.show()

# 5.5 Mapping function for cumulative proceeds between certification areas
# --- Original Notebook code module 51 ---
def plot_all_strategy_oos(strategy_return_frame, color_map):
    oos_returns = strategy_return_frame.loc[OOS_START:RESEARCH_END].dropna(how='any')
    if oos_returns.empty:
        raise ValueError('There are no strategic gains to be compared between certification zones.')
    zero_row = pd.DataFrame(0.0, index=[OOS_START], columns=oos_returns.columns)
    oos_cumulative = (1.0 + pd.concat([zero_row, oos_returns]).sort_index()).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(17, 9))
    for strategy_name in oos_cumulative.columns:
        is_hl = strategy_name in HIGHLIGHT_STRATEGIES
        ax.plot(
            oos_cumulative.index, oos_cumulative[strategy_name],
            color=color_map[strategy_name],
            linewidth=3.0 if is_hl else 1.1,
            linestyle='-',
            alpha=1.0 if is_hl else 0.45,
            zorder=5 if is_hl else 1,
            label=strategy_label(strategy_name),
        )

    benchmark_oos_returns = benchmark_returns_sgd[['ES3.SI', 'VT']].reindex(
        oos_returns.index
    ).dropna(how='any')
    benchmark_zero = pd.DataFrame(
        0.0, index=[OOS_START], columns=benchmark_oos_returns.columns
    )
    benchmark_oos_cumulative = (
        1.0 + pd.concat([benchmark_zero, benchmark_oos_returns]).sort_index()
    ).cumprod() - 1.0
    benchmark_styles = {
        'ES3.SI': {'color': MARKET_BENCHMARK_COLORS['ES3.SI'], 'linestyle': '-'},
        'VT': {'color': MARKET_BENCHMARK_COLORS['VT'], 'linestyle': '-'},
    }
    for ticker, label in MARKET_BENCHMARK_NAMES.items():
        ax.plot(
            benchmark_oos_cumulative.index, benchmark_oos_cumulative[ticker],
            linewidth=2.6, alpha=1.0, zorder=6,
            label=f'Market benchmarks:{label}', **benchmark_styles[ticker],
        )
    ax.axvline(OOS_START, color='#AAB2BD', linestyle='-', linewidth=1.0,
               label='The starting point of the validation area')
    ax.axhline(0.0, color='#555555', linewidth=0.8)
    ax.set_title(
        f'All{len(oos_cumulative.columns)}Accumulated gains between formal strategies and double market benchmark certification'
        f'（{OOS_START.date()}to{RESEARCH_END.date()}）',
        fontsize=15, fontweight='bold',
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative rate of return between certified areas')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
    style_axis(ax)
    place_legend(ax)
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    plt.show()
    return oos_cumulative

# 5.6 Extension window inspection of the length of different certification areas
#
#
# This section sets the starting point for validation at 2025-01-01 and expands the end point of the window in order to observe whether the strategic conclusions depend on a particular certification length. These windows are in the research phase and are all strictly closed to 2026-07-31, without reading the locking test set after 2026-08-01.
#
# All official strategies go into detail. The graphics focus on comparing the original Top10+EWMA strict risk parities and Top10+ Full Dynamic Double Layer+VIX open silos, with thermal effort to show the full strategy ratio of Sharp over different proven lengths.
#
# The current study covers the first 19 months in full and ends up to 2026-07-31; subsequent months are clearly marked as data not allowed. Annualized earnings from one to several months and Sharp are heavily influenced by a small number of trading days, which are used only for stability observations.
# --- Original Notebook code module 53--
# Fixed validation starting point, expanding endpoints on a monthly basis; future windows are not filled with data that have not yet occurred.
available_oos_end = min(
    RESEARCH_END,
    min(result['returns'].dropna().index.max() for result in results.values())
)

horizon_coverage_records = []
complete_horizons = []
partial_horizon = None
for months in range(1, 25):
    planned_end = (OOS_START + pd.DateOffset(months=months) - pd.Timedelta(days=1)).normalize()
    if available_oos_end >= planned_end:
        status = 'Full Overwrite'
        actual_end = planned_end
        complete_horizons.append((months, planned_end, f'{months}Month'))
    elif OOS_START <= available_oos_end and available_oos_end.to_period('M') == planned_end.to_period('M'):
        status = 'The current month is incomplete'
        actual_end = available_oos_end
        partial_horizon = (months, available_oos_end, f'{months}Month (incomplete)')
    else:
        status = 'Data not yet covered'
        actual_end = pd.NaT
    horizon_coverage_records.append({
        'Validation Length': f'{months}Month',
        'Plan end.': planned_end.date(),
        'Actual available endpoint': actual_end.date() if pd.notna(actual_end) else None,
        'Overwrite Status': status,
    })

horizon_coverage = pd.DataFrame(horizon_coverage_records).set_index('Validation Length')
display(horizon_coverage)

evaluated_horizons = complete_horizons.copy()
if partial_horizon is not None:
    evaluated_horizons.append(partial_horizon)

horizon_metric_frames = []
for months, window_end, horizon_label in evaluated_horizons:
    frame = pd.DataFrame({
        strategy_name: compute_metrics(result, OOS_START, window_end)
        for strategy_name, result in results.items()
    }).T
    frame['Number of months of windows'] = months
    frame['Window End'] = window_end
    frame['Window Status'] = 'Incomplete' if 'Incomplete' in horizon_label else 'Full'
    frame['Validation Length'] = horizon_label
    frame['Policy'] = frame.index
    horizon_metric_frames.append(frame.reset_index(drop=True))

oos_horizon_metrics = pd.concat(horizon_metric_frames, ignore_index=True)
oos_horizon_metrics = oos_horizon_metrics.set_index(['Validation Length', 'Policy'])

# Detailed DataFrame maintains all policies; displays are presented in Chinese format and limited to core indicators.
oos_horizon_metrics_display = oos_horizon_metrics[
    ['Window End', 'Window Status', 'Cumulative Return', 'Annualized Return', 'Annualized Volatility',
     'Sharpe Ratio', 'Maximum Drawdown', 'Annual One-Way Turnover', 'Number of Rebalances', 'Number of Trade Legs']
].copy()
for column in ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown', 'Annual One-Way Turnover']:
    oos_horizon_metrics_display[column] = oos_horizon_metrics_display[column].map(
        lambda value: f'{value:.2%}'
    )
oos_horizon_metrics_display['Sharpe Ratio'] = oos_horizon_metrics_display['Sharpe Ratio'].map(
    lambda value: f'{value:.2f}'
)
oos_horizon_metrics_display['Number of Rebalances'] = oos_horizon_metrics_display['Number of Rebalances'].astype(int)
oos_horizon_metrics_display['Number of Trade Legs'] = oos_horizon_metrics_display['Number of Trade Legs'].astype(int)
display(oos_horizon_metrics_display)

# Each window reports the highest cumulative returns and the highest Sharp ' s strategies respectively, avoiding observing only the last compartment.
horizon_best_records = []
for horizon_label, frame in oos_horizon_metrics.groupby(level='Validation Length', sort=False):
    local = frame.droplevel('Validation Length')
    return_winner = local['Cumulative Return'].idxmax()
    sharpe_winner = local['Sharpe Ratio'].idxmax()
    horizon_best_records.append({
        'Validation Length': horizon_label,
        'Window End': local['Window End'].iloc[0].date(),
        'Maximum accumulated gains strategy': return_winner,
        'Maximum accumulated proceeds': local.loc[return_winner, 'Cumulative Return'],
        'Sharp\'s top strategy.': sharpe_winner,
        'Supreme Sharp.': local.loc[sharpe_winner, 'Sharpe Ratio'],
    })
horizon_best = pd.DataFrame(horizon_best_records).set_index('Validation Length')
horizon_best_display = horizon_best.copy()
horizon_best_display['Maximum accumulated proceeds'] = horizon_best_display['Maximum accumulated proceeds'].map(lambda x: f'{x:.2%}')
horizon_best_display['Supreme Sharp.'] = horizon_best_display['Supreme Sharp.'].map(lambda x: f'{x:.2f}')
display(horizon_best_display)

baseline_name = BASELINE_STRATEGY
candidate_name = SUBMISSION_STRATEGY
focus = oos_horizon_metrics.reset_index()
focus = focus[focus['Policy'].isin([baseline_name, candidate_name])].copy()
focus['Show Months'] = focus['Validation Length'].str.extract(r'(\d+)').astype(int)

style_map = {
    baseline_name: {'color': STRATEGY_COLORS[baseline_name], 'linestyle': '-', 'marker': 'o'},
    candidate_name: {'color': STRATEGY_COLORS[candidate_name], 'linestyle': '-', 'marker': 's'},
}
fig, ax = plt.subplots(figsize=(10, 5.8))
for strategy_name in [baseline_name, candidate_name]:
    subset = focus[focus['Policy'] == strategy_name].sort_values('Show Months')
    style = style_map[strategy_name]
    ax.plot(subset['Show Months'], subset['Cumulative Return'], label=strategy_label(strategy_name),
            linewidth=2.4, markersize=4, **style)
ax.set_title('Equivalent benchmarks and elixir: cumulative gains from different verification lengths')
ax.set_ylabel('Cumulative Return')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.set_xlabel('Number of certification months from 2025-01-01')
ax.set_xticks(range(1, max(focus['Show Months']) + 1))
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()

fig, ax = plt.subplots(figsize=(10, 5.8))
for strategy_name in [baseline_name, candidate_name]:
    subset = focus[focus['Policy'] == strategy_name].sort_values('Show Months')
    ax.plot(subset['Show Months'], subset['Sharpe Ratio'], label=strategy_label(strategy_name),
            linewidth=2.4, markersize=4, **style_map[strategy_name])
ax.set_title('Equivalent benchmark and elixir: different verification lengths of Sharp')
ax.set_ylabel('Sharpe Ratio')
ax.set_xlabel('Number of certification months from 2025-01-01')
ax.set_xticks(range(1, max(focus['Show Months']) + 1))
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()

# All the official strategies of Sharp's stability struggle.
sharpe_matrix = oos_horizon_metrics['Sharpe Ratio'].unstack(level='Validation Length')
ordered_horizons = [label for _, _, label in evaluated_horizons]
sharpe_matrix = sharpe_matrix.reindex(columns=ordered_horizons)
fig, ax = plt.subplots(figsize=(17, max(7, 0.48 * len(sharpe_matrix))))
sharpe_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    'sharpe_stability', ['#2b549c', '#f0f0f0', '#b82626']
)
image = ax.imshow(sharpe_matrix.values, aspect='auto', cmap=sharpe_cmap)
ax.set_xticks(np.arange(len(sharpe_matrix.columns)))
ax.set_xticklabels(sharpe_matrix.columns, rotation=45, ha='right', fontsize=10)
ax.set_yticks(np.arange(len(sharpe_matrix.index)))
ax.set_yticklabels([strategy_label(name) for name in sharpe_matrix.index], fontsize=10)
ax.set_xlabel('Length of validation starting with 2025-01-01')
ax.set_title('Total formal strategy: Sharp ratio for different verification lengths')
colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
colorbar.set_label('Annualized Sharp')
for row in range(sharpe_matrix.shape[0]):
    for column in range(sharpe_matrix.shape[1]):
        value = sharpe_matrix.iat[row, column]
        if pd.notna(value):
            ax.text(column, row, f'{value:.2f}', ha='center', va='center',
                    fontsize=10, color='#27313B')
colorbar.ax.tick_params(labelsize=10)
fig.tight_layout()
plt.show()

first_unavailable = complete_horizons[-1][0] + (2 if partial_horizon is not None else 1)
print(
    f'Strict deadline for research data{available_oos_end.date()}；'
    f'Full cover 1 to{complete_horizons[-1][0]}Months;'
    f'{first_unavailable}Research indicators are not generated for 24 months.'
)

# 5.7 Lock-in retention test series: 2026-08-01 to 2026-10-01
#
#
# The unified downloading process will result in a complete original pattern, but all previously studied objects, indicators and graphics have been forced to be cut to 2026-07-31; this section is the only unit in Notebook that allows the transfer of 2026-08-01 and later data into a strategy evaluation function. This isolation belongs to the logical isolation of the code claiming protection, rather than to the physical isolation of independent documents or processes of independence. All policy rules, parameters, asset pools and warehousing mechanisms have been identified in the previous study and this section does not modify parameters or screening strategies based on test results.
#
# The tests are day-by-day: only 252 trade-day information prior to that date can be used to generate signals and risk estimates. In order to preserve the real hold when 2026-08-01 enters the test set, each policy is re-run from the original return starting point to the locked end point, but the indicator only counts the locked range. The test results do not return to `results` or `strategy_returns` from the research phase.
#
# The `end` parameter for Yahoo Finance is an exclusive date, so the download endpoint is set at 2026-10-02 to attempt to obtain 2026-10-01 data. If, at the time of implementation, certain markets had not provided a closing price as at 1 October, the actual test end would be based on a common available final date for all assets, with a clear indication that no future filling would take place.
# --- Original Notebook code module 55--
# The results must all be stopped before the lockdown test set.
research_end_by_strategy = pd.Series({
    name: result['returns'].dropna().index.max()
    for name, result in results.items()
}, name='Final date of research findings')
if not (research_end_by_strategy < HOLDOUT_START).all():
    leaked = research_end_by_strategy[research_end_by_strategy >= HOLDOUT_START]
    raise RuntimeError(f'The following results show data leakage:{leaked.to_dict()}')

# Lock the test module to allow SORA after 2026-07-31.
sora_holdout_df = load_sora_daily(HOLDOUT_END)
assert sora_holdout_df.index.max() <= HOLDOUT_END

holdout_results = {
    name: run_dynamic_backtest(
        name, allocator, covariance_function, end_date=HOLDOUT_END,
        sora_frame=sora_holdout_df, allow_holdout=True,
    )
    for name, (allocator, covariance_function) in strategy_specs.items()
}

holdout_results.update({
    CORE_ATTACK_NAME: run_core_attack_backtest(
        CORE_ATTACK_NAME, end_date=HOLDOUT_END,
        sora_frame=sora_holdout_df, allow_holdout=True,
    ),
    DYNAMIC_ATTACK_NAME: run_dynamic_attack_backtest(
        DYNAMIC_ATTACK_NAME, end_date=HOLDOUT_END,
        sora_frame=sora_holdout_df, allow_holdout=True,
    ),
    DYNAMIC_ATTACK_H20_NAME: run_dynamic_attack_backtest(
        DYNAMIC_ATTACK_H20_NAME, end_date=HOLDOUT_END,
        sora_frame=sora_holdout_df, allow_holdout=True, short_trend_window=ATTACK_SHORT_MA,
    ),
    STATIC_CLASSIC_NAME: run_all_asset_monthly_backtest(
        STATIC_CLASSIC_NAME, target_all_asset_classic, end_date=HOLDOUT_END,
        sora_frame=sora_holdout_df, allow_holdout=True,
    ),})
required_holdout_strategies = {FULLY_DYNAMIC_BUDGET_NAME, CORE_ATTACK_NAME, DYNAMIC_ATTACK_NAME, DYNAMIC_ATTACK_H20_NAME}
if not required_holdout_strategies.issubset(holdout_results):
    raise RuntimeError('The four core product strategies were not all tested outside the locked sample.')

actual_holdout_end = min(
    HOLDOUT_END,
    min(result['returns'].dropna().index.max() for result in holdout_results.values()),
)
if actual_holdout_end < HOLDOUT_START:
    raise RuntimeError('There are no commonly available strategic benefits for locking the test area.')

holdout_metric_table = pd.DataFrame({
    name: compute_metrics(result, HOLDOUT_START, actual_holdout_end)
    for name, result in holdout_results.items()
}).T
holdout_metric_table.index.name = 'Policy'
holdout_metric_table = holdout_metric_table.sort_values(
    'Sharpe Ratio', ascending=True, kind='mergesort'
)
holdout_metric_table.insert(
    0, 'Shape Sorting Number (low to high)', np.arange(1, len(holdout_metric_table) + 1)
)

holdout_display = holdout_metric_table.copy()
for column in ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown',
               'Positive-Day Ratio', 'Annual One-Way Turnover', 'Annual Two-Way Traded Notional', 'Average Gross Exposure']:
    holdout_display[column] = holdout_display[column].map(lambda x: f'{x:.2%}')
for column in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
    holdout_display[column] = holdout_display[column].map(lambda x: f'{x:.2f}')
holdout_display['Number of Rebalances'] = holdout_display['Number of Rebalances'].astype(int)
holdout_display['Number of Trade Legs'] = holdout_display['Number of Trade Legs'].astype(int)
display(holdout_display)

holdout_returns = pd.concat(
    [result['returns'].rename(name) for name, result in holdout_results.items()],
    axis=1,
).loc[HOLDOUT_START:actual_holdout_end].dropna(how='any')

ranked = holdout_metric_table.sort_values('Sharpe Ratio')
colors = [
    HIGHLIGHT_COLORS.get(strategy_name, MUTED_COLOR)
    for strategy_name in ranked.index
]

ranked_labels = [strategy_label(name) for name in ranked.index]
fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(ranked_labels, ranked['Cumulative Return'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title(f'Locks the cumulative rate of return of the test set{HOLDOUT_START.date()}to{actual_holdout_end.date()}）')
ax.set_xlabel('Cumulative Return')
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
style_axis(ax, axis='x')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(ranked_labels, ranked['Sharpe Ratio'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title(f'Lock test set year-old Sharp.{HOLDOUT_START.date()}to{actual_holdout_end.date()}）')
ax.set_xlabel('Sharpe Ratio')
style_axis(ax, axis='x')
fig.tight_layout()
plt.show()

holdout_summary = pd.DataFrame({
    'Value': [
        HOLDOUT_START.date(), actual_holdout_end.date(), len(holdout_returns),
        len(holdout_results), research_end_by_strategy.max().date(),
    ]
}, index=['Lock Test Start', 'Actual test endpoint', 'Common Test Trade Day', 'Number of testing strategies', 'Late date for research results'])
display(holdout_summary)
print('Lock-in test completed: As a short-term observation of the area not visible, not for long-term Alpha certification or parameter selection.')

# 5.8 Extended stress testing: historical shocks, costs and implementation delays
#
#
# This section only uses research data up to 2026-07-31 and does not read the locking test set. The tests were divided into three groups:
#
# 1. **History Event Window**: Declines in risk assets in the fourth quarter of 2018, rebounding after March 2020, appreciation of the United States dollar in 2022, BITs collapse, commodity booms, abnormal fluctuations in crude oil and pound sterling and British asset shocks;
# 2. ** Transaction cost sensitivity**: Quantification of actual trade frictions as a whole to 0.5, 1.0, 1.5 and 2.0 times, with the same back-to-back path and withholding taxes;
# ** One-day delay in implementation**: maintenance of signals, selection of assets and target weights, postponement of one transaction date for each target weight, and recalculation of transaction costs, financing costs and warehousing.
#
# Historical windows are situational analyses and do not represent the only official definition of events; multiple windows can overlap. Cost sensitivity is a fixed decision path test that does not assume that cost changes will in turn change the calibration signal. The delay in testing measures the risk at the time of execution rather than simply moving the whole sequence of proceeds.
# --- Original Notebook code unit 57 ---
# Prevents unexpected reading of locking test sets in this section.
if strategy_returns.index.max() > RESEARCH_END:
    raise RuntimeError('Pressure tests have detected research returns exceeding 2026-07-31.')

extended_stress_windows = {
    '2018Q4 risk asset decline': ('2018-10-01', '2018-12-31'),
    'It\'s a quick rebound after the 2020 outbreak.': ('2020-03-24', '2020-08-31'),
    'A fast appreciation of $2022': ('2022-01-03', '2022-09-27'),
    'Bitcoin was higher, and then independence fell.': ('2021-11-10', '2022-06-17'),
    '2021 to 2022 commodity boom': ('2021-11-01', '2022-06-08'),
    'Unusually volatile 2020 crude oil': ('2020-03-09', '2020-04-28'),
    'Pound2022 and British asset shock': ('2022-09-01', '2022-10-31'),
}

historical_stress_records = []
for scenario, (start, end) in extended_stress_windows.items():
    scenario_returns = strategy_returns.loc[start:end, FOCUS_STRATEGIES]
    for strategy_name in scenario_returns.columns:
        r = scenario_returns[strategy_name].dropna()
        if len(r) < 2:
            continue
        historical_stress_records.append({
            'Pressure scenario': scenario,
            'Policy': strategy_name,
            'Start date of transaction': r.index.min().date(),
            'End date of transaction': r.index.max().date(),
            'Number of trading days': len(r),
            'Cumulative Return': float((1.0 + r).prod() - 1.0),
            'Maximum Drawdown': float(drawdown_series(r).min()),
            'Annualized Volatility': float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            'Worst day of earnings': float(r.min()),
        })

extended_stress_test = pd.DataFrame(historical_stress_records).set_index(['Pressure scenario', 'Policy'])
extended_stress_display = extended_stress_test.copy()
for column in ['Cumulative Return', 'Maximum Drawdown', 'Annualized Volatility', 'Worst day of earnings']:
    extended_stress_display[column] = extended_stress_display[column].map(lambda x: f'{x:.2%}')
display(extended_stress_display)

stress_summary_records = []
for scenario, frame in extended_stress_test.groupby(level='Pressure scenario', sort=False):
    local = frame.droplevel('Pressure scenario')
    best_return_strategy = local['Cumulative Return'].idxmax()
    best_drawdown_strategy = local['Maximum Drawdown'].idxmax()
    scenario_start, scenario_end = extended_stress_windows[scenario]
    baseline_returns = strategy_returns.loc[scenario_start:scenario_end, BASELINE_STRATEGY].dropna()
    baseline_cumulative_return = float((1.0 + baseline_returns).prod() - 1.0)
    stress_summary_records.append({
        'Pressure scenario': scenario,
        'Maximum yield strategy': best_return_strategy,
        'Maximum accumulated proceeds': local.loc[best_return_strategy, 'Cumulative Return'],
        'Revert Minimum Policy': best_drawdown_strategy,
        'Minimal maximum retreat': local.loc[best_drawdown_strategy, 'Maximum Drawdown'],
        'Case Line Cumulative Gain': baseline_cumulative_return,
    })
stress_scenario_summary = pd.DataFrame(stress_summary_records).set_index('Pressure scenario')
stress_scenario_summary_display = stress_scenario_summary.copy()
for column in ['Maximum accumulated proceeds', 'Minimal maximum retreat', 'Case Line Cumulative Gain']:
    stress_scenario_summary_display[column] = stress_scenario_summary_display[column].map(lambda x: f'{x:.2%}')
display(stress_scenario_summary_display)

stress_heatmap = extended_stress_test['Cumulative Return'].unstack(level='Pressure scenario')
stress_heatmap = stress_heatmap.reindex(columns=list(extended_stress_windows))
limit = float(np.nanmax(np.abs(stress_heatmap.values)))
fig, ax = plt.subplots(figsize=(17, max(8, 0.45 * len(stress_heatmap))))
image = ax.imshow(stress_heatmap.values, aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
ax.set_xticks(np.arange(len(stress_heatmap.columns)))
ax.set_xticklabels(stress_heatmap.columns, rotation=35, ha='right', fontsize=10)
ax.set_yticks(np.arange(len(stress_heatmap.index)))
ax.set_yticklabels([strategy_label(name) for name in stress_heatmap.index], fontsize=10)
ax.set_title('Cumulative rate of return for the focus strategy in the historical pressure window')
colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
colorbar.set_label('Cumulative Return')
colorbar.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
for row in range(stress_heatmap.shape[0]):
    for column in range(stress_heatmap.shape[1]):
        value = stress_heatmap.iat[row, column]
        if pd.notna(value):
            text_color = 'white' if abs(value) > 0.62 * limit else '#27313B'
            ax.text(column, row, f'{value:.2%}', ha='center', va='center',
                    fontsize=10, color=text_color)
colorbar.ax.tick_params(labelsize=10)
fig.tight_layout()
plt.show()

# 5.8.1 Transaction cost sensitivity
#
#
# The current formal strategy is one unilateral 0.05 per cent. This section maintains the signals, target weights and reconciliation dates, with fixed transaction costs being reduced to 0.5, 1.0, 1.5 and 2.0 times. The actual cost of the sub-market is also shown in the stand-alone scenario in section 4.5 and is not used here.
# --- Original Notebook code module 59-
TRANSACTION_COST_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]


def returns_under_cost(result, cost_multiplier):
    """Revalue the cost without changing the signal and target weight path.

The uniform unilateral 0.05 per cent transaction fee is reduced by a factor, without changing the signal and hold path.
"""
    original_returns = result['returns'].loc[:RESEARCH_END].copy()
    original_cost_rate = result['transaction_cost'].reindex(original_returns.index).fillna(0.0)
    new_cost_rate = original_cost_rate * float(cost_multiplier)
    if (original_cost_rate >= 1.0).any() or (new_cost_rate >= 1.0).any():
        raise RuntimeError('A cost rate of 100 per cent or more cannot be reformulated.')
    after_tax_gross_factor = (1.0 + original_returns) / (1.0 - original_cost_rate)
    return after_tax_gross_factor * (1.0 - new_cost_rate) - 1.0


cost_sensitivity_records = []
for strategy_name in FOCUS_STRATEGIES:
    result = results[strategy_name]
    for cost_multiplier in TRANSACTION_COST_MULTIPLIERS:
        adjusted_returns = returns_under_cost(result, cost_multiplier)
        if cost_multiplier == 1.0:
            reconstruction_error = float(
                (adjusted_returns - result['returns'].loc[:RESEARCH_END]).abs().max()
            )
            if reconstruction_error > 1e-12:
                raise RuntimeError(
                    f'{strategy_name}Base cost gain re-engineering failed:{reconstruction_error:.3e}'
                )
        adjusted_result = {
            **result,
            'returns': adjusted_returns.rename(strategy_name),
        }
        metric = compute_metrics(adjusted_result, None, RESEARCH_END)
        cost_sensitivity_records.append({
            'Policy': strategy_name,
            'Fixed cost multiplier': cost_multiplier,
            'Cumulative Return': metric['Cumulative Return'],
            'Annualized Return': metric['Annualized Return'],
            'Sharpe Ratio': metric['Sharpe Ratio'],
            'Maximum Drawdown': metric['Maximum Drawdown'],
            'Annual One-Way Turnover': metric['Annual One-Way Turnover'],
        })

cost_sensitivity = pd.DataFrame(cost_sensitivity_records).set_index(['Policy', 'Fixed cost multiplier'])
cost_sensitivity_display = cost_sensitivity.copy()
for column in ['Cumulative Return', 'Annualized Return', 'Maximum Drawdown', 'Annual One-Way Turnover']:
    cost_sensitivity_display[column] = cost_sensitivity_display[column].map(lambda x: f'{x:.2%}')
cost_sensitivity_display['Sharpe Ratio'] = cost_sensitivity_display['Sharpe Ratio'].map(lambda x: f'{x:.2f}')
display(cost_sensitivity_display)

focus_cost_strategies = FOCUS_STRATEGIES.copy()
display(cost_sensitivity_display.loc[focus_cost_strategies])
cost_plot = cost_sensitivity.reset_index()
cost_plot = cost_plot[cost_plot['Policy'].isin(focus_cost_strategies)]
cost_colors = {name: HIGHLIGHT_COLORS.get(name, '#4C78A8') for name in focus_cost_strategies}
fig, ax = plt.subplots(figsize=(11, 6))
for strategy_name in focus_cost_strategies:
    subset = cost_plot[cost_plot['Policy'] == strategy_name].sort_values('Fixed cost multiplier')
    ax.plot(subset['Fixed cost multiplier'], subset['Cumulative Return'], marker='o', linestyle='-',
            color=cost_colors[strategy_name], label=strategy_label(strategy_name), linewidth=2)
ax.set_title('Focus strategy: Impact of transaction costs on cumulative gains')
ax.set_ylabel('Cumulative Return')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.set_xlabel('Fixed 0.05 per cent transaction cost multiplier')
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()

fig, ax = plt.subplots(figsize=(11, 6))
for strategy_name in focus_cost_strategies:
    subset = cost_plot[cost_plot['Policy'] == strategy_name].sort_values('Fixed cost multiplier')
    ax.plot(subset['Fixed cost multiplier'], subset['Sharpe Ratio'], marker='o', linestyle='-',
            color=cost_colors[strategy_name], label=strategy_label(strategy_name), linewidth=2)
ax.set_title('Focus strategy: Impact of transaction costs on Sharp ratio')
ax.set_ylabel('Sharpe Ratio')
ax.set_xlabel('Fixed 0.05 per cent transaction cost multiplier')
style_axis(ax)
place_legend(ax)
fig.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()

# 5.8.2 Delaying a transaction date is sensitive degrees
#
#
# The official strategy is closed on the next trading day after the signal. This section delays the date of the closing of the existing collections by a further date to be used to test the additional friction of execution; the delay scenario continues to use the full master's return of a uniform unilateral transaction fee of 0.05 per cent.
# --- Original Notebook code unit 61--
def one_day_delayed_execution(result):
    """An additional trade date is postponed from the official implementation date and the original strategic cost structure is followed."""
    research_calendar = returns.loc[strategy_start:RESEARCH_END].index
    original_targets = result['weights'].loc[:RESEARCH_END].copy()
    trade_assets = list(result.get('trade_assets', ALL_ASSETS))
    original_targets = original_targets.reindex(columns=trade_assets, fill_value=0.0)
    delayed_targets = {}
    for execution_date, target in original_targets.iterrows():
        location = research_calendar.searchsorted(execution_date, side='right')
        if location < len(research_calendar):
            delayed_targets[research_calendar[location]] = target.copy()

    current_weights = pd.Series(0.0, index=trade_assets)
    result_rf = result['risk_free_daily'].reindex(research_calendar).ffill()
    result_funding = result['funding_daily'].reindex(research_calendar).ffill()
    if result_rf.isna().any() or result_funding.isna().any():
        raise RuntimeError(f"{result['name']}SORA coverage is incomplete.")
    return_source = result.get('instrument_returns', portfolio_returns_for_backtest)
    return_source = return_source.reindex(index=research_calendar, columns=trade_assets).fillna(0.0)
    asset_cost = pd.Series(result.get('asset_cost_rate', COST), index=trade_assets, dtype=float)
    daily_records, turnover_records, trade_leg_records, cost_records, weight_records = [], [], [], [], []
    pre_return_records, end_of_day_records, trade_delta_records = [], [], []

    for date in research_calendar:
        pre_return_records.append(current_weights.rename(date))
        day_return = return_source.loc[date]
        cash_weight = 1.0 - float(current_weights.sum())
        cash_rate = float(result_rf.loc[date] if cash_weight >= 0 else result_funding.loc[date])
        before_cost_return = float(current_weights @ day_return) + cash_weight * cash_rate
        if 1.0 + before_cost_return <= 0:
            raise RuntimeError('The net value of the additional delay strategy is not positive.')
        current_weights = current_weights * (1.0 + day_return) / (1.0 + before_cost_return)

        cost_rate = 0.0
        if date in delayed_targets:
            target = delayed_targets[date]
            delta = target - current_weights
            turnover = float(delta.abs().sum())
            trade_legs = int((delta.abs() > 1e-10).sum())
            cost_rate = float((delta.abs() * asset_cost).sum())
            current_weights = target.copy()
            turnover_records.append((date, turnover))
            trade_leg_records.append((date, trade_legs))
            cost_records.append((date, cost_rate))
            weight_records.append(current_weights.rename(date))
            trade_delta_records.append(delta.rename(date))

        net_return = (1.0 - cost_rate) * (1.0 + before_cost_return) - 1.0
        daily_records.append((date, net_return))
        end_of_day_records.append(current_weights.rename(date))

    delayed_result = {
        'name': result['name'] + '(Additional delay of 1 day)',
        'returns': pd.Series(dict(daily_records), name=result['name'] + '(Additional delay of 1 day)'),
        'turnover': pd.Series(dict(turnover_records), dtype=float),
        'trade_legs': pd.Series(dict(trade_leg_records), dtype=float),
        'transaction_cost': pd.Series(dict(cost_records), dtype=float),
        'weights': pd.DataFrame(weight_records), 'risk_free_daily': result_rf,
        'pre_return_weights': pd.DataFrame(pre_return_records),
        'end_of_day_weights': pd.DataFrame(end_of_day_records),
        'trade_deltas': pd.DataFrame(trade_delta_records),
        'funding_daily': result_funding, 'asset_cost_rate': asset_cost,
        'trade_assets': trade_assets, 'instrument_returns': return_source,
    }
    return delayed_result


delayed_results = {
    name: one_day_delayed_execution(results[name])
    for name in FOCUS_STRATEGIES
}

delay_sensitivity_records = []
for strategy_name in FOCUS_STRATEGIES:
    original_result = results[strategy_name]
    original_metric = compute_metrics(original_result, None, RESEARCH_END)
    delayed_metric = compute_metrics(delayed_results[strategy_name], None, RESEARCH_END)
    delay_sensitivity_records.append({
        'Policy': strategy_name,
        'Original cumulative rate of return': original_metric['Cumulative Return'],
        'Cumulative rate of return after delay': delayed_metric['Cumulative Return'],
        'Change in cumulative proceeds': delayed_metric['Cumulative Return'] - original_metric['Cumulative Return'],
        'Former Sharp ratio': original_metric['Sharpe Ratio'],
        'Sharp ratio after delay': delayed_metric['Sharpe Ratio'],
        'Sharp changes': delayed_metric['Sharpe Ratio'] - original_metric['Sharpe Ratio'],
        'Max. Back.': original_metric['Maximum Drawdown'],
        'Maximum retreat after delay': delayed_metric['Maximum Drawdown'],
        'Annual change rate after delay': delayed_metric['Annual One-Way Turnover'],
    })

execution_delay_test = pd.DataFrame(delay_sensitivity_records).set_index('Policy')
execution_delay_display = execution_delay_test.copy()
for column in ['Original cumulative rate of return', 'Cumulative rate of return after delay', 'Change in cumulative proceeds',
               'Max. Back.', 'Maximum retreat after delay', 'Annual change rate after delay']:
    execution_delay_display[column] = execution_delay_display[column].map(lambda x: f'{x:.2%}')
for column in ['Former Sharp ratio', 'Sharp ratio after delay', 'Sharp changes']:
    execution_delay_display[column] = execution_delay_display[column].map(lambda x: f'{x:.2f}')
display(execution_delay_display.sort_values('Change in cumulative proceeds'))

delay_plot = execution_delay_test.sort_values('Change in cumulative proceeds')
bar_colors = [
    HIGHLIGHT_COLORS.get(name, '#4C78A8')
    for name in delay_plot.index
]
delay_labels = [strategy_label(name) for name in delay_plot.index]
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(delay_labels, delay_plot['Change in cumulative proceeds'], color=bar_colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Additional delay of one transaction date: change in cumulative proceeds')
ax.set_xlabel('Policy for delayed and reduced')
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
style_axis(ax, axis='x')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(delay_labels, delay_plot['Sharp changes'], color=bar_colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Additional delay of one transaction date: change in Sharp ratio')
ax.set_xlabel('Policy for delayed and reduced')
style_axis(ax, axis='x')
fig.tight_layout()
plt.show()

print(
    f'Pressure test data used as of{strategy_returns.index.max().date()}；'
    f'Lock test start to{HOLDOUT_START.date()}, not read in this section.'
)

# 5.9 Assumptions of combined shock: current combination static pressure test
#
# This section restores the economically meaningful simulation scenario tests and regularizes the following three core strategies:
#
# 1. **Top10 + fully dynamic double risk budget + EWMA strict risk parity**;
# 2. ** fully dynamic two-tiered EWMA risk budget + 70 per cent core + 30 per cent decentralized offensive **;
# 3. ** EBMA risk budget + dynamic 0% - 30% decentralized offensive **.
#
# The table also shows their cumulative return ranking up to the research period of 2026-07-31. The ranking, current hold and SORA are all based on research-period data and do not read lock-in test sets.
#
# The shock is interpreted as a change in the total return of SSD for a month and at the same time as a combination of weights currently recorded. It measures static losses when “shocks occur immediately and fail to adjust” and does not represent projections of future gains, nor is it used for reverse optimization parameters.
# --- Original Notebook code module 63-
# The reading of the lock-down test set is prohibited by targeting the comparator solely on the basis of the cumulative proceeds from the study area.
research_strategy_returns = strategy_returns.loc[:RESEARCH_END].copy()
if research_strategy_returns.index.max() >= HOLDOUT_START:
    raise RuntimeError('Simulation pressure tests read lock test sets.')

research_cumulative_ranking = (
    (1.0 + research_strategy_returns).prod() - 1.0
).sort_values(ascending=False)
top_two_cumulative_names = research_cumulative_ranking.head(2).index.tolist()
SIMULATED_STRESS_STRATEGIES = CORE_COMPARISON_STRATEGIES.copy()

stress_strategy_selection = pd.DataFrame({
    'Policy': SIMULATED_STRESS_STRATEGIES,
    'Cumulative return ranking for the research period': [
        int(research_cumulative_ranking.index.get_loc(name) + 1)
        for name in SIMULATED_STRESS_STRATEGIES
    ],
    'Cumulative rate of return for the research period': [
        float(research_cumulative_ranking.loc[name])
        for name in SIMULATED_STRESS_STRATEGIES
    ],
    'Reason for inclusion': [
        'Core risk budget strategy' if name == FULLY_DYNAMIC_BUDGET_NAME
        else ('Fixed attack sleeves' if name == CORE_ATTACK_NAME
              else ('Dynamic attack sleeve + 20-day confirmation (final)' if name == DYNAMIC_ATTACK_H20_NAME
                    else 'Dynamic attack sleeves.'))
        for name in SIMULATED_STRESS_STRATEGIES
    ],
}).set_index('Policy')
display(stress_strategy_selection.style.format({'Cumulative rate of return for the research period': '{:.2%}'}))


def build_hypothetical_shock(category_shocks, overrides=None):
    shock = assets['Asset Class'].map(category_shocks).astype(float)
    if overrides:
        for ticker, value in overrides.items():
            shock.loc[ticker] = float(value)
    if shock.isna().any():
        raise RuntimeError(f'Simulation shock does not cover assets:{shock.index[shock.isna()].tolist()}')
    if (shock <= -1.0).any():
        raise RuntimeError('Simulation shocks cannot be less than or equal to -100%.')
    return shock.reindex(ALL_ASSETS)


hypothetical_scenarios = {
    'Inflationary stock debt double kill': build_hypothetical_shock(
        {'Equities': -0.18, 'Bonds': -0.12, 'Commodities': 0.12, 'FX': 0.00, 'Crypto': -0.35},
        {'USO': 0.25, 'GLD': 0.10, 'UUP': 0.08, 'FXY': -0.06, 'FXE': -0.08,
         '511010.SS': -0.05},
    ),
    'Global recession and risk aversion': build_hypothetical_shock(
        {'Equities': -0.25, 'Bonds': 0.07, 'Commodities': -0.18, 'FX': 0.00, 'Crypto': -0.50},
        {'USO': -0.28, 'GLD': 0.06, 'UUP': 0.09, 'FXY': 0.05, 'FXE': -0.06,
         '511010.SS': 0.03},
    ),
    'Global liquidity crisis': build_hypothetical_shock(
        {'Equities': -0.30, 'Bonds': -0.06, 'Commodities': -0.22, 'FX': -0.04, 'Crypto': -0.55},
        {'GLD': -0.10, 'UUP': 0.10, 'FXY': 0.03, 'FXE': -0.10},
    ),
    'Asset and exchange rate shocks in Asia': build_hypothetical_shock(
        {'Equities': -0.10, 'Bonds': -0.03, 'Commodities': -0.08, 'FX': -0.03, 'Crypto': -0.25},
        {'1308.T': -0.20, '510300.SS': -0.22, '2800.HK': -0.25,
         'ES3.SI': -0.18, 'FXY': -0.12, 'UUP': 0.08, 'GLD': 0.04},
    ),
    'Rapid post-crisis risk retreat': build_hypothetical_shock(
        {'Equities': 0.20, 'Bonds': -0.05, 'Commodities': 0.12, 'FX': -0.02, 'Crypto': 0.40},
        {'GLD': 0.00, 'UUP': -0.06, 'FXY': -0.04, 'FXE': 0.05},
    ),
}
hypothetical_shock_table = pd.DataFrame(hypothetical_scenarios).T
display(hypothetical_shock_table.style.format('{:.1%}'))

hypothetical_stress_records = []
hypothetical_contribution_records = []
for strategy_name in SIMULATED_STRESS_STRATEGIES:
    result = results[strategy_name]
    if 'end_of_day_weights' not in result or result['end_of_day_weights'].empty:
        raise RuntimeError(f'{strategy_name}Lack of real hold at the end of the study.')
    latest_weights = (
        result['end_of_day_weights'].loc[:RESEARCH_END].iloc[-1]
        .reindex(ALL_ASSETS).fillna(0.0).astype(float)
    )
    cash_weight = 1.0 - float(latest_weights.sum())
    latest_rf_daily = float(result['risk_free_daily'].loc[:RESEARCH_END].iloc[-21:].mean())
    cash_month_return = (1.0 + latest_rf_daily) ** 21 - 1.0

    for scenario_name, shock in hypothetical_scenarios.items():
        asset_contributions = latest_weights * shock
        cash_contribution = cash_weight * cash_month_return
        portfolio_shock_return = float(asset_contributions.sum() + cash_contribution)
        worst_asset = asset_contributions.idxmin()
        hypothetical_stress_records.append({
            'Simulation': scenario_name,
            'Policy': strategy_name,
            'Warehousing record date': result['end_of_day_weights'].loc[:RESEARCH_END].index[-1].date(),
            'Total asset weights': float(latest_weights.sum()),
            'Cash weights': cash_weight,
            'Combination of one month of impact gains': portfolio_shock_return,
            'Maximum negative contribution assets': worst_asset,
            'Maximum negative contribution': float(asset_contributions.loc[worst_asset]),
        })
        for ticker, contribution in asset_contributions.items():
            hypothetical_contribution_records.append({
                'Simulation': scenario_name, 'Policy': strategy_name,
                'Code': ticker, 'Contribution of shocks': float(contribution),
            })

hypothetical_stress_test = pd.DataFrame(hypothetical_stress_records).set_index(['Simulation', 'Policy'])
hypothetical_stress_contributions = pd.DataFrame(hypothetical_contribution_records)
display(hypothetical_stress_test.style.format({
    'Total asset weights': '{:.2%}', 'Cash weights': '{:.2%}',
    'Combination of one month of impact gains': '{:.2%}', 'Maximum negative contribution': '{:.2%}',
}))

hypothetical_plot = hypothetical_stress_test['Combination of one month of impact gains'].unstack('Policy')
hypothetical_plot = hypothetical_plot.reindex(columns=SIMULATED_STRESS_STRATEGIES)
hypothetical_plot = hypothetical_plot.rename(columns=strategy_label)
hypothetical_plot = hypothetical_plot.rename(index={
    'Asset and exchange rate shocks in Asia': 'Asian assets against \n exchange rate shocks',
    'Global liquidity crisis': 'Global liquidity \n crisis',
    'Global recession and risk aversion': 'Global recession and risk aversion of \n',
    'Rapid post-crisis risk retreat': 'Rapid \n risk rebound after crisis',
    'Inflationary stock debt double kill': 'Inflationary \n stock debt double kill',
})
stress_colors = [STRATEGY_COLORS[name] for name in SIMULATED_STRESS_STRATEGIES]
ax = hypothetical_plot.plot(kind='bar', figsize=(15, 8.2), width=0.66, color=stress_colors)
ax.axhline(0.0, color='black', linewidth=0.8)
ax.set_title('Assumptions of combined impact results (current recorded weights) for the four core comparators')
ax.set_xlabel('Simulation')
ax.set_ylabel('Combination of one month of impact gains')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.tick_params(axis='x', rotation=0, labelsize=9, pad=10)
ax.margins(y=0.10)
style_axis(ax, axis='y')
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    loc='upper center', bbox_to_anchor=(0.5, -0.18),
    ncol=3, fontsize=9, frameon=False,
)
ax.xaxis.set_label_coords(0.5, -0.31)
for container in ax.containers:
    ax.bar_label(container, labels=[f'{value:.2%}' for value in container.datavalues],
                 padding=3, fontsize=8)
plt.tight_layout(rect=[0, 0.30, 1, 1])
plt.show()

# 5.10 Bootstream Monte Carlo pressure test
#
# This section draws blocks of 20 consecutive trading days from the net proceeds of the strategy up to 2026-07-31, generating 1,000 simulating paths per 252 trading days. A continuous block allows for a partial accumulation of associated and volatile short-term benefits.
#
# -** Sample of general blocks**: all historical blocks are subject to the same probability;
# -** Crisis-weighted sample**: Blocks containing cross-asset market agents with a date of less than 10 per cent of history increase their extraction weights to four times that of ordinary blocks.
#
# The three comparators use the same sample date blocks in each path and preserve the same period of relationship between the strategies. The test is reordered with the net gains from the strategies achieved and therefore includes historical reconciliations and cost results of the original strategy, but does not solve Top10 and EWMA weights on the synthetic price path; it is a distributed pressure test, not a probabilistic projection for the future.
# --- Original Notebook code unit 65-
BOOTSTRAP_SEED = 202609
BOOTSTRAP_PATHS = 1000
BOOTSTRAP_HORIZON = 252
BOOTSTRAP_BLOCK = 20
CRISIS_BLOCK_MULTIPLIER = 4.0

bootstrap_returns = strategy_returns.loc[:RESEARCH_END, SIMULATED_STRESS_STRATEGIES].dropna()
if bootstrap_returns.index.max() >= HOLDOUT_START:
    raise RuntimeError('The Bootstream pressure test read the lock test set.')
if len(bootstrap_returns) < BOOTSTRAP_HORIZON + BOOTSTRAP_BLOCK:
    raise RuntimeError('The historical sample was insufficient to implement block Bootstream.')

market_proxy = returns[ALL_ASSETS].mean(axis=1).reindex(bootstrap_returns.index)
candidate_starts = np.arange(0, len(bootstrap_returns) - BOOTSTRAP_BLOCK + 1)
crisis_threshold = float(market_proxy.quantile(0.10))
crisis_blocks = np.array([
    bool((market_proxy.iloc[start:start + BOOTSTRAP_BLOCK] <= crisis_threshold).any())
    for start in candidate_starts
])
uniform_probability = np.full(len(candidate_starts), 1.0 / len(candidate_starts))
crisis_probability = np.where(crisis_blocks, CRISIS_BLOCK_MULTIPLIER, 1.0).astype(float)
crisis_probability /= crisis_probability.sum()


def block_bootstrap_paths(return_frame, start_probability, n_paths, horizon, block_length, seed):
    rng = np.random.default_rng(seed)
    values = return_frame.to_numpy(dtype=float)
    starts = np.arange(0, len(return_frame) - block_length + 1)
    simulations = np.empty((n_paths, horizon, values.shape[1]), dtype=float)
    blocks_needed = int(np.ceil(horizon / block_length))
    for path_number in range(n_paths):
        chosen = rng.choice(starts, size=blocks_needed, replace=True, p=start_probability)
        path = np.concatenate([values[s:s + block_length] for s in chosen], axis=0)[:horizon]
        simulations[path_number] = path
    return simulations


def summarize_bootstrap_paths(simulations, strategy_names, sampling_mode):
    cumulative_wealth = np.cumprod(1.0 + simulations, axis=1)
    terminal_returns = cumulative_wealth[:, -1, :] - 1.0
    running_peaks = np.maximum.accumulate(np.concatenate([
        np.ones((simulations.shape[0], 1, simulations.shape[2])), cumulative_wealth
    ], axis=1), axis=1)[:, 1:, :]
    max_drawdowns = (cumulative_wealth / running_peaks - 1.0).min(axis=1)
    records = []
    path_records = []
    for column_number, strategy_name in enumerate(strategy_names):
        terminal = terminal_returns[:, column_number]
        drawdown = max_drawdowns[:, column_number]
        records.append({
            'Sample method': sampling_mode,
            'Policy': strategy_name,
            '1 per cent end-of-cycle gains': float(np.quantile(terminal, 0.01)),
            '5% end-of-cycle earnings': float(np.quantile(terminal, 0.05)),
            'Middle income at end of period': float(np.median(terminal)),
            'Probability of loss': float((terminal < 0.0).mean()),
            'Maximum retreat median': float(np.median(drawdown)),
            'Maximum exit 5%.': float(np.quantile(drawdown, 0.05)),
            'Over 20% chance of retreat.': float((drawdown <= -0.20).mean()),
        })
        path_records.extend({
            'Sample method': sampling_mode, 'Policy': strategy_name,
            'Path Number': path_number, 'End of period rate of return': float(terminal[path_number]),
            'Maximum Drawdown': float(drawdown[path_number]),
        } for path_number in range(len(terminal)))
    return records, path_records


bootstrap_summary_records, bootstrap_path_records = [], []
for mode_number, (sampling_mode, probabilities) in enumerate([
    ('Sample of general blocks', uniform_probability),
    ('Weighted sample of crisis blocks', crisis_probability),
]):
    simulated_paths = block_bootstrap_paths(
        bootstrap_returns, probabilities, BOOTSTRAP_PATHS,
        BOOTSTRAP_HORIZON, BOOTSTRAP_BLOCK,
        BOOTSTRAP_SEED + mode_number,
    )
    summary_rows, path_rows = summarize_bootstrap_paths(
        simulated_paths, bootstrap_returns.columns.tolist(), sampling_mode
    )
    bootstrap_summary_records.extend(summary_rows)
    bootstrap_path_records.extend(path_rows)

bootstrap_stress_summary = pd.DataFrame(bootstrap_summary_records).set_index(['Sample method', 'Policy'])
bootstrap_stress_paths = pd.DataFrame(bootstrap_path_records)
display(bootstrap_stress_summary.style.format({
    '1 per cent end-of-cycle gains': '{:.2%}', '5% end-of-cycle earnings': '{:.2%}',
    'Middle income at end of period': '{:.2%}', 'Probability of loss': '{:.2%}',
    'Maximum retreat median': '{:.2%}', 'Maximum exit 5%.': '{:.2%}',
    'Over 20% chance of retreat.': '{:.2%}',
}))

for metric, title in [
    ('End of period rate of return', 'Distribution of income at the end of Bootstream as at 252'),
    ('Maximum Drawdown', 'Bootstream maximum retreat distribution on 252'),
]:
    fig, axis = plt.subplots(figsize=(13, 6))
    plot_groups, labels = [], []
    for sampling_mode in ['Sample of general blocks', 'Weighted sample of crisis blocks']:
        for strategy_name in SIMULATED_STRESS_STRATEGIES:
            values = bootstrap_stress_paths.loc[
                (bootstrap_stress_paths['Sample method'] == sampling_mode)
                & (bootstrap_stress_paths['Policy'] == strategy_name), metric
            ].to_numpy()
            plot_groups.append(values)
            labels.append(f'{sampling_mode} / {strategy_label(strategy_name)}')
    # Matpllotlib 3.9+ Renames ``labels`` as ``tick labels``.
    # Use new parameters to ensure that Notebook can be implemented from scratch in the current course environment.
    axis.boxplot(plot_groups, tick_labels=labels, showfliers=False)
    axis.axhline(0.0, color='black', linewidth=0.8)
    axis.set_title(title)
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
    axis.tick_params(axis='x', rotation=70, labelsize=7)
    style_axis(axis, axis='y')
    fig.tight_layout()
    plt.show()

print(
    f'Bootstream only{bootstrap_returns.index.min().date()}to'
    f'{bootstrap_returns.index.max().date()}Net gains during the research period;'
    f'Simulation path ={BOOTSTRAP_PATHS}, path length ={BOOTSTRAP_HORIZON}I don\'t know.'
    f'Length of blocks ={BOOTSTRAP_BLOCK}Day.'
)

# 5.11 VaR and ES: historical simulation and bottom risks of Bootstream
#
# This section continues to use the three comparative strategies described above and reports on two types of tail risks:
#
# 1. **History Simulation 1-day VaR/ES**: direct use of daily net gains from strategy up to 2026-07-31 calculated at 95 per cent and 99 per cent confidence levels respectively;
# 2. **VaR/ES** for the one-year period of Bootstream: 95 per cent and 99 per cent end-of-cycle loss is calculated using 1,000 items in the previous section, 252 general blocks on trading days and crisis weighted paths.
#
# VaR states that the threshold of loss will not normally be exceeded at the given level of confidence; ES indicates the average loss of the worst tail sample after the loss exceeds VaR. In the table, both VaR and ES are shown as losses in positive numbers, while the monetary amounts are converted as assumed ** SGD 1,000,000.
# --- Original Notebook code unit 67-
VAR_ES_CAPITAL_SGD = 1_000_000.0
VAR_ES_LEVELS = [0.95, 0.99]


def empirical_var_es(return_values, confidence_level):
    values = pd.Series(return_values, dtype=float).dropna().to_numpy()
    if len(values) < 30:
        raise RuntimeError('There are less than 30 VAR/ES effective observations.')
    losses = -values
    value_at_risk = float(np.quantile(losses, confidence_level))
    tail_losses = losses[losses >= value_at_risk]
    expected_shortfall = float(tail_losses.mean())
    return value_at_risk, expected_shortfall, len(values), len(tail_losses)


# I. Historical simulation 1 day VAR and ES
historical_var_es_records = []
for strategy_name in SIMULATED_STRESS_STRATEGIES:
    daily_net_returns = strategy_returns.loc[:RESEARCH_END, strategy_name].dropna()
    if daily_net_returns.index.max() >= HOLDOUT_START:
        raise RuntimeError('History VAR/ES reads the lock test set.')
    for confidence_level in VAR_ES_LEVELS:
        var_value, es_value, observations, tail_count = empirical_var_es(
            daily_net_returns, confidence_level
        )
        historical_var_es_records.append({
            'Policy': strategy_name,
            'Trust Horizontal': confidence_level,
            'Observation': observations,
            'End observations': tail_count,
            '1 VaR': var_value,
            '1 January ES': es_value,
            'VAR Amount (ZXQ0QZ)': var_value * VAR_ES_CAPITAL_SGD,
            'ES Amount (SGD)': es_value * VAR_ES_CAPITAL_SGD,
        })

historical_var_es = pd.DataFrame(historical_var_es_records).set_index(['Policy', 'Trust Horizontal'])
display(historical_var_es.style.format({
    '1 VaR': '{:.2%}', '1 January ES': '{:.2%}',
    'VAR Amount (ZXQ0QZ)': 'SGD {:,.2f}', 'ES Amount (SGD)': 'SGD {:,.2f}',
}))


# VAR and ES of the cumulative distribution of proceeds as at 252
bootstrap_var_es_records = []
for sampling_mode in ['Sample of general blocks', 'Weighted sample of crisis blocks']:
    for strategy_name in SIMULATED_STRESS_STRATEGIES:
        terminal_returns = bootstrap_stress_paths.loc[
            (bootstrap_stress_paths['Sample method'] == sampling_mode)
            & (bootstrap_stress_paths['Policy'] == strategy_name),
            'End of period rate of return',
        ]
        for confidence_level in VAR_ES_LEVELS:
            var_value, es_value, observations, tail_count = empirical_var_es(
                terminal_returns, confidence_level
            )
            bootstrap_var_es_records.append({
                'Sample method': sampling_mode,
                'Policy': strategy_name,
                'Trust Horizontal': confidence_level,
                'Simulation of paths': observations,
                'Number of tail paths': tail_count,
                '252 VaR': var_value,
                '252 ES': es_value,
                'VAR Amount (ZXQ0QZ)': var_value * VAR_ES_CAPITAL_SGD,
                'ES Amount (SGD)': es_value * VAR_ES_CAPITAL_SGD,
            })

bootstrap_var_es = pd.DataFrame(bootstrap_var_es_records).set_index(
    ['Sample method', 'Policy', 'Trust Horizontal']
)
display(bootstrap_var_es.style.format({
    '252 VaR': '{:.2%}', '252 ES': '{:.2%}',
    'VAR Amount (ZXQ0QZ)': 'SGD {:,.2f}', 'ES Amount (SGD)': 'SGD {:,.2f}',
}))


# 3. End risk comparison: The lower the value, the smaller the tail loss.
plot_table = historical_var_es.reset_index()
plot_table = plot_table[plot_table['Trust Horizontal'] == 0.99].set_index('Policy')
fig, ax = plt.subplots(figsize=(13, 5.5))
x = np.arange(len(plot_table))
width = 0.36
ax.bar(x - width / 2, plot_table['1 VaR'], width, label='99% 1 day VaR', color='#4C78A8')
ax.bar(x + width / 2, plot_table['1 January ES'], width, label='99% 1 day ES', color='#E45756')
ax.set_xticks(x)
ax.set_xticklabels([strategy_label(name) for name in plot_table.index], rotation=18, ha='right')
ax.set_title('Focus Policy History Simulation 99% a day VAR and ES')
ax.set_ylabel('Loss rate (positive)')
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
ax.legend()
plt.tight_layout()
plt.show()

print(
    f'VaR/ES only as of{RESEARCH_END.date()}Research data;'
    f'Currency amount according to SGD{VAR_ES_CAPITAL_SGD:,.2f}The principal amount is converted.'
)

