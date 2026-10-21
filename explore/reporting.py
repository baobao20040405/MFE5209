"""Performance metrics, holdings diagnostics, and core visualizations.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# Part IV System of indicators and combination diagnosis
#
# 4.1 Harmonized indicator function
#
#
# This section defines only harmonized performance indicators and evaluation areas. When all 12 formal policy definitions are completed, the table of indicators and visualization are centralized in Part VI to avoid different versions of the policy set being read in different cells.
# --- Original Notebook code module 35-
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


periods = {
    'In Sample (through 2024-12-31)': (None, IN_SAMPLE_END),
    'Validation (2025-01-01 to 2026-07-31)': (OOS_START, RESEARCH_END),
    'Full Research Period (excluding holdout)': (None, RESEARCH_END),
}

# 4.2 Core Dynamics Strategy Updates to Warden Diagnostics
#
#
# This section shows only the state-of-the-art of the core strategy that needs to be explained in terms of momentum screening, leverage or dynamic risk budgeting, and does not assume the "total strategy aggregation" function. For a harmonized performance comparison of all 12 strategies, see the harmonized performance summary in Part VI.
# --- Original Notebook code unit 37-
CORE_COMPARISON_STRATEGIES = [
    FULLY_DYNAMIC_BUDGET_NAME,
    CORE_ATTACK_NAME,
    DYNAMIC_ATTACK_NAME,
    DYNAMIC_ATTACK_H20_NAME,
]
if len(CORE_COMPARISON_STRATEGIES) != 4 or len(set(CORE_COMPARISON_STRATEGIES)) != 4:
    raise RuntimeError('The core group must contain exactly four elements without repetition.')
missing_core_comparison = [name for name in CORE_COMPARISON_STRATEGIES if name not in results]
if missing_core_comparison:
    raise RuntimeError(f'The core comparison strategy is not yet registered:{missing_core_comparison}')

latest_tables = []
latest_strategy_names = [
    STRATEGY_NAMES['inverse_vol'], STRATEGY_NAMES['classic_erc'],
    STRATEGY_NAMES['ewma_erc'], STRATEGY_NAMES['downside_erc'],
    DYNAMIC_BUDGET_NAME, HIERARCHICAL_BUDGET_NAME, GARCH_MOMENTUM_NAME,
    *CORE_COMPARISON_STRATEGIES,
]
latest_strategy_names = list(dict.fromkeys(latest_strategy_names))

for name in latest_strategy_names:
    obj = results[name]
    if 'end_of_day_weights' not in obj or obj['end_of_day_weights'].empty:
        raise RuntimeError(f'{name}Lack of real hold at the end of the day.')
    last_date = obj['end_of_day_weights'].loc[:RESEARCH_END].index[-1]
    latest_weights = obj['end_of_day_weights'].loc[last_date].reindex(ALL_ASSETS).fillna(0.0)
    latest_weights = latest_weights[latest_weights.abs() > 1e-10]

    momentum_lookup = pd.Series(dtype=float)
    selections = obj.get('selections', pd.DataFrame())
    if not selections.empty and {'Rebalance Date', 'Code', 'Motion Score'}.issubset(selections.columns):
        selection_dates = pd.to_datetime(selections['Rebalance Date'])
        eligible_dates = selection_dates.loc[selection_dates <= pd.Timestamp(last_date)]
        latest_selection = selections.loc[selection_dates == eligible_dates.max()] if not eligible_dates.empty else pd.DataFrame()
        if not latest_selection.empty:
            momentum_lookup = latest_selection.drop_duplicates('Code').set_index('Code')['Motion Score']

    latest = pd.DataFrame({
        'Policy': name,
        'Code': latest_weights.index,
        'Asset Name': [asset_metadata.loc[ticker, 'Asset Name'] for ticker in latest_weights.index],
        'Motion Score': [momentum_lookup.get(ticker, np.nan) for ticker in latest_weights.index],
        'Real weight at end of period': latest_weights.values,
        'Warehousing record date': last_date,
    })
    latest['Motion Score'] = latest['Motion Score'].map(
        lambda value: '—' if pd.isna(value) else f'{value:.2%}'
    )
    latest['Real weight at end of period'] = latest['Real weight at end of period'].map(lambda value: f'{value:.2%}')
    latest_tables.append(latest)

display(pd.concat(latest_tables, ignore_index=True))

# Broad asset weights for each official strategy: a strategy-by-strategy map, uniform category colour and vertical calibration.
def build_class_weight_history(result):
    source = result['end_of_day_weights'].loc[strategy_start:RESEARCH_END].copy()
    class_weights = pd.DataFrame(index=source.index)
    for asset_class in asset_metadata['Asset Class'].dropna().unique():
        members = [ticker for ticker in source.columns
                   if ticker in asset_metadata.index
                   and asset_metadata.loc[ticker, 'Asset Class'] == asset_class]
        if members:
            class_weights[asset_class] = source[members].sum(axis=1)
    recorded_cash = result.get('cash_weight', pd.Series(dtype=float)).reindex(source.index)
    class_weights['Cash'] = recorded_cash.fillna(1.0 - source.sum(axis=1)).clip(lower=0.0)
    return class_weights.resample('ME').last().dropna(how='all')


strategy_class_weight_histories = {}
for strategy_name, strategy_result in results.items():
    class_history = build_class_weight_history(strategy_result)
    strategy_class_weight_histories[strategy_name] = class_history
    active_columns = [column for column in ASSET_CLASS_COLORS
                      if column in class_history and class_history[column].abs().max() > 1e-10]
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.stackplot(
        class_history.index,
        *[class_history[column].to_numpy(dtype=float) for column in active_columns],
        labels=active_columns,
        colors=[ASSET_CLASS_COLORS[column] for column in active_columns],
        alpha=0.88,
    )
    ax.set_title(f'{strategy_label(strategy_name)}: Month-End Asset-Class Weights',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Weight')
    ax.set_ylim(0.0, max(1.02, float(class_history[active_columns].sum(axis=1).max()) * 1.03))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
    style_axis(ax, axis='y')
    place_legend(ax, item_count=len(active_columns))
    fig.tight_layout(rect=[0, 0.12 if len(active_columns) > 4 else 0, 1, 1])
    plt.show()

# 4.3 Formal Strategy Cumulative Gain Mapping Function
# --- Original Notebook code unit 39-
BASELINE_STRATEGY = STRATEGY_NAMES['equal_weight']
FOCUS_STRATEGIES = [
    DYNAMIC_BUDGET_NAME, *CORE_COMPARISON_STRATEGIES,
    STRATEGY_NAMES['ewma_erc'], STATIC_CLASSIC_NAME, BASELINE_STRATEGY,
]
if len(FOCUS_STRATEGIES) != 8 or len(set(FOCUS_STRATEGIES)) != 8:
    raise RuntimeError('The priority strategy list must contain eight items and should not be repeated.')
missing_focus = [name for name in FOCUS_STRATEGIES if name not in results]
if missing_focus:
    raise RuntimeError(f'Focus strategy not yet registered:{missing_focus}')

HIGHLIGHT_STRATEGIES = FOCUS_STRATEGIES.copy()
HIGHLIGHT_COLORS = STRATEGY_COLORS.copy()
MUTED_COLOR = '#DCE2E8'


def plot_all_strategy_cumulative(strategy_return_frame):
    cumulative = (1.0 + strategy_return_frame).cumprod() - 1.0
    strategy_names = cumulative.columns.tolist()
    color_map = {
        name: HIGHLIGHT_COLORS.get(name, MUTED_COLOR)
        for name in strategy_names
    }

    fig, ax = plt.subplots(figsize=(17, 9))
    # Draw unfocused strategies to ensure that the eight focus curves are not shielded.
    for strategy_name in [name for name in strategy_names if name not in FOCUS_STRATEGIES]:
        ax.plot(
            cumulative.index, cumulative[strategy_name],
            color=MUTED_COLOR, linewidth=1.0, linestyle='-', alpha=0.34,
            zorder=1, label=strategy_label(strategy_name),
        )
    for strategy_name in FOCUS_STRATEGIES:
        is_baseline = strategy_name == BASELINE_STRATEGY
        is_submission = strategy_name == SUBMISSION_STRATEGY
        ax.plot(
            cumulative.index, cumulative[strategy_name],
            color=HIGHLIGHT_COLORS.get(strategy_name, '#4C78A8'),
            linewidth=4.0 if is_submission else (3.4 if is_baseline else 2.6),
            linestyle='-',
            alpha=1.0, zorder=9 if is_submission else (8 if is_baseline else 5),
            label=(f'{strategy_label(strategy_name)}(baseline)' if is_baseline
                   else strategy_label(strategy_name)),
        )

    # All cumulative income maps are superseded with ES3.SI and VT, using the same starting point and the same SDD total return.
    benchmark_frame = benchmark_returns_sgd[['ES3.SI', 'VT']].reindex(
        strategy_return_frame.index
    ).dropna(how='any')
    benchmark_cumulative = (1.0 + benchmark_frame).cumprod() - 1.0
    benchmark_styles = {
        'ES3.SI': {'color': MARKET_BENCHMARK_COLORS['ES3.SI'], 'linestyle': '-'},
        'VT': {'color': MARKET_BENCHMARK_COLORS['VT'], 'linestyle': '-'},
    }
    for ticker, label in MARKET_BENCHMARK_NAMES.items():
        ax.plot(
            benchmark_cumulative.index, benchmark_cumulative[ticker],
            linewidth=2.6, alpha=1.0, zorder=6,
            label=f'Market Benchmark: {label}', **benchmark_styles[ticker],
        )
    ax.axvline(IN_SAMPLE_END, color='#AAB2BD', linestyle='-', linewidth=1.0,
               label='In-Sample End')
    ax.axhline(0.0, color='#555555', linewidth=0.8)
    ax.set_title(
        f'Cumulative Net Returns: {len(strategy_names)} Strategies and Two Market Benchmarks (SGD)',
        fontsize=15, fontweight='bold'
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
    style_axis(ax)
    place_legend(ax)
    plt.tight_layout(rect=[0, 0.16, 1, 1])
    plt.show()
    return cumulative, color_map

# 4.4 Formal policy retreat drawing function
# --- Original Notebook code module 41--
def plot_all_strategy_drawdowns(strategy_return_frame, color_map):
    """Plot filled drawdowns for the final strategy and benchmark, with supporting strategies."""
    drawdowns = strategy_return_frame.apply(drawdown_series)
    fig, ax = plt.subplots(figsize=(17, 9))

    for strategy_name in drawdowns.columns:
        is_focus = strategy_name in FOCUS_STRATEGIES
        is_submission = strategy_name == SUBMISSION_STRATEGY
        is_base = strategy_name == BASELINE_STRATEGY
        color = color_map[strategy_name]
        if is_submission or is_base:
            fill_color = color if is_submission else '#8C9299'
            fill_alpha = 0.52 if is_submission else 0.22
            ax.fill_between(
                drawdowns.index, drawdowns[strategy_name].to_numpy(dtype=float), 0.0,
                color=fill_color, alpha=fill_alpha, linewidth=0, edgecolor='none',
                label=strategy_label(strategy_name), zorder=4 if is_submission else 1,
            )
        else:
            ax.plot(
                drawdowns.index, drawdowns[strategy_name], color=color,
                linewidth=1.5 if is_focus else 0.75,
                alpha=0.72 if is_focus else 0.16,
                label=strategy_label(strategy_name), zorder=3 if is_focus else 2,
            )

    ax.axvline(IN_SAMPLE_END, color='#AAB2BD', linestyle='-', linewidth=1.0,
               label='In-Sample End')
    ax.axhline(0.0, color='#555555', linewidth=0.8)
    ax.set_title(f'Drawdown Curves for All {len(drawdowns.columns)} Formal Strategies',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
    style_axis(ax)
    place_legend(ax)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    plt.show()
    return drawdowns

# 4.5 Independent realistic cost scenarios: final strategy and market benchmarks
#
# The official master return continues to be one unilateral 0.05 per cent. This section changes the final strategy only to submarket commissions,
# Exchanges of currencies, market price differentials, dynamic market shocks and withholding tax revaluation; rights benchmark maintained
# Uniform unilateral 0.05 per cent, compared with SSD-denominated VT and ES3.SI. This scenario does not cover results.
# --- Original Notebook code module 43-
REALISTIC_SCENARIO_NAMES = {
    'fixed_submission': f'{strategy_label(SUBMISSION_STRATEGY)} (Fixed 0.05%)',
    'realistic_submission': f'{strategy_label(SUBMISSION_STRATEGY)} (Realistic Costs)',
    'fixed_equal': f'{strategy_label(BASELINE_STRATEGY)} (Fixed 0.05%)',
    'VT': 'VT Global Equities (SGD)',
    'ES3.SI': 'ES3.SI Straits Times Index (SGD)',
}

realistic_cost_comparison_returns = pd.concat({
    REALISTIC_SCENARIO_NAMES['fixed_submission']:
        results[SUBMISSION_STRATEGY]['returns'],
    REALISTIC_SCENARIO_NAMES['realistic_submission']:
        realistic_submission_result['returns'],
    REALISTIC_SCENARIO_NAMES['fixed_equal']:
        results[BASELINE_STRATEGY]['returns'],
    REALISTIC_SCENARIO_NAMES['VT']:
        benchmark_returns_sgd['VT'],
    REALISTIC_SCENARIO_NAMES['ES3.SI']:
        benchmark_returns_sgd['ES3.SI'],
}, axis=1).loc[strategy_start:RESEARCH_END].dropna(how='any')

realistic_cost_rf = results[SUBMISSION_STRATEGY]['risk_free_daily'].reindex(
    realistic_cost_comparison_returns.index
).ffill()
if realistic_cost_rf.isna().any():
    raise RuntimeError('The SORA coverage of the real-cost stand-alone scenario is incomplete.')


def compact_return_metrics(return_series, risk_free_series):
    """Generate streamlined and comparable indicators of the same calibre for strategic and non-tradable indicators."""
    aligned = pd.concat([return_series, risk_free_series], axis=1).dropna()
    r = aligned.iloc[:, 0].astype(float)
    rf = aligned.iloc[:, 1].astype(float)
    years = len(r) / TRADING_DAYS
    cumulative = float((1.0 + r).prod() - 1.0)
    annual_return = float((1.0 + cumulative) ** (1.0 / years) - 1.0)
    annual_volatility = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    excess = r - rf
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))
    maximum_drawdown = float(drawdown_series(r).min())
    return pd.Series({
        'Cumulative Return': cumulative,
        'Annualized Return': annual_return,
        'Annualized Volatility': annual_volatility,
        'Sharpe Ratio': sharpe,
        'Maximum Drawdown': maximum_drawdown,
    })


realistic_cost_comparison_table = pd.DataFrame({
    name: compact_return_metrics(series, realistic_cost_rf)
    for name, series in realistic_cost_comparison_returns.items()
}).T.sort_values('Sharpe Ratio', ascending=False)
display(realistic_cost_comparison_table.style.format({
    'Cumulative Return': '{:.2%}', 'Annualized Return': '{:.2%}',
    'Annualized Volatility': '{:.2%}', 'Sharpe Ratio': '{:.2f}', 'Maximum Drawdown': '{:.2%}',
}))

realistic_cost_colors = {
    REALISTIC_SCENARIO_NAMES['fixed_submission']: STRATEGY_COLORS[SUBMISSION_STRATEGY],
    REALISTIC_SCENARIO_NAMES['realistic_submission']: '#C85A6A',
    REALISTIC_SCENARIO_NAMES['fixed_equal']: STRATEGY_COLORS[BASELINE_STRATEGY],
    REALISTIC_SCENARIO_NAMES['VT']: MARKET_BENCHMARK_COLORS['VT'],
    REALISTIC_SCENARIO_NAMES['ES3.SI']: MARKET_BENCHMARK_COLORS['ES3.SI'],
}
realistic_cost_cumulative = (1.0 + realistic_cost_comparison_returns).cumprod() - 1.0
fig, ax = plt.subplots(figsize=(14, 7))
for name in realistic_cost_cumulative.columns:
    is_fixed_submission = name == REALISTIC_SCENARIO_NAMES['fixed_submission']
    is_realistic_submission = name == REALISTIC_SCENARIO_NAMES['realistic_submission']
    ax.plot(
        realistic_cost_cumulative.index,
        realistic_cost_cumulative[name],
        color=realistic_cost_colors[name],
        linewidth=3.4 if is_fixed_submission else (3.0 if is_realistic_submission else 2.2),
        linestyle='-',
        label=name,
        zorder=5 if is_fixed_submission else (4 if is_realistic_submission else 2),
    )
ax.axhline(0.0, color='#555555', linewidth=0.8)
ax.axvline(IN_SAMPLE_END, color='#AAB2BD', linewidth=1.0, label='In-Sample End')
ax.set_title('Realistic-Cost Scenario: Final Strategy, Equal Weight, and Equity Benchmarks')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2%}'))
style_axis(ax)
place_legend(ax, item_count=6)
fig.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()

assert 'cost_model' not in results[SUBMISSION_STRATEGY], 'The formal results were unexpectedly covered by realistic cost scenarios.'
assert realistic_submission_result['cost_model'] == 'Real cost scenarios for independent sub-markets'
print('The cost of real differences is exported only as a stand-alone scenario; formal results still use a unified unilateral of 0.05 per cent.')

