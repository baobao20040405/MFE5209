"""Structural benchmarks, parameter sensitivity, digestion and Walker-Forward testing.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# Part VII. Validity, digestion and modelling
#
# 7.1 Structural benchmarks and counter-fact comparisons
#
#
# The official internal Benchmark is ** the global asset-equity benchmark**;** the global asset-class balance** serves as the structural benchmark that does not depend on a dynamic screening. The formal set of strategies is strictly set at 12.
# --- Original Notebook code unit 79-
STRUCTURE_BENCHMARKS = [
    BASELINE_STRATEGY, STATIC_CLASSIC_NAME,
    STRATEGY_NAMES['ewma_erc'], *CORE_COMPARISON_STRATEGIES,
]
structure_benchmark_metrics = pd.DataFrame({
    name: compute_metrics(results[name], strategy_start, RESEARCH_END)
    for name in STRUCTURE_BENCHMARKS
}).T.sort_values('Sharpe Ratio', ascending=False)
display(structure_benchmark_metrics[[
    'Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Maximum Drawdown', 'Annual One-Way Turnover'
]].style.format({
    'Cumulative Return': '{:.2%}', 'Annualized Return': '{:.2%}', 'Annualized Volatility': '{:.2%}',
    'Sharpe Ratio': '{:.2f}', 'Maximum Drawdown': '{:.2%}', 'Annual One-Way Turnover': '{:.2%}',
}))

# 7.2 BTC of the core EWMA model melts with common samples
#
#
# In order to avoid mixing model changes with sample changes, this section first compares “bTC-containing” and “bTC-free” between identical study areas; then separate reports report the results of extended samples “without BTC, using earlier common starting points”. The extended sample is used only for robustness and does not directly compare the cumulative yield with the results after 2015.
#
# The following research functions are used in a uniform manner: 12-1 momentum, Top N filter, EWMA strict risk parities, closing date of the next transaction and underlying transaction costs. All historical windows are strictly closed to signal day closing, and new weights do not yield pre-offer returns.
#
# > This section uses a uniform simplified core model (Vehicular Screening + EWMA Strict ERC) to identify the marginal role of asset selection with the BTC; it does not contain dynamic double-tier budgets, dynamic offensives, semi-speed execution and no-trade zones for the final strategy, and therefore cannot directly replace the final strategy's digestion conclusions.
# --- Original Notebook code unit 81--
def run_research_ewma_model(
    return_frame, universe, start_date=None, end_date=RESEARCH_END,
    top_n=TOP_N, score_days=LOOKBACK - MOMENTUM_SKIP,
    decay=EWMA_LAMBDA, check_interval=CHECK_INTERVAL,
):
    """Model of freezing rules only for robustness tests; not registered in official results."""
    frame = return_frame.loc[:end_date, universe].dropna(how='any')
    minimum_history = max(LOOKBACK, score_days + MOMENTUM_SKIP)
    if start_date is None:
        start_date = frame.index[minimum_history]
    start_date = max(pd.Timestamp(start_date), frame.index[minimum_history])
    dates = frame.loc[start_date:end_date].index
    rf = align_sora_returns(dates, sora_daily_df)
    current = pd.Series(0.0, index=universe, dtype=float)
    pending = None
    daily_records, turnover_records, weight_records = [], [], []
    signal_records = []
    first_location = frame.index.get_loc(dates[0])

    for date in dates:
        location = frame.index.get_loc(date)
        day = frame.loc[date]
        cash_weight = 1.0 - float(current.sum())
        gross = float(current @ day) + cash_weight * float(rf.loc[date])
        if 1.0 + gross <= 0:
            raise RuntimeError('The robust retrospect combination is not positive.')
        current = current * (1.0 + day) / (1.0 + gross)

        cost_rate = 0.0
        if pending is not None:
            if not pending['Signal Date'] < date:
                raise RuntimeError('A robust retrospection of the presence signal is done by day.')
            target = pending['Target weights']
            turnover = float((target - current).abs().sum())
            cost_rate = COST * turnover
            current = target.copy()
            turnover_records.append((date, turnover))
            weight_records.append(current.rename(date))
            signal_records.append({'Signal Date': pending['Signal Date'], 'Execution Date': date})
            pending = None

        net = (1.0 - cost_rate) * (1.0 + gross) - 1.0
        daily_records.append((date, net))

        history = frame.iloc[location - LOOKBACK + 1:location + 1]
        if len(history) != LOOKBACK or history.index.max() != date:
            raise RuntimeError('A robust retrospect history window error.')
        if (location - first_location) % int(check_interval) == 0:
            scoring_window = history.iloc[-(score_days + MOMENTUM_SKIP):-MOMENTUM_SKIP]
            score = (1.0 + scoring_window).prod() - 1.0
            selected = score.nlargest(min(int(top_n), len(universe))).index.tolist()
            covariance = ewma_covariance(history[selected], decay=float(decay))
            selected_weights = strict_erc(covariance)
            target = selected_weights.reindex(universe, fill_value=0.0)
            target /= target.sum()
            pending = {'Signal Date': date, 'Target weights': target}

    result = {
        'returns': pd.Series(dict(daily_records), dtype=float),
        'turnover': pd.Series(dict(turnover_records), dtype=float),
        'weights': pd.DataFrame(weight_records), 'risk_free_daily': rf,
        'signals': pd.DataFrame(signal_records),
    }
    if not result['signals'].empty and not (
        result['signals']['Signal Date'] < result['signals']['Execution Date']
    ).all():
        raise RuntimeError('The robust model failed to isolate information in the future.')
    return result


def robustness_metrics(result):
    r = result['returns'].dropna()
    rf = result['risk_free_daily'].reindex(r.index)
    years = len(r) / TRADING_DAYS
    total = float((1.0 + r).prod() - 1.0)
    annual = (1.0 + total) ** (1.0 / years) - 1.0
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    excess = r - rf
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))
    max_dd = float(drawdown_series(r).min())
    traded_notional = float(result['turnover'].sum() / years)
    turnover = 0.5 * traded_notional
    return pd.Series({
        'Start': r.index.min(), 'Endpoint': r.index.max(), 'Trade day': len(r),
        'Cumulative Return': total, 'Annualized Return': annual, 'Annualized Volatility': vol,
        'Sharpe Ratio': sharpe, 'Maximum Drawdown': max_dd, 'Annual One-Way Turnover': turnover,
        'Annual Two-Way Traded Notional': traded_notional,
    })


NO_BTC_ASSETS = [ticker for ticker in ALL_ASSETS if ticker != 'BTC-USD']
no_btc_prices_full = pd.DataFrame(index=calendar, columns=NO_BTC_ASSETS, dtype=float)
for ticker in NO_BTC_ASSETS:
    currency = assets.loc[ticker, 'Local Currency']
    no_btc_prices_full[ticker] = (
        local_bday[ticker] if currency == 'SGD'
        else local_bday[ticker] * fx_to_sgd[currency]
    )
no_btc_start = max(no_btc_prices_full[ticker].first_valid_index() for ticker in NO_BTC_ASSETS)
no_btc_end = min(no_btc_prices_full[ticker].last_valid_index() for ticker in NO_BTC_ASSETS)
no_btc_returns_full = (
    no_btc_prices_full.loc[no_btc_start:no_btc_end]
    .dropna(how='any').pct_change(fill_method=None).dropna(how='any')
)

btc_same_sample = run_research_ewma_model(returns, ALL_ASSETS, strategy_start)
no_btc_same_sample = run_research_ewma_model(
    returns[NO_BTC_ASSETS], NO_BTC_ASSETS, strategy_start
)
no_btc_extended = run_research_ewma_model(
    no_btc_returns_full, NO_BTC_ASSETS, start_date=None
)
btc_ablation_metrics = pd.DataFrame({
    'Including BTC (same area)': robustness_metrics(btc_same_sample),
    'excluding BTC (same area)': robustness_metrics(no_btc_same_sample),
    'Excludes BTC (extended starting point)': robustness_metrics(no_btc_extended),
}).T
display(btc_ablation_metrics.style.format({
    'Cumulative Return': '{:.2%}', 'Annualized Return': '{:.2%}', 'Annualized Volatility': '{:.2%}',
    'Sharpe Ratio': '{:.2f}', 'Maximum Drawdown': '{:.2%}', 'Annual One-Way Turnover': '{:.2%}',
}))

btc_ablation_conclusion = pd.DataFrame({
    'Result': [btc_ablation_metrics.loc['Including BTC (same area)', 'Sharpe Ratio']
             - btc_ablation_metrics.loc['excluding BTC (same area)', 'Sharpe Ratio']]
}, index=['With BTC and no BTC.'])
display(btc_ablation_conclusion.style.format('{:.2f}'))

# 7.3 Key Parameter Stability of the Core EWMA Model
#
#
# Instead of looking for new best interests, this section examines the stability in the vicinity of the recommended parameters. Each time only one dimension is changed, the rest maintain baseline setting: Top10, 12-1, EWMA λ = 0.94, every 15 days. Test dimensions are Top N, kinetic window, EWMA decay factor and check interval. All results are based on the same end-point of the study, the closing of the book at the next transaction date and the flat base cost.
#
# > In order to avoid misreading the research tests as final product certification, this section only tests the TopN+EWMA strict ERC core layer of the motor mass. The final submission strategy includes dynamic double budgets, dynamic offensives, half-speed implementation and no-trade zones, whose robustness is judged by a combination of subsequent digestion, forward evaluation and cost testing.
# --- Original Notebook code module 83-
parameter_tests = {
    'Top N': [('Top N', value) for value in [6, 8, 10, 12, 14]],
    'Dynamic Window': [('Dynamic Window', value) for value in [63, 126, 189, LOOKBACK - MOMENTUM_SKIP]],
    'EWMA λ': [('EWMA λ', value) for value in [0.94, 0.96, 0.97, 0.98]],
    'Check interval': [('Check interval', value) for value in [10, 15, 20, 21]],
}

parameter_records = []
for dimension, settings in parameter_tests.items():
    for _, value in settings:
        kwargs = {
            'top_n': TOP_N, 'score_days': LOOKBACK - MOMENTUM_SKIP,
            'decay': EWMA_LAMBDA, 'check_interval': CHECK_INTERVAL,
        }
        if dimension == 'Top N':
            kwargs['top_n'] = int(value)
        elif dimension == 'Dynamic Window':
            kwargs['score_days'] = int(value)
        elif dimension == 'EWMA λ':
            kwargs['decay'] = float(value)
        elif dimension == 'Check interval':
            kwargs['check_interval'] = int(value)
        test_result = run_research_ewma_model(
            returns, ALL_ASSETS, strategy_start, **kwargs
        )
        metric = robustness_metrics(test_result)
        parameter_records.append({
            'Parameters': dimension, 'Parameter Value': value,
            'Cumulative Return': metric['Cumulative Return'], 'Annualized Return': metric['Annualized Return'],
            'Sharpe Ratio': metric['Sharpe Ratio'], 'Maximum Drawdown': metric['Maximum Drawdown'],
            'Annual One-Way Turnover': metric['Annual One-Way Turnover'],
        })

parameter_sensitivity = pd.DataFrame(parameter_records).set_index(['Parameters', 'Parameter Value'])
display(parameter_sensitivity.style.format({
    'Cumulative Return': '{:.2%}', 'Annualized Return': '{:.2%}', 'Sharpe Ratio': '{:.2f}',
    'Maximum Drawdown': '{:.2%}', 'Annual One-Way Turnover': '{:.2%}',
}))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()
shared_handles = shared_labels = None
for ax, dimension in zip(axes, parameter_tests):
    view = parameter_sensitivity.xs(dimension)
    ax.plot(view.index.astype(float), view['Sharpe Ratio'], marker='o', linestyle='-', color='#3366A3', label='Sharp.')
    ax2 = ax.twinx()
    ax2.plot(view.index.astype(float), view['Annual One-Way Turnover'], marker='s', linestyle='-', color='#E07A5F', label='Change hands.')
    ax.set_title(dimension)
    ax.set_ylabel('Sharpe Ratio')
    ax2.set_ylabel('Annual One-Way Turnover')
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=2))
    style_axis(ax)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if shared_handles is None:
        shared_handles = handles1 + handles2
        shared_labels = labels1 + labels2

fig.suptitle('Neighbourhood stability of key parameters (non-optimal)', fontsize=15, y=0.98)
fig.legend(
    shared_handles, shared_labels, loc='lower center', bbox_to_anchor=(0.5, 0.015),
    ncol=2, frameon=False, fontsize=10,
)
fig.tight_layout(rect=[0, 0.07, 1, 0.95], h_pad=2.2, w_pad=2.8)
plt.show()

parameter_stability_summary = parameter_sensitivity.groupby(level=0).agg(
    sharp_minimum=('Sharpe Ratio', 'min'), sharp_maximum=('Sharpe Ratio', 'max'),
    sharp_s_standard_deviation=('Sharpe Ratio', 'std'), maximum_retreat_margin=('Maximum Drawdown', 'min'),
    minimal_exchange_rate=('Annual One-Way Turnover', 'min'), maximum_exchange_rate=('Annual One-Way Turnover', 'max'),
)
display(parameter_stability_summary.style.format({
    'Sharp minimum': '{:.2f}', 'Sharp maximum': '{:.2f}', 'Sharp\'s standard deviation.': '{:.2f}',
    'Maximum retreat margin': '{:.2%}', 'Minimal exchange rate': '{:.2%}', 'Maximum exchange rate': '{:.2%}',
}))

# 7.4 Dynamic risk budget stratification
#
#
# The digestion experiment uses the same inter-research and uniform indicators, comparing strict EWMA ERC, single-layer dynamic budget, static category double layer, full dynamic double layer, fixed offensive and dynamic offensive versions. Since the single layer of dynamic and static categories are two different branches, the adjacent margin values in the table are used only to describe marginal changes and are not strictly causally explained.
# --- Original Notebook code unit 85--
ABLATION_STRATEGIES = [
    STRATEGY_NAMES['ewma_erc'],
    DYNAMIC_BUDGET_NAME,
    HIERARCHICAL_BUDGET_NAME,
    *CORE_COMPARISON_STRATEGIES,
]
ablation_table = pd.DataFrame({
    name: compute_metrics(results[name], strategy_start, RESEARCH_END)
    for name in ABLATION_STRATEGIES
}).T[[
    'Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
    'Maximum Drawdown', 'Annual One-Way Turnover', 'Number of Rebalances',
]]
ablation_table['Change over previous line of Sharp'] = ablation_table['Sharpe Ratio'].diff()
ablation_table['Change over previous line'] = ablation_table['Annual One-Way Turnover'].diff()
display(ablation_table.style.format({
    'Cumulative Return': '{:.2%}', 'Annualized Return': '{:.2%}', 'Annualized Volatility': '{:.2%}',
    'Sharpe Ratio': '{:.2f}', 'Maximum Drawdown': '{:.2%}', 'Annual One-Way Turnover': '{:.2%}',
    'Change over previous line of Sharp': '{:+.2f}', 'Change over previous line': '{:+.2%}',
    'Number of Rebalances': '{:.0f}',
}))

# 7.5 Annual Walker-Forward testing of freezing rules
#
#
# All policy rules and parameters are fixed and the study area is divided into a front-window for successive natural years. Each window evaluates only the current year ' s gains; the strategy still uses the previous 252-day historical signals before the window begins. The optimal parameters are not re-examined in each window, thus avoiding the duplication of annual evaluations. This test does not re-select parameters within each discount and is not equivalent to a full rolling training-valk-forward optimization.
#
# Data from locked 2026-08-01 to 2026-10-01 are not included in this section. The first and last incomplete year is not used to determine the number of full-year wins.
# --- Original Notebook code module 87-
WALK_FORWARD_STRATEGIES = [
    BASELINE_STRATEGY, STATIC_CLASSIC_NAME, STRATEGY_NAMES['ewma_erc'],
    FULLY_DYNAMIC_BUDGET_NAME, CORE_ATTACK_NAME, DYNAMIC_ATTACK_NAME, DYNAMIC_ATTACK_H20_NAME,
]
first_full_year = strategy_start.year + 1
last_full_year = RESEARCH_END.year - (RESEARCH_END.month < 12)
walk_forward_records = []
for year in range(first_full_year, last_full_year + 1):
    fold_start = pd.Timestamp(f'{year}-01-01')
    fold_end = min(pd.Timestamp(f'{year}-12-31'), RESEARCH_END)
    for strategy_name in WALK_FORWARD_STRATEGIES:
        metric = compute_metrics(results[strategy_name], fold_start, fold_end)
        walk_forward_records.append({
            'Forward year': year, 'Policy': strategy_name,
            'Cumulative Return': metric['Cumulative Return'], 'Sharpe Ratio': metric['Sharpe Ratio'],
            'Maximum Drawdown': metric['Maximum Drawdown'], 'Annual One-Way Turnover': metric['Annual One-Way Turnover'],
        })
walk_forward_metrics = pd.DataFrame(walk_forward_records).set_index(['Forward year', 'Policy'])
display(walk_forward_metrics.style.format({
    'Cumulative Return': '{:.2%}', 'Sharpe Ratio': '{:.2f}',
    'Maximum Drawdown': '{:.2%}', 'Annual One-Way Turnover': '{:.2%}',
}))

baseline_fold = walk_forward_metrics.xs(BASELINE_STRATEGY, level='Policy')['Sharpe Ratio']
walk_forward_summary = walk_forward_metrics.groupby(level='Policy').agg(**{
    'Median Annual Sharpe': ('Sharpe Ratio', 'median'),
    'Minimum Annual Sharpe': ('Sharpe Ratio', 'min'),
    'Positive-Year Ratio': ('Cumulative Return', lambda x: float((x > 0).mean())),
    'Worst Annual Drawdown': ('Maximum Drawdown', 'min'),
})
for strategy_name in WALK_FORWARD_STRATEGIES:
    strategy_fold = walk_forward_metrics.xs(strategy_name, level='Policy')['Sharpe Ratio']
    walk_forward_summary.loc[strategy_name, 'Years Beating Benchmark Sharpe'] = float(
        (strategy_fold > baseline_fold).mean()
    )
display(walk_forward_summary.sort_values('Median Annual Sharpe', ascending=False).style.format({
    'Median Annual Sharpe': '{:.2f}', 'Minimum Annual Sharpe': '{:.2f}',
    'Positive-Year Ratio': '{:.2%}', 'Worst Annual Drawdown': '{:.2%}',
    'Years Beating Benchmark Sharpe': '{:.2%}',
}))

walk_sharpe = walk_forward_metrics['Sharpe Ratio'].unstack('Policy')
walk_strategy_order = [name for name in WALK_FORWARD_STRATEGIES if name in walk_sharpe.columns]
walk_sharpe = walk_sharpe.reindex(columns=walk_strategy_order)
walk_plot = walk_sharpe.rename(columns=strategy_label)
ax = walk_plot.plot(
    figsize=(14, 6), marker='o', linewidth=1.8,
    color=[STRATEGY_COLORS[name] for name in walk_strategy_order],
)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Freezing rule Walker-Forward: Sharp ratio per full year')
ax.set_xlabel('Forward year')
ax.set_ylabel('Sharpe Ratio')
style_axis(ax)
place_legend(ax)
plt.tight_layout(rect=[0, 0.16, 1, 1])
plt.show()

# 7.6 Multifactor Alpha and Deflated Sharpe
#
#
# The return of ES3.SI or VT alone cannot be fully explained. This section adds four categories of total SSD return factors: global equities, US medium-term national debt, commodity basket and US dollar. Re-use SORA's excess earnings were wrong with the Newey-West standard and were only imputed ex post and did not enter the trading signal.
#
# Deflad Sharpe has roughly revised the selection deviations resulting from the simultaneous comparison of 12 official strategies, taking into account the earnings bias and peak. Because of the high degree of relevance of the strategies, the statistics represent a conservative diagnosis rather than an accurate probability of independent testing.
# --- Original Notebook code unit 89-
from scipy.stats import norm, skew, kurtosis


def multifactor_alpha(result, start=strategy_start, end=RESEARCH_END):
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
    x = aligned[factors.columns].sub(aligned['SORA'], axis=0)
    x = sm.add_constant(x)
    max_lags = max(1, int(4 * (len(aligned) / 100) ** (2 / 9)))
    model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    return pd.Series({
        'Yearic Multifactor Alpha': float(model.params['const'] * TRADING_DAYS),
        'Alpha t (HAC)': float(model.tvalues['const']),
        'Alpha p (HAC)': float(model.pvalues['const']),
        'Stock Beta': float(model.params['Global equities']),
        'Beta Bond': float(model.params['Medium-term national debt']),
        'Commodities Beta': float(model.params['Commodity basket']),
        'US$ Beta': float(model.params['United States dollars']),
        'Return to R2': float(model.rsquared),
    })


multifactor_alpha_table = pd.DataFrame({
    name: multifactor_alpha(results[name]) for name in final_strategy_names
}).T.sort_values('Yearic Multifactor Alpha', ascending=False)
display(multifactor_alpha_table.style.format({
    'Yearic Multifactor Alpha': '{:.2%}', 'Alpha t (HAC)': '{:.2f}',
    'Alpha p (HAC)': '{:.2f}', 'Stock Beta': '{:.2f}', 'Beta Bond': '{:.2f}',
    'Commodities Beta': '{:.2f}', 'US$ Beta': '{:.2f}', 'Return to R2': '{:.2f}',
}))


def deflated_sharpe_probability(result, trial_daily_sharpes):
    r = result['returns'].loc[strategy_start:RESEARCH_END].dropna()
    rf = result['risk_free_daily'].reindex(r.index)
    excess = (r - rf).dropna()
    daily_sr = float(excess.mean() / excess.std(ddof=1))
    trials = np.asarray(trial_daily_sharpes, dtype=float)
    trial_count = len(trials)
    if trial_count < 2 or not np.isfinite(trials).all():
        raise ValueError('Deflad Sharpe needs at least two limited candidate days.')
    trial_std = float(np.std(trials, ddof=1))
    euler_gamma = 0.5772156649015329
    trial_mean = float(np.mean(trials))
    expected_max = trial_mean + trial_std * (
        (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trial_count)
        + euler_gamma * norm.ppf(1.0 - 1.0 / (trial_count * np.e))
    )
    sample_skew = float(skew(excess, bias=False))
    sample_kurtosis = float(kurtosis(excess, fisher=False, bias=False))
    denominator = np.sqrt(max(
        1.0 - sample_skew * daily_sr
        + ((sample_kurtosis - 1.0) / 4.0) * daily_sr ** 2,
        1e-12,
    ))
    z_score = (daily_sr - expected_max) * np.sqrt(len(excess) - 1.0) / denominator
    return pd.Series({
        'General annual Sharp.': daily_sr * np.sqrt(TRADING_DAYS),
        'Multitest Base Day Sharp': expected_max,
        'Candidate strategy averaged Sharp': trial_mean,
        'Deflated Sharpe Probability': float(norm.cdf(z_score)),
        'Gain margin': sample_skew, 'Peak of return': sample_kurtosis,
        'Comparison of strategies': trial_count,
    })


trial_daily_sharpes = []
for name in final_strategy_names:
    r = results[name]['returns'].loc[strategy_start:RESEARCH_END].dropna()
    rf = results[name]['risk_free_daily'].reindex(r.index)
    trial_daily_sharpes.append(float((r - rf).mean() / (r - rf).std(ddof=1)))

deflated_sharpe_table = pd.DataFrame({
    name: deflated_sharpe_probability(results[name], trial_daily_sharpes)
    for name in final_strategy_names
}).T.sort_values('Deflated Sharpe Probability', ascending=False)
display(deflated_sharpe_table.style.format({
    'General annual Sharp.': '{:.2f}', 'Multitest Base Day Sharp': '{:.2f}',
    'Candidate strategy averaged Sharp': '{:.2f}',
    'Deflated Sharpe Probability': '{:.2%}', 'Gain margin': '{:.2f}', 'Peak of return': '{:.2f}',
    'Comparison of strategies': '{:.0f}',
}))

# 7.7 Realizability and cost boundaries
#
# The official master survey uses a unilateral 0.05 per cent transaction fee to maintain horizontal comparability between the original research calibre and the overall strategy. Sub-market commissions, swaps, price differential slips, market shocks and withholding taxes are used only in section 4.5 for stand-alone realistic cost scenarios and do not cover formal results. Since Yahoo does not have historical order books, coupons FX transactions and investor tax identities, realistic scenario parameters are transparent research assumptions rather than real costs.
#
# SORA cash gains have entered net value. The current strategy does not use leverage, nor does it have negative cash, so financing costs are not “missing zero-cost borrowing”; it is not applicable under the current product structure. If leverage is allowed in the future, SORA plus financing spreads should be used, and bonds and flat limits should be retraced.
# --- Original Notebook code unit 90--
tradability_limitations = pd.DataFrame({
    'Why can\'t we be reliable?': [
        'Formal master survey not separately listed; section 4.5 is recorded at a fixed rate in the currency of the transaction',
        'Formal master survey not listed; section 4.5 is based on market fixed price differential slide points Enter',
        'Formal master survey not separately listed; Section 4.5 modeled by ALV, Volatility and AUM',
        'Formal master survey not separately listed; section 4.5 is accounted for by annual asset tax negative agent',
    ],
    'Complete the data required for the model': [
        'FX quotes, real exchange transactions, cash accounts in currency',
        'Historical NBBO/order books, written-in and time stamp',
        'Target AUM, order size, ADV, participation rate and impact function calibration sample',
        'Accounts and tax status, place of registration, breakdown of funds and applicable tax rates',
    ],
    'Current processing': ['Independent reality', 'Independent reality', 'Independent reality', 'Independent reality'],
}, index=['FX Conversion Costs', 'Real trade price difference and slide point', 'Market Impact', 'Taxes and withholding taxes'])
display(tradability_limitations)

