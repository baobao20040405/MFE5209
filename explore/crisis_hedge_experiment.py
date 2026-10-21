"""Independent crisis sleeve research. Never modifies submitted product rules.

Accounting: frozen GMI-ADG fund subaccount + cash + collateralized short SPY
SGD total-return proxy. Short proceeds are segregated and earn zero rebate;
50% additional margin earns zero. Borrow fee is assumed, not observed.
This is a feasibility screen, not a broker-executable short backtest.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from product_evaluation import load_product_context
from realistic_costs import cost_schedule, replay_costs

END = pd.Timestamp('2026-07-31')
CAPITAL = 1_000_000.
OUT = ROOT / 'explore' / 'report_work' / 'crisis_hedge_experiment'
VARIANTS = {
    'GMI-ADG': ('none', 1., 0.),
    'CASH-30': ('broad', .70, 0.),
    'SHORT-10': ('broad', .70, .10),
    'SHORT-20': ('broad', .70, .20),
    'CORR-SHORT-20': ('correlation', .70, .20),
}
COLORS = {'GMI-ADG': '#b82626', 'CASH-30': '#2b549c',
          'SHORT-10': '#e5a236', 'SHORT-20': '#55b7a7', 'CORR-SHORT-20': '#9278ae'}


def crisis_signals(context, dates):
    r = context['returns']
    # Equal-weight within each asset class, equal class representation in breadth.
    classes = pd.DataFrame({group: r.loc[:, context['assets'].index[
        context['assets']['Asset Class'].eq(group)]].mean(axis=1)
        for group in ['Equities', 'Bonds', 'Commodities', 'FX', 'Crypto']})
    class_nav = (1 + classes).cumprod()
    negative = (class_nav.pct_change(63, fill_method=None) < 0).mean(axis=1)
    corr = classes.rolling(60, min_periods=60).corr()
    average = corr.groupby(level=0).apply(lambda x: float(
        x.to_numpy()[np.triu_indices(5, k=1)].mean()))
    vt = context['benchmark_prices_sgd']['VT']
    vt_down = (vt < vt.rolling(200).mean()) & (vt.pct_change(63, fill_method=None) < 0)
    frame = pd.DataFrame({'Negative Class Breadth': negative,
        'Mean Class Correlation': average, 'VT Downtrend': vt_down}).reindex(dates)
    # Entry is deliberately fixed before inspection; no threshold optimization.
    # Three of five classes negative; correlation threshold of 0.30 is a sensitivity variant.
    for mode in ['broad', 'correlation']:
        active, states = False, []
        for date in dates:
            row = frame.loc[date]
            entry = bool(row['VT Downtrend']) and row['Negative Class Breadth'] >= .60
            if mode == 'correlation':
                entry = entry and row['Mean Class Correlation'] >= .30
            # Hysteresis: leave only on equity trend recovery or <=1 negative class.
            exit_signal = not bool(row['VT Downtrend']) or row['Negative Class Breadth'] <= .20
            if not active and entry:
                active = True
            elif active and exit_signal:
                active = False
            states.append(active)
        frame[mode] = states
    return frame


def simulate(base, spy, rf, signals, mode, core_target, short_target,
             cost_mode, borrow_rate, liquidity, capital=CAPITAL):
    """Close-t signal executes close-t+1, changes returns only from t+2.

    Core and short exposures drift between state transitions. Transaction fees
    are funded from cash, never deducted twice from the embedded core return.
    """
    core, cash, liability, locked = capital, 0., 0., 0.
    pending, state = None, False
    rows = []
    for date in base.index:
        previous = core + cash - liability
        opening_core, opening_short = core / previous, liability / previous
        margin = .50 * liability
        free_cash = max(0., cash - locked - margin)
        borrow = liability * borrow_rate / 252
        core *= 1 + base.loc[date]
        cash += free_cash * rf.loc[date] - borrow
        # Total-return short proxy includes dividend compensation; no extra dividend fee.
        liability *= 1 + spy.loc[date]
        nav = core + cash - liability
        fee, traded = 0., 0.
        if pending is not None and pending != state:
            state = pending
            target_core = core_target if state else 1.
            target_short = short_target if state else 0.
            new_core, new_liability = nav * target_core, nav * target_short
            core_trade, short_trade = abs(new_core - core), abs(new_liability - liability)
            unit_fee = .0005 if cost_mode == 'Fixed 0.05%' else .0015
            # US commission+spread+USD conversion = 0.15%; core adjustment uses
            # this conservative basket proxy, not a falsely exact cross-market fill.
            impact = 0.
            if cost_mode != 'Fixed 0.05%' and short_trade > 0:
                adv, vol = liquidity.loc[date, ['ADV', 'Volatility']]
                if not np.isfinite(adv) or adv <= 0 or not np.isfinite(vol):
                    raise ValueError('Missing lagged SPY liquidity.')
                impact = short_trade * min(.01, .10 * vol * np.sqrt(short_trade / adv))
            # Fund fees by slightly reducing core purchase, so returning to 100%
            # core does not accidentally create a negative cash balance.
            for _ in range(20):
                new_core = nav * target_core - fee
                core_trade = abs(new_core - core)
                fee = (core_trade + short_trade) * unit_fee + impact
            new_core = nav * target_core - fee
            # Buying core consumes cash; short issuance adds segregated proceeds.
            cash -= new_core - core
            cash += new_liability - liability
            core, liability = new_core, new_liability
            locked = liability
            cash -= fee
            traded = (core_trade + short_trade) / nav
        pending = bool(signals.loc[date, mode]) if mode != 'none' else False
        end_nav = core + cash - liability
        if end_nav <= 0 or cash + 1e-6 < locked + .50 * liability:
            raise ValueError(f'Insolvency or insufficient modeled margin on {date}.')
        rows.append({'Return': end_nav / previous - 1, 'NAV': end_nav,
                     'Core Weight': opening_core, 'Short Weight': opening_short,
                     'Crisis Active': state, 'Turnover': traded,
                     'Overlay Cost / NAV': fee / previous,
                     'Borrow Cost / NAV': borrow / previous})
    return pd.DataFrame(rows, index=base.index)


def metrics(r, rf):
    wealth = (1 + r).cumprod()
    excess = r - rf.reindex(r.index)
    return {'Cumulative Return': wealth.iloc[-1] - 1,
            'Annualized Return': wealth.iloc[-1] ** (252 / len(r)) - 1,
            'Sharpe Ratio': excess.mean() / excess.std(ddof=1) * np.sqrt(252),
            'Maximum Drawdown': (wealth / wealth.cummax().clip(lower=1) - 1).min(),
            'Daily ES 5%': r[r <= r.quantile(.05)].mean()}


def run_experiment():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'code').glob('*.py')}
    context, sora = load_product_context(END, CAPITAL)
    baseline = context['run_dynamic_attack_backtest'](end_date=END, sora_frame=sora)
    dates = baseline['returns'].index
    rf = baseline['risk_free_daily'].reindex(dates)
    realistic, _ = replay_costs(baseline, context, dates, CAPITAL, cost_schedule(context['assets']))
    signals = crisis_signals(context, dates)
    spy = context['returns']['SPY'].reindex(dates)
    liquidity = pd.DataFrame({'ADV': context['dollar_volume_sgd']['SPY'].shift(1).rolling(20, min_periods=5).median(),
        'Volatility': context['returns']['SPY'].shift(1).rolling(20, min_periods=5).std()}).reindex(dates)
    rows, daily = [], {}
    periods = {'Full History': (dates[0], END), 'Development': (dates[0], '2024-12-31'),
               'Historical Validation': ('2025-01-01', END),
               '2018 Selloff': ('2018-10-01', '2018-12-31'),
               'COVID Decline': ('2020-02-20', '2020-03-23'),
               'COVID Rebound': ('2020-03-24', '2020-06-30'),
               '2022 Stock-Bond Stress': ('2022-01-01', '2022-10-31')}
    for cost_mode, result in [('Fixed 0.05%', baseline), ('Realistic Costs', realistic)]:
        for name, (mode, core, hedge) in VARIANTS.items():
            frame = simulate(result['returns'], spy, rf, signals, mode, core, hedge,
                             cost_mode, .03, liquidity)
            if name == 'GMI-ADG':
                np.testing.assert_allclose(frame['Return'], result['returns'], atol=1e-12)
            daily[(cost_mode, name)] = frame
            for period, (start, end) in periods.items():
                segment = frame.loc[start:end]
                row = metrics(segment['Return'], rf)
                row.update({'Cost Scenario': cost_mode, 'Variant': name, 'Period': period,
                    'Crisis Days': int(segment['Crisis Active'].sum()),
                    'Mean Short Weight': segment['Short Weight'].mean(),
                    'Extra Traded Notional': segment['Turnover'].sum()})
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'comparison.csv', index=False)
    pd.concat(daily, names=['Cost Scenario', 'Variant', 'Date']).to_csv(OUT / 'daily_ledger.csv')
    signals.to_csv(OUT / 'crisis_signals.csv', index_label='Date')
    # Borrow stress is not a new signal search; show all three financing assumptions.
    sensitivity = []
    for borrow in [.01, .03, .10]:
        for name in ['SHORT-20', 'CORR-SHORT-20']:
            mode, core, hedge = VARIANTS[name]
            frame = simulate(realistic['returns'], spy, rf, signals, mode, core, hedge,
                             'Realistic Costs', borrow, liquidity)
            for period in ['Full History', 'Historical Validation']:
                start, end = periods[period]
                row = metrics(frame.loc[start:end, 'Return'], rf)
                sensitivity.append(dict(row, Variant=name, Period=period, BorrowRate=borrow))
    pd.DataFrame(sensitivity).to_csv(OUT / 'borrow_sensitivity.csv', index=False)
    for cost_mode in ['Fixed 0.05%', 'Realistic Costs']:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for name in VARIANTS:
            r = daily[(cost_mode, name)]['Return']
            base_date = r.index[0] - pd.Timedelta(days=1)
            wealth = pd.concat([pd.Series([1.], index=[base_date]), (1 + r).cumprod()])
            axes[0].plot(wealth.index, wealth - 1, label=name, color=COLORS[name], linewidth=1.6)
            dd = wealth / wealth.cummax() - 1
            axes[1].plot(dd.index, dd, color=COLORS[name], linewidth=1.3)
        axes[0].set_title(f'Crisis-Sleeve Experiment | {cost_mode} | Through 2026-07-31')
        axes[0].set_ylabel('Cumulative Return'); axes[1].set_ylabel('Drawdown')
        axes[1].set_xlabel('Date')
        for ax in axes:
            ax.grid(False); ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(OUT / ('fixed_costs.png' if cost_mode == 'Fixed 0.05%' else 'realistic_costs.png'), dpi=160)
        plt.close(fig)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'code').glob('*.py')}
    assert hashes == after, 'Submitted code changed during experiment.'
    (OUT / 'methodology.json').write_text(json.dumps({
        'end': str(END.date()), 'capital_sgd': CAPITAL, 'variants': VARIANTS,
        'entry': 'VT below MA200 and 63-day return negative; >=3/5 classes negative 63-day returns',
        'correlation_variant': 'Additionally 60-day mean off-diagonal class correlation >=0.30 at entry',
        'exit': 'Equity downtrend condition no longer true OR <=1/5 negative classes',
        'execution': 'Signal close t, trade close t+1, return impact from t+2',
        'borrow': '3% annual base; 1% and 10% sensitivities; zero rebate; 50% additional margin',
        'limitations': ['Frozen baseline fund path; overlay does not feed into core drawdown decisions',
            'Short SPY SGD total-return proxy: currency/broker execution and stock-loan availability not certified',
            'No October holdout inspected; historical validation was already viewed in prior research',
            'Baseline realistic sleeve is frozen-path sensitivity; overlay core resizing uses 0.15% cost proxy',
            'No crisis guarantee, no short availability or forced recall model; no optimized thresholds'],
        'product_code_sha256': hashes, 'unchanged_after_run': hashes == after}, indent=2), encoding='utf-8')
    print(summary[summary['Period'].isin(['Full History', 'Historical Validation'])].to_string(index=False))
    return summary


if __name__ == '__main__':
    run_experiment()
