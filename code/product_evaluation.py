"""Standalone final GMI-ATC product: evaluate five net-of-cost SGD series (GMI-ATC, GMI-ADG, GMA-EW, VT, STI)."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_sora_daily, align_sora_returns, process_sora, sora_alignment_audit
from download import download_market_data
from export import export_asset_closing_prices
from plots import plot_comparison, display_charts
from validation import validate_quote_coverage, QuoteCoverageError
from realistic_costs import cost_schedule, replay_costs

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def _execute(filename, context):
    path = CODE / filename
    context['__file__'] = str(path)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), context)


def load_product_context(end, capital, *, raw_sora=None, refresh=False):
    """Load standalone product files; never import anything from explore/."""
    context = {'__name__': '__final_product__'}
    _execute('config.py', context)
    context.update(INITIAL_CAPITAL_SGD=float(capital), DOWNLOAD_END=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
                   display=lambda *args, **kwargs: None)
    if context['COST'] != 0.0005:
        raise ValueError('The submitted product requires a fixed one-way fee of 0.05%.')
    sora = load_sora_daily(end, raw_sora)
    context.update(sora_daily_df=sora, align_sora_returns=align_sora_returns,
                   download_market_data=lambda *args, **kwargs: download_market_data(*args, refresh=refresh, **kwargs))
    _execute('assets.py', context)
    metadata = pd.DataFrame(context['asset_rows'], columns=[
        'Asset Class', 'Market', 'Asset Name', 'Code', 'ETF English Name', 'Local Currency']).set_index('Code')
    tickers = metadata.index.tolist()
    context.update(assets=metadata, asset_metadata=metadata, ALL_ASSETS=tickers, PORTFOLIO_ASSETS=tickers)
    _execute('market_data.py', context)
    for module in ['strategy.py', 'backtest.py', 'evaluation.py']:
        _execute(module, context)
    return context, sora


def run_market_reference(context, ticker, end, sora):
    """Buy-and-hold ETF: next-close entry and 0.05% executed-notional fee.

    Entry occurs on the same first execution date as the replayed product. The
    test inherits that position, just as the product inherits earlier holdings.
    No fictional exit fee is charged to an unliquidated ending position.
    """
    dates = context['returns'].loc[context['strategy_start']:end].index
    name = 'STI' if ticker == 'ES3.SI' else ticker
    price = context['benchmark_prices_sgd'][ticker].reindex(dates)
    if price.isna().any():
        raise ValueError(f'{name} benchmark prices have gaps in the shared valuation period.')
    asset = context['benchmark_returns_sgd'][ticker].reindex(dates)
    rf = align_sora_returns(dates, sora, allow_holdout=True)
    net = rf.copy()
    if len(dates) < 3:
        raise ValueError('Not enough observations for benchmark entry.')
    net.iloc[1] = (1 + rf.iloc[1]) * (1 - context['COST']) - 1
    net.iloc[2:] = asset.iloc[2:]
    before = pd.DataFrame(0.0, index=dates, columns=[ticker])
    before.iloc[2:, 0] = 1
    closing = pd.DataFrame(1.0, index=dates, columns=[ticker])
    closing.iloc[0, 0] = 0
    delta = pd.DataFrame(0.0, index=dates, columns=[ticker]); delta.iloc[1, 0] = 1
    event = pd.DatetimeIndex([dates[1]])
    return {'name': name, 'returns': net.rename(name), 'weights': closing.loc[event],
        'turnover': pd.Series([1.0], index=event), 'trade_legs': pd.Series([1], index=event),
        'transaction_cost': pd.Series([context['COST']], index=event),
        'pre_return_weights': before, 'end_of_day_weights': closing, 'trade_deltas': delta,
        'risk_free_daily': rf, 'funding_daily': rf, 'trade_assets': [ticker],
        'instrument_prices': context['benchmark_prices_sgd'][[ticker]],
        'instrument_returns': context['benchmark_returns_sgd'][[ticker]],
        'asset_cost_rate': pd.Series(context['COST'], index=[ticker])}


def _evaluate_final_product(start='2026-10-01', end='2026-10-31', *, initial_capital=1_000_000.0,
                           allow_partial=False, show_plots=False, export_prices=True,
                           output_dir=None, raw_sora=None, refresh=False,
                           strict_quote_coverage=True):
    """Evaluate frozen final rules, retaining the earlier portfolio state.

    Capital is SGD. All five series use actual executed-notional fees of 0.05%.
    The requested date window changes evaluation only; it never tunes a model.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    today = pd.Timestamp.now(tz='Asia/Singapore').tz_localize(None).normalize()
    if pd.isna(start) or pd.isna(end) or start > end or not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('Enter valid dates with start <= end and a positive finite investment.')
    output = Path(output_dir or ROOT / 'output').resolve()
    tables, figures = output / 'tables', output / 'figures'
    for folder in (tables, figures):
        folder.mkdir(parents=True, exist_ok=True)
    status = {'strategy': 'GMI-ATC', 'comparisons': ['GMI-ATC', 'GMI-ADG', 'GMA-EW', 'VT', 'STI'],
        'requested_start': str(start.date()), 'requested_end': str(end.date()),
        'initial_capital_sgd': initial_capital, 'cost': 'Fixed 0.05% one-way', 'status': 'PENDING'}
    status_file = output / 'evaluation_status.json'
    status['output_dir'] = str(output)
    if end >= today and not allow_partial:
        status['reason'] = 'Requested period is unfinished. Choose partial evaluation or run after the period ends.'
        # An unfinished request must not relabel earlier successful figures/tables.
        (output / 'evaluation_request_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        return status
    effective_end = min(end, today - pd.Timedelta(days=1))
    if effective_end < start:
        raise ValueError('No completed observation can exist in this interval yet.')
    context, sora = load_product_context(effective_end, initial_capital, raw_sora=raw_sora, refresh=refresh)
    last_price = context['returns'].index.max()
    last_publication = sora.index.max()
    if effective_end - last_price > pd.Timedelta(days=7):
        raise ValueError(f'Update prices / raw SORA. Latest price: {last_price.date()}; latest publication: {last_publication.date()}.')
    # A Singapore holiday need not be a holiday in the other markets. Validate
    # availability as of each return date instead of clipping to publication day.
    actual_end = min(effective_end, last_price)
    align_sora_returns(context['returns'].loc[context['strategy_start']:actual_end].index,
                       sora, allow_holdout=True)
    final_business_day = pd.bdate_range(end=end, periods=1)[0]
    if not allow_partial and actual_end < final_business_day:
        raise ValueError('Data do not reach the requested final business day. Update inputs or allow partial evaluation.')
    if start < context['strategy_start']:
        raise ValueError(f'Evaluation cannot start before the 252-day warm-up ends: {context["strategy_start"].date()}.')
    historical_quote_note = None
    try:
        quote_audit = validate_quote_coverage(context, start, actual_end)
    except QuoteCoverageError as exc:
        if not strict_quote_coverage:
            historical_quote_note = str(exc)
            quote_audit = pd.DataFrame([{
                'Scope': 'Historical research replay',
                'Policy': 'Research convention: valuation prices may be forward-filled for at most 5 business days.',
                'Observed quote warning': historical_quote_note,
            }])
        elif not allow_partial or exc.safe_end is None:
            raise
        else:
            valid_dates = context['returns'].index
            valid_dates = valid_dates[(valid_dates >= start) & (valid_dates <= exc.safe_end)]
            if len(valid_dates) < 2:
                raise ValueError('Fewer than two fully observed evaluation dates. Refresh prices or choose an earlier interval.') from exc
            actual_end = valid_dates.max()
            quote_audit = validate_quote_coverage(context, start, actual_end)
            status['partial_reason'] = f'Latest common observed coverage ends {actual_end.date()}. {exc}'
    results = {
        'GMI-ATC': context['run_dynamic_attack_h20_backtest'](end_date=actual_end, sora_frame=sora, allow_holdout=True),
        'GMI-ADG': context['run_dynamic_attack_backtest'](end_date=actual_end, sora_frame=sora, allow_holdout=True),
        'GMA-EW': context['run_dynamic_backtest'](context['STRATEGY_NAMES']['equal_weight'],
            context['target_equal_weight'], end_date=actual_end, sora_frame=sora, allow_holdout=True),
        'VT': run_market_reference(context, 'VT', actual_end, sora),
        'STI': run_market_reference(context, 'ES3.SI', actual_end, sora),
    }
    for result in results.values():
        # Exposure must use daily actual holdings even if no rebalance falls
        # inside the selected interval (common for buy-and-hold references).
        result['gross_exposure'] = result['pre_return_weights'].abs().sum(axis=1)
    returns = pd.concat({name: result['returns'].loc[start:actual_end] for name, result in results.items()}, axis=1)
    if len(returns) < 2 or returns.isna().any().any():
        raise ValueError('Require at least two complete shared return observations.')
    metrics = pd.DataFrame({name: context['compute_metrics'](result, start, actual_end) for name, result in results.items()}).T
    metrics.index.name = 'Strategy'
    course, ledgers = context['build_course_required_metrics'](results, start, actual_end)
    course = course.rename(columns={'The odds.': 'Trade Win Rate', 'Number of evacuations': 'Drawdown Events',
        'Number of retreats repaired': 'Recovered Drawdowns',
        'Unrecovered evacuations at the end of the period': 'Unrecovered Drawdowns',
        'Average withdrawal duration (trade date)': 'Average Drawdown Duration (trading days)',
        'Maximum withdrawal duration (trade date)': 'Maximum Drawdown Duration (trading days)',
        'Total number of transactions completed': 'Completed FIFO Matches',
        'Average holding period (trade date)': 'Average Holding Period (trading days)',
        'Average profit for single transactions (SGD)': 'Average Profit per FIFO Match (SGD)',
        'Average rate of return on individual transactions': 'Average FIFO Match Return',
        'Number of unprocessed authorizations at end of period': 'Open FIFO Lots',
        'Unsettled assets at end of period': 'Open Assets'})
    course.index.name = 'Strategy'
    metrics.to_csv(tables / 'performance_metrics.csv')
    quote_audit.to_csv(tables / 'observed_quote_coverage.csv', index=False)
    course.to_csv(tables / 'course_metrics.csv')
    returns.to_csv(tables / 'daily_returns.csv', index_label='Date')
    nav = (1 + returns).cumprod() * initial_capital
    nav.to_csv(tables / 'daily_nav.csv', index_label='Date')
    (nav - initial_capital).to_csv(tables / 'daily_pnl.csv', index_label='Date')
    (nav / nav.cummax().clip(lower=initial_capital) - 1).to_csv(tables / 'daily_drawdown.csv', index_label='Date')
    weights = {name: result['end_of_day_weights'].reindex(returns.index) for name, result in results.items()}
    pd.concat(weights, names=['Strategy', 'Date']).to_csv(tables / 'daily_weights.csv')
    weights['GMI-ATC'].to_csv(tables / 'chosen_product_weights.csv', index_label='Date')
    pd.concat(ledgers, names=['Strategy', 'Match']).to_csv(tables / 'closed_trades.csv')
    trades = []
    for name, result in results.items():
        delta = result['trade_deltas'].reindex(returns.index).fillna(0)
        rows = delta.stack()
        rows = rows[rows.abs() > 1e-10].rename('Executed Weight Change').reset_index()
        rows.columns = ['Date', 'Ticker', 'Executed Weight Change']; rows['Strategy'] = name
        rows['One-Way Fee'] = context['COST']
        trades.append(rows)
    pd.concat(trades, ignore_index=True).to_csv(tables / 'executed_trades.csv', index=False)
    pd.DataFrame({'Strategy': list(results), 'One-Way Fee': context['COST'],
        'Convention': 'Fee times absolute executed-notional weight; no fee without trading.'}).to_csv(tables / 'cost_schedule.csv', index=False)
    sora_alignment_audit(context['returns'].loc[context['strategy_start']:actual_end].index, sora).to_csv(tables / 'sora_alignment.csv')
    partial = end >= today or actual_end < final_business_day
    colors = {'GMI-ATC': context['STRATEGY_COLORS'][context['DYNAMIC_ATTACK_H20_NAME']],
        'GMI-ADG': context['STRATEGY_COLORS'][context['DYNAMIC_ATTACK_NAME']],
        'GMA-EW': context['STRATEGY_COLORS'][context['STRATEGY_NAMES']['equal_weight']],
        'VT': context['MARKET_BENCHMARK_COLORS']['VT'], 'STI': context['MARKET_BENCHMARK_COLORS']['ES3.SI']}
    plot_metadata = context['assets'].copy()
    plot_metadata.loc['VT', 'Asset Class'] = 'Equities'
    charts, grouped = plot_comparison(returns, weights, plot_metadata, initial_capital, figures,
        colors, context['ASSET_CLASS_COLORS'], partial=partial)
    # Separate sensitivity outputs: never overwrite the official fixed-fee series.
    cost_metadata = context['assets'].copy()
    cost_metadata.loc['VT', ['Market', 'Local Currency']] = ['United States', 'USD']
    schedule = cost_schedule(cost_metadata)
    schedule.to_csv(tables / 'realistic_cost_schedule.csv', index_label='Ticker')
    realistic, components = {}, {}
    for name, result in results.items():
        realistic[name], components[name] = replay_costs(
            result, context, returns.index, initial_capital, schedule)
    realistic_returns = pd.DataFrame({name: item['returns'] for name, item in realistic.items()})
    realistic_metrics = pd.DataFrame({name: context['compute_metrics'](item, start, actual_end)
                                      for name, item in realistic.items()}).T
    realistic_metrics.index.name = 'Strategy'
    realistic_metrics.to_csv(tables / 'realistic_performance_metrics.csv')
    pd.concat({'Fixed 0.05%': metrics, 'Realistic Costs': realistic_metrics},
              names=['Cost Scenario', 'Strategy']).to_csv(tables / 'cost_scenario_metrics.csv')
    realistic_returns.to_csv(tables / 'realistic_daily_returns.csv', index_label='Date')
    ((1 + realistic_returns).cumprod() * initial_capital).to_csv(
        tables / 'realistic_daily_nav.csv', index_label='Date')
    pd.concat(components, names=['Strategy', 'Date']).to_csv(tables / 'realistic_daily_cost_components.csv')
    realistic_charts, _ = plot_comparison(realistic_returns, weights, plot_metadata,
        initial_capital, figures / 'realistic_costs', colors,
        context['ASSET_CLASS_COLORS'], partial=partial, cost_scenario='Realistic Costs (Frozen Path)')
    charts.update({f'realistic_{key}': value for key, value in realistic_charts.items()})
    pd.concat(grouped, names=['Strategy', 'Date']).to_csv(tables / 'asset_class_weights.csv')
    if export_prices:
        export_asset_closing_prices(context, tables, end=actual_end)
    else:
        context['assets'].to_csv(tables / 'asset_list.csv')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')}
    status.update(status='PARTIAL' if partial else 'COMPLETED', actual_start=str(returns.index[0].date()),
        actual_end=str(returns.index[-1].date()), observations=len(returns), charts=charts,
        source_sha256=hashes, sora_publication_date=str(last_publication.date()),
        quote_coverage='Observed exchange-session Close coverage for all 18 assets, VT and VIX inputs, plus VIX Open, checked over the evaluation interval; FX input quotes use London sessions as a conservative minimum-coverage proxy, not a global FX calendar. See observed_quote_coverage.csv. Does not certify intraday liquidity or historical warm-up completeness.',
        interpretation='Current fixed rules replay earlier holdings before the evaluation interval; historical windows inspected during research are not untouched holdout evidence.',
        final_strategy_note='GMI-ATC = GMI-ADG plus one offensive-eligibility rule: signal-day close above its 20-day moving average (checked only on the 15-session signal dates). The 20-day window was selected on the 2015-2026 research period under both cost scenarios; the rule was introduced on 2026-10-21, so October 2026 is reported as an observation rather than independent out-of-sample evidence. GMI-ADG is the no-confirmation ablation.',
        benchmark_convention='All series are SGD total-return proxies net of 0.05% executed-notional fees. VT/STI are buy-and-hold, entered at the next close after the first replay signal, with no forced final liquidation.',
        trade_definition='FIFO units reconcile to post-fee NAV and closing weights. Recorded fees are allocated, not estimated twice. Win rate, holding period and profit per trade exclude fee-funding-only sales and open lots. Opening holdings are marked at the evaluation boundary. No ordinary closed trades means unavailable, not zero win rate.',
        sora_convention='Latest prior-value-date SORA published by return date; annual percent / 100 * actual calendar gap / 365. Publication availability improves holiday alignment relative to research shift/reindex.')
    status['realistic_cost_convention'] = (
        'Supplementary frozen-weight/trade-path sensitivity, not a reoptimized or independently '
        'executed portfolio. Initial scenario NAV equals requested capital; inherited positions '
        'are not charged again. Market/currency fees and annual tax proxies follow explore; '
        'VT annual tax proxy is 0.36%. FX charged per foreign trade without currency netting. '
        'Impact = min(1%, 0.10 * lagged 20-session daily volatility * sqrt(order SGD / '
        'lagged 20-session median SGD volume)), minimum 5 observations. Order value uses '
        'pre-cost post-return NAV. No empirical broker calibration. No realistic FIFO metrics.')
    if historical_quote_note:
        status['historical_quote_warning'] = historical_quote_note
        status['quote_coverage'] = (
            'Historical research convention: isolated source gaps are carried forward for no more than '
            '5 business days. This is not the strict observed-quote certification used by the formal '
            'out-of-sample evaluator. See observed_quote_coverage.csv.'
        )
    status_file.write_text(json.dumps(status, indent=2), encoding='utf-8')
    if show_plots:
        display_charts(charts)
    return {'status': status, 'metrics': metrics, 'course_metrics': course,
            'daily_returns': returns, 'daily_nav': nav, 'weights': weights, 'charts': charts,
            'realistic_metrics': realistic_metrics, 'realistic_daily_returns': realistic_returns}


def _publish_outputs(stage, output):
    """Replace one complete run, rolling back all moved artifacts on failure.

    Move the old status away first and publish the new status last. Therefore
    no valid success marker labels a partially published mixture of runs.
    """
    output.mkdir(parents=True, exist_ok=True)
    names = ['evaluation_status.json', 'tables', 'figures']
    # Do not use TemporaryDirectory here: failed recovery must never delete
    # the last recoverable old files when its context manager exits.
    backup = Path(tempfile.mkdtemp(prefix='.previous-run-', dir=output.parent))
    saved, installed = [], []
    try:
        for name in names:
            if (output / name).exists():
                os.replace(output / name, backup / name)
                saved.append(name)
        for name in ['tables', 'figures', 'evaluation_status.json']:
            os.replace(stage / name, output / name)
            installed.append(name)
    except BaseException as publish_error:
        recovery_errors = []
        for name in reversed(installed):
            try:
                os.replace(output / name, stage / name)
            except OSError as exc:
                recovery_errors.append(f'Remove new {name}: {exc}')
        for name in reversed(saved):
            # Never restore a success marker over a partially recovered run.
            if name == 'evaluation_status.json' and recovery_errors:
                continue
            try:
                os.replace(backup / name, output / name)
            except OSError as exc:
                recovery_errors.append(f'Restore old {name}: {exc}')
        if recovery_errors:
            raise RuntimeError(
                f'Output publication and recovery failed. Recoverable old files '
                f'are retained at {backup}; other restored files remain at {output}. '
                f'Close applications locking these files before manual recovery. '
                + '; '.join(recovery_errors)
            ) from publish_error
        shutil.rmtree(backup, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def evaluate_final_product(start='2026-10-01', end='2026-10-31', *, initial_capital=1_000_000.0,
                           allow_partial=False, show_plots=False, export_prices=True,
                           output_dir=None, raw_sora=None, refresh=False,
                           strict_quote_coverage=True):
    """Evaluate and publish only after every table, chart and export succeeds."""
    output = Path(output_dir or ROOT / 'output').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.evaluation-', dir=output.parent) as temporary:
        stage = Path(temporary)
        report = _evaluate_final_product(start, end, initial_capital=initial_capital,
            allow_partial=allow_partial, show_plots=False, export_prices=export_prices,
            output_dir=stage, raw_sora=raw_sora, refresh=refresh,
            strict_quote_coverage=strict_quote_coverage)
        if not isinstance(report.get('status'), dict):
            report['output_dir'] = str(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'evaluation_request_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            return report
        charts = {key: str(output / Path(path).relative_to(stage))
                  for key, path in report['charts'].items()}
        report['charts'] = charts
        report['status'].update(output_dir=str(output), charts=charts)
        (stage / 'evaluation_status.json').write_text(json.dumps(report['status'], indent=2), encoding='utf-8')
        _publish_outputs(stage, output)
    if show_plots:
        display_charts(charts)
    return report


def evaluate_historical_backtest(*, show_plots=False, refresh=False):
    """Run the frozen historical replay with fixed dates and fixed SGD capital."""
    return evaluate_final_product(
        '2015-09-07', '2026-07-31', initial_capital=1_000_000.0,
        allow_partial=False, show_plots=show_plots, export_prices=False,
        output_dir=ROOT / 'output' / 'historical_backtest', refresh=refresh,
        strict_quote_coverage=False,
    )


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start'); parser.add_argument('--end')
    parser.add_argument('--capital', type=float)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--no-price-export', action='store_true')
    parser.add_argument('--output-dir'); parser.add_argument('--raw-sora')
    parser.add_argument('--refresh', action='store_true', help='Refresh downloaded market snapshots.')
    parser.add_argument('--process-sora', action='store_true', help='Only clean and validate the MAS export.')
    args = parser.parse_args()
    if args.process_sora:
        frame = process_sora(args.raw_sora)
        print(f'Processed {len(frame)} SORA observations; last publication: {frame["Publication Date"].max().date()}.')
        return
    if len(sys.argv) == 1 and sys.stdin.isatty():
        print('Final GMI-ATC product | SGD | fixed one-way fee 0.05%')
        while True:
            try:
                args.capital = float((input('Initial investment in SGD [1000000]: ').strip() or '1000000').replace(',', ''))
                if not math.isfinite(args.capital) or args.capital <= 0:
                    raise ValueError()
                break
            except ValueError:
                print('Enter a positive finite amount.')
        from datetime import datetime
        while True:
            try:
                args.start = input('Test start YYYY-MM-DD [2026-10-01]: ').strip() or '2026-10-01'
                args.end = input('Test end YYYY-MM-DD [2026-10-31]: ').strip() or '2026-10-31'
                if datetime.strptime(args.start, '%Y-%m-%d') > datetime.strptime(args.end, '%Y-%m-%d'):
                    raise ValueError()
                break
            except ValueError:
                print('Enter valid dates with start <= end.')
        today = pd.Timestamp.now(tz='Asia/Singapore').tz_localize(None).normalize()
        if pd.Timestamp(args.end) >= today:
            args.allow_partial = input('Unfinished period: show partial results? [y/N]: ').strip().lower() in {'y', 'yes'}
        args.show_plots = input('Display charts after saving? [Y/n]: ').strip().lower() not in {'n', 'no'}
    try:
        report = evaluate_final_product(args.start or '2026-10-01', args.end or '2026-10-31',
            initial_capital=args.capital if args.capital is not None else 1_000_000,
            allow_partial=args.allow_partial, show_plots=args.show_plots,
            export_prices=not args.no_price_export, output_dir=args.output_dir,
            raw_sora=args.raw_sora, refresh=args.refresh)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f'Evaluation could not finish: {exc}\n')
    status = report['status'] if isinstance(report.get('status'), dict) else report
    print(f'Status: {status["status"]}')
    if 'reason' in status:
        print(status['reason'])
    if status.get('partial_reason'):
        print(status['partial_reason'])
    if 'metrics' in report:
        shown = pd.concat({'Fixed 0.05%': report['metrics'],
                           'Realistic Costs (Frozen Path)': report['realistic_metrics']},
                          names=['Cost Scenario', 'Strategy'])[['Cumulative Return', 'Annualized Return', 'Sharpe Ratio',
            'Annualized Volatility', 'Maximum Drawdown', 'End of period net value (SGD)']].copy()
        for column in shown:
            shown[column] = shown[column].map(lambda x: f'{x:,.2f}' if column in ['Sharpe Ratio', 'End of period net value (SGD)'] else f'{x:.2%}')
        print(shown.to_string())
        print(f'Actual period: {status["actual_start"]} to {status["actual_end"]}')
    print(f'Output: {status["output_dir"]}')


if __name__ == '__main__':
    cli()
