"""Regression checks; use bundled snapshots and write only temporary outputs.

Run: python -m unittest discover -s tests -v
"""
import hashlib
import json
import importlib.util
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import product_evaluation as product
from export import export_asset_closing_prices
from validation import validate_quote_coverage, QuoteCoverageError


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context, cls.sora = product.load_product_context('2026-10-16', 10000)

    def test_fifo_round_trip_and_post_fee_quantity(self):
        dates = pd.bdate_range('2026-01-05', periods=3)
        result = {'name': 'flat round trip', 'returns': pd.Series([0, -.0005, -.0005], index=dates),
            'pre_return_weights': pd.DataFrame({'SPY': [0., 0., 1.]}, index=dates),
            'end_of_day_weights': pd.DataFrame({'SPY': [0., 1., 0.]}, index=dates),
            'trade_deltas': pd.DataFrame({'SPY': [0., 1., -1.]}, index=dates),
            'trade_assets': ['SPY'], 'asset_cost_rate': .0005,
            'instrument_prices': pd.DataFrame({'SPY': 100.}, index=pd.bdate_range('2026-01-02', periods=4))}
        ledger, diagnostics = self.context['build_fifo_trade_ledger'](result, dates[0], dates[-1], 1000)
        self.assertAlmostEqual(ledger['Match share'].sum(), 9.995)
        self.assertAlmostEqual(ledger['Net profit (SGD)'].sum(), 1000 * ((1 + result['returns']).prod() - 1))
        self.assertEqual(diagnostics['Unsettled assets at end of period'], 0)
        self.assertLess(diagnostics['Maximum FIFO holding reconciliation error (SGD)'], 1e-7)

    def test_holiday_does_not_clip_valid_market_date(self):
        # 1 May is a Singapore holiday; latest available SORA publication is 30 April.
        sora = self.sora.loc[:'2026-05-01']
        self.assertEqual(sora.index.max(), pd.Timestamp('2026-04-30'))
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp, patch.object(product, 'load_product_context', return_value=(self.context, sora)):
            report = product.evaluate_final_product('2026-04-30', '2026-05-01', output_dir=tmp, export_prices=False)
        self.assertEqual(report['status']['actual_end'], '2026-05-01')

    def test_export_failure_preserves_entire_previous_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            out = Path(tmp) / 'output'
            for name in ['tables', 'figures']:
                (out / name).mkdir(parents=True)
                (out / name / 'old.txt').write_text('previous successful run')
            (out / 'evaluation_status.json').write_text('{"status":"COMPLETED"}')
            old = hashes(out)
            with patch.object(product, 'load_product_context', return_value=(self.context, self.sora)), patch.object(product, 'export_asset_closing_prices', side_effect=RuntimeError('simulated failure')):
                with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                    product.evaluate_final_product('2026-10-01', '2026-10-15', output_dir=out)
            self.assertEqual(old, hashes(out))

    def test_publish_failure_rolls_back(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            out, stage = Path(tmp) / 'output', Path(tmp) / 'stage'
            for folder, content in [(out, 'old'), (stage, 'new')]:
                folder.mkdir()
                for name in ['tables', 'figures']:
                    (folder / name).mkdir()
                    (folder / name / 'data.txt').write_text(content)
                (folder / 'evaluation_status.json').write_text(content)
            old = hashes(out)
            replace, calls = product.os.replace, []
            def fail_once(src, dest):
                calls.append(str(src))
                if len(calls) == 5:
                    raise OSError('simulated publish failure')
                return replace(src, dest)
            with patch.object(product.os, 'replace', side_effect=fail_once):
                with self.assertRaises(OSError):
                    product._publish_outputs(stage, out)
            self.assertEqual(old, hashes(out))

    def test_real_holdings_and_fee_accounting(self):
        result = self.context['run_dynamic_attack_backtest'](end_date='2026-10-16', sora_frame=self.sora, allow_holdout=True)
        before = result['pre_return_weights']
        gross = (before * result['instrument_returns']).sum(axis=1) + (1 - before.sum(axis=1)) * result['risk_free_daily']
        fee = result['transaction_cost'].reindex(gross.index, fill_value=0)
        np.testing.assert_allclose(result['returns'], (1 + gross) * (1 - fee) - 1, atol=1e-12)
        np.testing.assert_allclose(fee.reindex(result['trade_deltas'].index), result['trade_deltas'].abs().sum(axis=1) * .0005, atol=1e-12)
        for start in ['2026-10-01', '2026-07-01']:
            ledger, diagnostics = self.context['build_fifo_trade_ledger'](result, start, '2026-10-16', 10000)
            self.assertLess(diagnostics['Maximum FIFO holding reconciliation error (SGD)'], 1e-6)
        shorter = self.context['run_dynamic_attack_backtest'](end_date='2026-07-31', sora_frame=self.sora, allow_holdout=True)
        np.testing.assert_allclose(shorter['returns'], result['returns'].loc[shorter['returns'].index], atol=1e-12)

    def test_h20_final_product_rules_and_no_lookahead(self):
        """GMI-ATC: same accounting as GMI-ADG, MA20 rule respected, no future data used."""
        run = self.context['run_dynamic_attack_h20_backtest']
        result = run(end_date='2026-10-16', sora_frame=self.sora, allow_holdout=True)
        before = result['pre_return_weights']
        gross = (before * result['instrument_returns']).sum(axis=1) + (1 - before.sum(axis=1)) * result['risk_free_daily']
        fee = result['transaction_cost'].reindex(gross.index, fill_value=0)
        np.testing.assert_allclose(result['returns'], (1 + gross) * (1 - fee) - 1, atol=1e-12)
        triggers = result['triggers']
        self.assertTrue((triggers['Signal Date'] < triggers['Execution Date']).all())
        prices = self.context['prices_sgd']
        window = self.context['ATTACK_SHORT_MA']
        self.assertEqual(window, 20)
        for _, row in triggers.iterrows():
            for ticker in [t for t in row['Attack assets'].split(',') if t]:
                history = prices[ticker].loc[:row['Signal Date']]
                self.assertGreater(history.iloc[-1], history.iloc[-window:].mean())
        shorter = run(end_date='2026-07-31', sora_frame=self.sora, allow_holdout=True)
        np.testing.assert_allclose(shorter['returns'], result['returns'].loc[shorter['returns'].index], atol=1e-12)
        # Replace every return/VIX observation after the cutoff with noise: earlier output must not move.
        cutoff = pd.Timestamp('2024-08-01')
        context = self.context
        saved = {key: context[key] for key in ['returns', 'portfolio_returns_for_backtest', 'vix_open_daily']}
        try:
            rng = np.random.default_rng(0)
            for key in ['returns', 'portfolio_returns_for_backtest']:
                frame = saved[key].copy()
                late = frame.index > cutoff
                frame.loc[late] = rng.normal(0, 0.03, (late.sum(), frame.shape[1]))
                context[key] = frame
            vix = saved['vix_open_daily'].copy(); vix.loc[vix.index > cutoff] = 60.0
            context['vix_open_daily'] = vix
            perturbed = run(end_date='2026-10-16', sora_frame=self.sora, allow_holdout=True)
        finally:
            context.update(saved)
        np.testing.assert_array_equal(perturbed['returns'].loc[:cutoff].values, result['returns'].loc[:cutoff].values)
        np.testing.assert_array_equal(perturbed['end_of_day_weights'].loc[:cutoff].values,
                                      result['end_of_day_weights'].loc[:cutoff].values)

    def test_h20_option_off_reproduces_gmi_adg(self):
        base = self.context['run_dynamic_attack_backtest'](end_date='2026-10-16', sora_frame=self.sora, allow_holdout=True)
        explicit = self.context['run_dynamic_attack_backtest'](end_date='2026-10-16', sora_frame=self.sora,
                                                              allow_holdout=True, short_trend_window=None)
        np.testing.assert_array_equal(base['returns'].values, explicit['returns'].values)

    def test_no_price_export_still_rejects_stale_completion(self):
        context = dict(self.context)
        context['raw'] = context['raw'].copy()
        context['raw'].loc['2026-10-15':, ('Close', 'SPY')] = np.nan
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp, patch.object(product, 'load_product_context', return_value=(context, self.sora)):
            with self.assertRaises(QuoteCoverageError):
                product.evaluate_final_product('2026-10-01', '2026-10-15', output_dir=tmp, export_prices=False)

    def test_explicit_partial_uses_last_observed_common_day(self):
        context = dict(self.context)
        context['raw'] = context['raw'].copy()
        context['raw'].loc['2026-10-15':, ('Close', 'SPY')] = np.nan
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp, patch.object(product, 'load_product_context', return_value=(context, self.sora)):
            result = product.evaluate_final_product('2026-10-01', '2026-10-15', initial_capital=10000,
                output_dir=tmp, export_prices=False, allow_partial=True)
            self.assertEqual(result['status']['status'], 'PARTIAL')
            self.assertEqual(result['status']['actual_end'], '2026-10-14')
            self.assertIn('partial_reason', result['status'])
            audit = pd.read_csv(Path(tmp) / 'tables/observed_quote_coverage.csv')
            self.assertEqual(audit['Missing Sessions'].sum(), 0)

    def test_pending_request_preserves_previous_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            output = Path(tmp)
            (output / 'evaluation_status.json').write_text('{"status":"COMPLETED"}')
            future = pd.Timestamp.now().normalize() + pd.Timedelta(days=365)
            result = product.evaluate_final_product(str(future.date()), str((future + pd.Timedelta(days=10)).date()), output_dir=output)
            self.assertEqual(result['status'], 'PENDING')
            self.assertEqual(json.loads((output / 'evaluation_status.json').read_text())['status'], 'COMPLETED')


class OutputSafetyTests(unittest.TestCase):
    def test_double_failure_keeps_recoverable_old_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            out, stage = Path(tmp) / 'out', Path(tmp) / 'stage'
            for folder, content in [(out, 'old'), (stage, 'new')]:
                folder.mkdir()
                for name in ['tables', 'figures']:
                    (folder / name).mkdir()
                    (folder / name / 'data.txt').write_text(content)
                (folder / 'evaluation_status.json').write_text(content)
            original = hashes(out)
            replace, calls = product.os.replace, []
            def fail_twice(src, dest):
                calls.append(str(src))
                if len(calls) in [5, 6]:
                    raise OSError('simulated file lock')
                return replace(src, dest)
            with patch.object(product.os, 'replace', side_effect=fail_twice):
                with self.assertRaisesRegex(RuntimeError, 'retained at') as error:
                    product._publish_outputs(stage, out)
            backups = list(Path(tmp).glob('.previous-run-*'))
            self.assertEqual(len(backups), 1)
            backup = backups[0]
            self.assertIn(str(backup), str(error.exception))
            self.assertFalse((out / 'evaluation_status.json').exists())
            # Every old file remains recoverable, either restored or retained.
            for name, digest in original.items():
                candidates = [folder / name for folder in [out, backup]]
                self.assertTrue(any(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in candidates), name)

    def price_context(self, observed, downloaded):
        class FakeYahoo:
            @staticmethod
            def download(*args, **kwargs):
                return pd.concat({'Close': downloaded}, axis=1)
        return {'pd': pd, 'ALL_ASSETS': list(observed), 'close_local': observed,
            'yf': FakeYahoo, 'fx_to_sgd': pd.DataFrame(index=observed.index),
            'assets': pd.DataFrame({'Local Currency': 'SGD'}, index=observed.columns)}

    def test_price_export_rejects_truncated_download(self):
        prices = pd.DataFrame(100., index=pd.bdate_range('2026-10-01', '2026-10-16'), columns=[f'A{i}' for i in range(18)])
        context = self.price_context(prices, prices.iloc[:2])
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                export_asset_closing_prices(context, tmp)
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_price_export_rejects_interior_gap(self):
        prices = pd.DataFrame(100., index=pd.bdate_range('2026-10-01', periods=4), columns=[f'A{i}' for i in range(18)])
        raw = prices.copy(); raw.iloc[1, 0] = np.nan
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                export_asset_closing_prices(self.price_context(prices, raw), tmp)

    def test_price_export_preserves_market_holidays(self):
        prices = pd.DataFrame(100., index=pd.bdate_range('2026-10-01', periods=4), columns=[f'A{i}' for i in range(18)])
        prices.iloc[-1, 0] = np.nan
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            export_asset_closing_prices(self.price_context(prices, prices.copy()), tmp)
            actual = pd.read_csv(Path(tmp) / 'daily_closing_prices_local_currency.csv', index_col=0)
            self.assertTrue(pd.isna(actual.iloc[-1, 0]))
            notes = json.loads((Path(tmp) / 'price_data_notes.json').read_text())
            self.assertEqual(notes['per_asset_coverage']['A0']['observations'], 3)
            self.assertEqual(notes['per_asset_coverage']['A0']['last_observation'], str(prices.index[-2].date()))

    def test_price_export_rejects_missing_fx(self):
        prices = pd.DataFrame(100., index=pd.bdate_range('2026-10-01', periods=4), columns=[f'A{i}' for i in range(18)])
        context = self.price_context(prices, prices.copy())
        context['assets'].iloc[0, 0] = 'USD'
        context['fx_to_sgd']['USD'] = np.nan
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            with self.assertRaisesRegex(RuntimeError, 'SGD conversion'):
                export_asset_closing_prices(context, tmp)

    def test_weekend_crypto_uses_prior_fx_not_future_rate(self):
        dates = pd.date_range('2026-10-09', '2026-10-12')
        prices = pd.DataFrame(100., index=dates, columns=[f'A{i}' for i in range(18)])
        context = self.price_context(prices, prices.copy())
        context['assets'].iloc[0, 0] = 'USD'
        context['fx_to_sgd'] = pd.DataFrame({'USD': [1.3, 1.4]}, index=dates[[0, 3]])
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            export_asset_closing_prices(context, tmp)
            converted = pd.read_csv(Path(tmp) / 'daily_adjusted_closing_prices_sgd.csv', index_col=0)
            np.testing.assert_allclose(converted['A0'], [130., 130., 130., 140.])

    def test_early_history_kept_without_fabricating_old_fx(self):
        prices = pd.DataFrame(100., index=pd.bdate_range('2026-10-01', periods=4), columns=[f'A{i}' for i in range(18)])
        context = self.price_context(prices, prices.copy())
        context['assets'].iloc[0, 0] = 'USD'
        context['fx_to_sgd']['USD'] = [np.nan, np.nan, 1.3, 1.3]
        context['portfolio_prices_sgd'] = prices.iloc[2:]
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            export_asset_closing_prices(context, tmp)
            converted = pd.read_csv(Path(tmp) / 'daily_adjusted_closing_prices_sgd.csv', index_col=0)
            self.assertTrue(converted['A0'].iloc[:2].isna().all())
            self.assertTrue(converted['A0'].iloc[2:].notna().all())


class CalendarAndEntryTests(unittest.TestCase):
    def context(self):
        dates = pd.date_range('2026-04-25', '2026-05-08')
        assets = ['SPY', 'ES3.SI', '510300.SS', 'BTC-USD']
        tickers = assets + ['VT', '^VIX', '^VIX3M']
        close = pd.DataFrame(100., index=dates, columns=tickers)
        raw = pd.concat({'Close': close, 'Open': close.copy()}, axis=1)
        return {'raw': raw, 'ALL_ASSETS': assets}

    def test_real_exchange_holidays_are_allowed(self):
        context = self.context()
        for ticker in ['ES3.SI', '510300.SS']:
            context['raw'].loc['2026-05-01', ('Close', ticker)] = np.nan
        audit = validate_quote_coverage(context, '2026-04-30', '2026-05-01')
        self.assertEqual(int(audit['Missing Sessions'].sum()), 0)

    def test_missing_open_exchange_session_is_rejected(self):
        context = self.context()
        context['raw'].loc['2026-05-01':, ('Close', 'SPY')] = np.nan
        with self.assertRaises(QuoteCoverageError) as error:
            validate_quote_coverage(context, '2026-04-30', '2026-05-08')
        self.assertEqual(error.exception.safe_end, pd.Timestamp('2026-04-30'))

    def test_interior_gap_cannot_be_hidden_by_partial_cutoff(self):
        context = self.context()
        context['raw'].loc['2026-05-01', ('Close', 'SPY')] = np.nan
        with self.assertRaises(QuoteCoverageError) as error:
            validate_quote_coverage(context, '2026-04-30', '2026-05-08')
        self.assertIsNone(error.exception.safe_end)

    def test_crypto_weekend_observations_are_required(self):
        context = self.context()
        context['raw'].loc['2026-05-02', ('Close', 'BTC-USD')] = np.nan
        with self.assertRaisesRegex(QuoteCoverageError, 'BTC-USD'):
            validate_quote_coverage(context, '2026-04-30', '2026-05-08')

    def test_vix_open_input_is_checked(self):
        context = self.context()
        context['raw'].loc['2026-05-01', ('Open', '^VIX')] = np.nan
        with self.assertRaisesRegex(QuoteCoverageError, 'VIX Open'):
            validate_quote_coverage(context, '2026-04-30', '2026-05-08')

    def test_missing_fx_quote_is_not_hidden_by_filled_conversion(self):
        context = self.context()
        context['FX_DOWNLOAD'] = ['SGD=X']
        context['raw'][('Close', 'SGD=X')] = 1.3
        context['raw'].loc['2026-05-01', ('Close', 'SGD=X')] = np.nan
        with self.assertRaisesRegex(QuoteCoverageError, 'SGD=X'):
            validate_quote_coverage(context, '2026-04-30', '2026-05-08')

    def test_research_entry_delegates_to_current_product(self):
        spec = importlib.util.spec_from_file_location('_research_entry_test', ROOT / 'explore/product_evaluation.py')
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        self.assertEqual(Path(bridge.evaluate_final_product.__code__.co_filename).resolve(), ROOT / 'code/product_evaluation.py')
        self.assertEqual(Path(bridge.export_asset_closing_prices.__code__.co_filename).resolve(), ROOT / 'code/export.py')

    def test_supported_python_entry_points_launch_directly(self):
        entries = [ROOT / 'code/main.py', ROOT / 'product_evaluation.py']
        args = ['--start', '2099-01-01', '--end', '2099-01-31', '--capital', '10000']
        for path in entries:
            with self.subTest(entry=path.name):
                run = subprocess.run(
                    [sys.executable, str(path), *args], cwd=ROOT,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn('Status: PENDING', run.stdout)

    def test_internal_stages_explain_direct_launch(self):
        expected = {
            ROOT / 'code/market_data.py': 'internal data-preparation stage',
        }
        for path, message in expected.items():
            with self.subTest(entry=path.name):
                run = subprocess.run(
                    [sys.executable, str(path)], cwd=ROOT,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(message, run.stdout)

    def test_direct_backtest_runs_fixed_historical_period(self):
        run = subprocess.run(
            [sys.executable, str(ROOT / 'code/backtest.py')], cwd=ROOT,
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('Historical period: 2015-09-07 to 2026-07-31', run.stdout)
        self.assertIn('Initial capital: SGD 1,000,000.00', run.stdout)

    def test_all_python_and_notebook_cells_compile(self):
        import nbformat
        from IPython.core.inputtransformer2 import TransformerManager
        files = [*list((ROOT / 'code').glob('*.py')), *list((ROOT / 'explore').glob('*.py')),
                 ROOT / 'product_evaluation.py']
        for path in files:
            with self.subTest(source=str(path)):
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        transformer = TransformerManager()
        for path in [ROOT / 'Final_Product.ipynb', *list((ROOT / 'explore').glob('*.ipynb'))]:
            notebook = nbformat.read(path, 4)
            nbformat.validate(notebook)
            for number, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    with self.subTest(notebook=path.name, cell=number):
                        compile(transformer.transform_cell(cell.source), f'{path}:{number}', 'exec')


if __name__ == '__main__':
    unittest.main()
