import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from realistic_costs import cost_schedule, replay_costs


class RealisticCostTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2026-01-01', periods=30)
        self.window = self.dates[-2:]
        self.prices = pd.DataFrame({'ES3.SI': np.linspace(100, 110, 30)}, index=self.dates)
        raw = pd.concat({'Volume': pd.DataFrame({'ES3.SI': 100000.}, index=self.dates)}, axis=1)
        self.context = {'prices_sgd': self.prices, 'benchmark_prices_sgd':
                        pd.DataFrame({'VT': 100.}, index=self.dates), 'raw': raw}
        self.schedule = cost_schedule(pd.DataFrame({'Market': ['Singapore'],
                    'Local Currency': ['SGD']}, index=['ES3.SI']))
        self.result = {'name': 'STI', 'returns': pd.Series([-.0005, 0.], index=self.window),
            'transaction_cost': pd.Series([.0005, 0.], index=self.window),
            'trade_deltas': pd.DataFrame({'ES3.SI': [1., 0.]}, index=self.window),
            'pre_return_weights': pd.DataFrame({'ES3.SI': [0., 1.]}, index=self.window),
            'end_of_day_weights': pd.DataFrame({'ES3.SI': 1.}, index=self.window)}

    def test_fee_replacement_and_no_trade(self):
        result, costs = replay_costs(self.result, self.context, self.window, 10000, self.schedule)
        self.assertAlmostEqual(costs.iloc[0]['Commission'], .0006)
        self.assertAlmostEqual(costs.iloc[0]['Spread and Slippage'], .0008)
        self.assertAlmostEqual(result['returns'].iloc[0], -costs.iloc[0]['Transaction Cost'])
        self.assertEqual(costs.iloc[1].sum(), 0)
        self.assertAlmostEqual(self.result['returns'].iloc[0], -.0005)

    def test_larger_capital_increases_impact(self):
        _, small = replay_costs(self.result, self.context, self.window, 10000, self.schedule)
        _, large = replay_costs(self.result, self.context, self.window, 1000000, self.schedule)
        self.assertGreater(large.iloc[0]['Market Impact'], small.iloc[0]['Market Impact'])

    def test_missing_liquidity_rejected(self):
        self.context['raw'].loc[:, ('Volume', 'ES3.SI')] = np.nan
        with self.assertRaises(ValueError):
            replay_costs(self.result, self.context, self.window, 10000, self.schedule)


if __name__ == '__main__':
    unittest.main()
