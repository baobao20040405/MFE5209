"""Accounting and timing checks for isolated crisis research."""
import unittest
import numpy as np
import pandas as pd
from crisis_hedge_experiment import simulate


class CrisisSleeveTests(unittest.TestCase):
    def inputs(self):
        dates = pd.bdate_range('2020-01-01', periods=6)
        return (pd.Series(0., index=dates), pd.Series(0., index=dates),
            pd.Series(0., index=dates), pd.DataFrame({'broad': [True] * 6}, index=dates),
            pd.DataFrame({'ADV': 1e8, 'Volatility': .02}, index=dates))

    def test_no_hedge_reconciles_baseline(self):
        base, spy, rf, signals, liquidity = self.inputs()
        base.iloc[:] = [.01, -.02, .03, -.01, 0., .005]
        frame = simulate(base, spy, rf, signals, 'none', 1., 0.,
                         'Fixed 0.05%', .03, liquidity)
        np.testing.assert_allclose(frame['Return'], base, atol=1e-12)

    def test_execution_lag_and_short_sign(self):
        base, spy, rf, signals, liquidity = self.inputs()
        spy.iloc[1:3] = -.10
        frame = simulate(base, spy, rf, signals, 'broad', .7, .2,
                         'Fixed 0.05%', 0., liquidity)
        self.assertEqual(frame['Short Weight'].iloc[1], 0.)
        self.assertGreater(frame['Short Weight'].iloc[2], .19)
        self.assertGreater(frame['Return'].iloc[2], .019)
        self.assertLess(frame['Return'].iloc[1], 0.)  # entry fee, no same-day short profit

    def test_exit_and_borrow(self):
        base, spy, rf, signals, liquidity = self.inputs()
        signals.iloc[3:] = False
        free = simulate(base, spy, rf, signals, 'broad', .7, .2,
                        'Fixed 0.05%', 0., liquidity)
        costly = simulate(base, spy, rf, signals, 'broad', .7, .2,
                          'Fixed 0.05%', .10, liquidity)
        self.assertLess(costly['NAV'].iloc[-1], free['NAV'].iloc[-1])
        self.assertAlmostEqual(costly['Short Weight'].iloc[-1], 0.)

    def test_future_signal_does_not_change_prefix(self):
        base, spy, rf, signals, liquidity = self.inputs()
        first = simulate(base, spy, rf, signals, 'broad', .7, .2,
                         'Fixed 0.05%', .03, liquidity)
        signals.iloc[-2:] = False
        second = simulate(base, spy, rf, signals, 'broad', .7, .2,
                          'Fixed 0.05%', .03, liquidity)
        np.testing.assert_allclose(first['NAV'].iloc[:4], second['NAV'].iloc[:4])


if __name__ == '__main__':
    unittest.main()
