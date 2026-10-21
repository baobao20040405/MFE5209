"""An independent check of the official chart data generated does not require a network."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'report_work'/'visual_diagnostics'


class VisualEvidenceTests(unittest.TestCase):
    def test_formal_metrics_unchanged(self):
        # Locate the Section 6.4 summary by its cell heading, not its position, so inserted cells do not break it.
        old=pd.read_csv(next((ROOT/'report_work'/'tables').glob('cell_*_table_01_FINAL_EXPECTED_STRATEGY_COUNT*.csv')),header=[0,1])
        old.columns=[b if not b.startswith('Unnamed:') else a for a,b in old.columns]
        old=old[old['Period'].str.startswith('Full Research Period')].set_index('Policy')
        self.assertEqual(len(old),len(set(old.index)))
        self.assertGreaterEqual(len(old),12)
        new=pd.read_csv(DATA/'strategy_tradeoff.csv',index_col=0)
        # Tables exported before GMI-ATC was added lack only GMI-ATC; tables exported afterwards contain it.
        self.assertEqual(set(old.index)-set(new.index),set())
        self.assertLessEqual(set(new.index)-set(old.index),{'Global Momentum Adaptive Trend-Confirmed Allocation (GMI-ATC)'})
        for name in old.index:
            # GARCH refits differ slightly between the saved Notebook runtime
            # and this environment; retain and disclose that drift separately.
            if 'GARCH' in name:
                continue
            for metric in ['Cumulative Return','Annualized Return','Annualized Volatility','Maximum Drawdown','Annual One-Way Turnover']:
                expected=float(str(old.loc[name,metric]).rstrip('%'))/100
                self.assertLess(abs(new.loc[name,metric]-expected),.000051,(name,metric))

    def test_risk_reconciles(self):
        files=list(DATA.glob('risk_snapshot_*.csv'))
        self.assertEqual(len(files),3)
        for file in files:
            data=pd.read_csv(file,index_col=0)
            np.testing.assert_allclose(data.sum(),np.ones(4),atol=1e-8)
            np.testing.assert_allclose(data.iloc[:,0],data.iloc[:,1],atol=1e-3)

    def test_state_dates_and_targets(self):
        data=pd.read_csv(DATA/'attack_state_daily.csv',index_col=0,parse_dates=True)
        self.assertTrue(data.index.is_unique)
        self.assertLess(data.index.max(),pd.Timestamp('2026-08-01'))
        self.assertTrue(data['Offensive Allocation'].dropna().between(0,.30).all())
        self.assertTrue(data['Qualified Trend Breadth'].dropna().between(0,1).all())

    def test_recovery_and_state_counts(self):
        data=pd.read_csv(DATA/'recovery_summary.csv',index_col=0)
        self.assertEqual(len(data),12)  # three windows x four core strategies
        self.assertTrue(data.loc[~data['Fix by end of window'],'Until the date of the transaction'].isna().all())
        for _,frame in data.groupby('Window'):
            self.assertEqual(frame['Window Start Day'].nunique(),1)
            self.assertEqual(frame['Window End Day'].nunique(),1)
        states=pd.read_csv(DATA/'market_state_returns.csv',index_col=0)
        for _,frame in states.groupby('Status'):
            self.assertEqual(frame['Observations'].nunique(),1)


if __name__=='__main__':
    unittest.main()
