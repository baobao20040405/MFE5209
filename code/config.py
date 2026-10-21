"""Frozen GMI-ADG model parameters and original plotting style."""
import warnings

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

import yfinance as yf

plt.style.use('seaborn-v0_8-white')

pd.set_option('display.max_columns', 60)

pd.set_option('display.width', 180)

pd.set_option('display.float_format', lambda value: f'{value:.2f}')

ASSET_CLASS_COLORS = {
    'Equities': '#C7253D', 'Bonds': '#2878B8', 'Commodities': '#F18713',
    'FX': '#67CDB4', 'Crypto': '#5C22D4', 'Cash': '#A7ADB4',
    'Trading Friction': '#6B4C3B', 'Other': '#C9CED4',
}

STRATEGY_NAMES = {
    'inverse_vol': 'Global Momentum Inverse Volatility (GMI-IV)',
    'classic_erc': 'Global Momentum Classic Risk Parity (GMI-CRP)',
    'ewma_erc': 'Global Momentum EWMA Risk Parity (GMI-ERP)',
    'downside_erc': 'Global Momentum Downside Risk Parity (GMI-DRP)',
    'equal_weight': 'Global Multi-Asset Equal Weight (GMA-EW)',
    'single_dynamic': 'Global Momentum Dynamic Risk Budget (GMI-DRB)',
    'static_hierarchical': 'Global Momentum Static Hierarchical Risk Budget (GMI-SHRB)',
    'garch_erc': 'Global Momentum GARCH Risk Parity (GMI-GRP)',
    'fully_dynamic': 'Global Momentum Adaptive Hierarchical Risk Budget (GMI-AHRB)',
    'fixed_attack': 'Global Momentum Fixed Offensive Allocation (GMI-FOA)',
    'dynamic_attack': 'Global Momentum Adaptive Defense and Growth (GMI-ADG)',
    'monthly_erc': 'Global Multi-Asset Classic Risk Parity (GMA-CRP)',
}

STRATEGY_COLORS = {
    STRATEGY_NAMES['inverse_vol']: '#45A3B8',
    STRATEGY_NAMES['classic_erc']: '#2A9D8F',
    STRATEGY_NAMES['ewma_erc']: '#5B8FD1',
    STRATEGY_NAMES['downside_erc']: '#78A083',
    STRATEGY_NAMES['equal_weight']: '#333333',
    STRATEGY_NAMES['single_dynamic']: '#E07A5F',
    STRATEGY_NAMES['static_hierarchical']: '#9C89B8',
    STRATEGY_NAMES['garch_erc']: '#7656A8',
    STRATEGY_NAMES['fully_dynamic']: '#3366A3',
    STRATEGY_NAMES['fixed_attack']: '#E6A23C',
    STRATEGY_NAMES['dynamic_attack']: '#9D2235',
    STRATEGY_NAMES['monthly_erc']: '#B07D4F',
}

MARKET_BENCHMARK_COLORS = {
    'ES3.SI': '#287271',
    'VT': '#3D518C',
}

DOWNLOAD_START = '1990-01-01'

DOWNLOAD_END = '2026-10-02'

LOOKBACK = 252

VOL_LOOKBACK = 63

MOMENTUM_SKIP = 21

MOMENTUM_6M = 126

MA_WINDOW = 200

EWMA_LAMBDA = 0.94

TOP_N = 10

COST = 0.0005

INITIAL_CAPITAL_SGD = 1_000_000.0

TRADING_DAYS = 252

IN_SAMPLE_END = pd.Timestamp('2024-12-31')

OOS_START = pd.Timestamp('2025-01-01')

HOLDOUT_START = pd.Timestamp('2026-08-01')

HOLDOUT_END = pd.Timestamp('2026-10-01')

RESEARCH_END = HOLDOUT_START - pd.Timedelta(days=1)

RC_TRIGGER = 0.05

WEIGHT_TRIGGER = 0.03

CHECK_INTERVAL = 15

STRATEGY_NAMES = {key: STRATEGY_NAMES[key] for key in ('dynamic_attack', 'equal_weight')}
STRATEGY_COLORS = {STRATEGY_NAMES[key]: STRATEGY_COLORS[STRATEGY_NAMES[key]] for key in STRATEGY_NAMES}
DYNAMIC_ATTACK_NAME = STRATEGY_NAMES['dynamic_attack']
# Final product (added 2026-10-21): GMI-ADG plus a 20-day short-trend confirmation
# for offensive-sleeve eligibility. GMI-ADG (no confirmation) is kept as the ablation.
STRATEGY_NAMES['dynamic_attack_h20'] = 'Global Momentum Adaptive Trend-Confirmed Allocation (GMI-ATC)'
STRATEGY_COLORS[STRATEGY_NAMES['dynamic_attack_h20']] = '#C51B7D'
DYNAMIC_ATTACK_H20_NAME = STRATEGY_NAMES['dynamic_attack_h20']
ATTACK_SHORT_MA = 20
DYNAMIC_ATTACK_MAX_SHARE = 0.30
DYNAMIC_ATTACK_VOL_SHOCK_MULTIPLIER = 1.50
DYNAMIC_ATTACK_NO_TRADE_BAND = 0.02
DYNAMIC_ATTACK_PARTIAL_EXECUTION = 0.50
VIX_CASH_HIGH = 30.0
VIX_CASH_LOW = 25.0
ATTACK_MAX_ASSETS = 3
ATTACK_CORRELATION_LIMIT = 0.75
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YF_CACHE = PROJECT_ROOT / 'data' / 'cache'
YF_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE))
plt.rcParams['axes.grid'] = False
