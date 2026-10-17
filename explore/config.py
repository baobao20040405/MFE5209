"""Global configuration and dependency.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# Part I. Data, parameters and calibration
#
# 1.1 Parameters, dynamic accommodations and sample division
#
#
# - Data source: Yahoo Finance ' s compound price;
# - Monetary currency: Singapore dollar (ZXQ0QZ);
# - Core asset pool: a total of 18 targets, including Yen FXY and Euro FXE; all ETF except Bitcoin;
# - Top10 main signal: 12-1 momentum; contrast signal: cumulative gain over the past 126 trading days;
# - Collegium window: 252 trading days; EWMA decay factor: 0.94;
# - Check every 15 transaction days; only data up to $t$ are used to signal the date of closing of the $t+1$ transaction and the new weight is effective between the subsequent receipt and the receipt proceeds;
# - ERC trigger: the maximum risk contribution deviation, for example, exceeds 5 percentage points;
# - Other strategy triggers: maximum asset weight deviation from target by more than 3 percentage points;
# - Cost model: formal master measurement applies a uniform unilateral 0.05 per cent; sub-market commissions, swaps, price differentials and slide points, trade-off shocks and withholding taxes are used only as stand-alone realistic cost scenarios;
# - No-risk rate: use of SORA days and one-day delay on available information;
# - Sample endpoint: 2024-12-31; external starting point: 2025-01-01.
# - Double risk budget: use of base risk budget for 40 per cent of equities, 35 per cent of bonds, 15 per cent of commodities, 5 per cent of foreign exchange, 5 per cent of encrypted assets; dynamic distribution of momentum and trend multipliers within categories.
# - GARCH strategy: Project by item for the motor Top10 to be combined with GARCH (1,1) to forecast the differences in the matrix structure of 252 days;
# - All-asset static pool strategy: no assets are screened, all 18 assets are configured and recommenced once a month.
# --- Original Notebook code module 2 ---
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import os
import sys

# Resolve the archived research directory when launched from the project root,
# a notebook frontend, or the command-line runner on another computer.
_research_candidates = []
if '__file__' in globals():
    _research_candidates.append(Path(__file__).resolve().parent)
for _parent in (Path.cwd(), *Path.cwd().parents):
    _research_candidates.extend([_parent, _parent / 'explore'])
PROJECT_ROOT = next((_candidate for _candidate in _research_candidates
                     if (_candidate / 'SORA_daily.csv').is_file()
                     and (_candidate / 'strategies.py').is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError('Cannot locate the research folder. Open the project or explore directory.')
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
import yfinance as yf
from arch import arch_model

plt.style.use('seaborn-v0_8-white')
plt.rcParams['figure.figsize'] = (13, 6)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 180)
pd.set_option('display.float_format', lambda value: f'{value:.2f}')

# The asset class is colored consistently in the ownership and contribution maps.
ASSET_CLASS_COLORS = {
    'Equities': '#C7253D', 'Bonds': '#2878B8', 'Commodities': '#F18713',
    'FX': '#67CDB4', 'Crypto': '#5C22D4', 'Cash': '#A7ADB4',
    'Trading Friction': '#6B4C3B', 'Other': '#C9CED4',
}

# Project-wide uniform product name, colour and chart layout. The policy logic uses the official name and the chart uses short labels.
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
STRATEGY_LABELS = {
    STRATEGY_NAMES['inverse_vol']: 'GMI-IV',
    STRATEGY_NAMES['classic_erc']: 'GMI-CRP',
    STRATEGY_NAMES['ewma_erc']: 'GMI-ERP',
    STRATEGY_NAMES['downside_erc']: 'GMI-DRP',
    STRATEGY_NAMES['equal_weight']: 'GMA-EW',
    STRATEGY_NAMES['single_dynamic']: 'GMI-DRB',
    STRATEGY_NAMES['static_hierarchical']: 'GMI-SHRB',
    STRATEGY_NAMES['garch_erc']: 'GMI-GRP',
    STRATEGY_NAMES['fully_dynamic']: 'GMI-AHRB',
    STRATEGY_NAMES['fixed_attack']: 'GMI-FOA',
    STRATEGY_NAMES['dynamic_attack']: 'GMI-ADG',
    STRATEGY_NAMES['monthly_erc']: 'GMA-CRP',
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
plt.rcParams['axes.grid'] = False


def strategy_label(name):
    """Return compact strategy labels for every reader-facing chart element."""
    label = str(name)
    for full_name, abbreviation in sorted(
        STRATEGY_LABELS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        label = label.replace(full_name, abbreviation)
    return label


def style_axis(ax, axis='both'):
    ax.grid(False)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#AEB6BF')
    ax.spines['bottom'].set_color('#AEB6BF')
    return ax


def place_legend(ax, item_count=None, handles=None, labels=None):
    """Shows single rows when the legend is smaller; four rows at the top of the figure for more items."""
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    labels = [strategy_label(label) for label in labels]
    pairs = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith('_')]
    if not pairs:
        return None
    handles, labels = map(list, zip(*pairs))
    count = len(labels) if item_count is None else item_count
    if count <= 6:
        return ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14),
                         ncol=count, fontsize=9, frameon=False)
    return ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14),
                     ncol=min(4, count), fontsize=8, frameon=False)

YF_CACHE = Path.cwd() / '.yf_cache'
YF_CACHE.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE.resolve()))

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
INITIAL_CAPITAL_SGD = 1_000_000.0  # Initial funding assumed by currency P&L
# Real cost assumptions for manual fixation. The value is a proportion of the amount of the unilateral transaction and does not represent an itemized quote from the history of the holder.
MARKET_COMMISSION_RATE = {
    'United States': 0.00020, 'Germany': 0.00080, 'United Kingdom': 0.00050, 'Japan': 0.00050,
    'China': 0.00030, 'Hong Kong SAR': 0.00080, 'Singapore': 0.00060,
    'Global': 0.00040, 'Euro Area': 0.00040,
}
MARKET_SPREAD_SLIPPAGE_RATE = {
    'United States': 0.00030, 'Germany': 0.00080, 'United Kingdom': 0.00080, 'Japan': 0.00060,
    'China': 0.00100, 'Hong Kong SAR': 0.00080, 'Singapore': 0.00080,
    'Global': 0.00050, 'Euro Area': 0.00050,
}
FX_CONVERSION_RATE = {
    'SGD': 0.0, 'USD': 0.0010, 'EUR': 0.0015, 'GBP': 0.0015,
    'JPY': 0.0015, 'CNY': 0.0020, 'HKD': 0.0015,
}
# Annualized withholding tax drag agent: assumed allocation rate x applied withholding rate; non-distributed asset set to zero.
ANNUAL_WITHHOLDING_TAX_DRAG = {
    'SPY': 0.0036, 'QQQ': 0.0018, 'IWM': 0.0036, 'EXS1.DE': 0.0038,
    'ISF.L': 0.0, '1308.T': 0.0030, '510300.SS': 0.0025,
    '2800.HK': 0.0, 'ES3.SI': 0.0, 'IEF': 0.0105,
    '511010.SS': 0.0020, 'USO': 0.0, 'DBB': 0.0, 'GLD': 0.0,
    'UUP': 0.0015, 'FXY': 0.0015, 'FXE': 0.0015, 'BTC-USD': 0.0,
}
IMPACT_LOOKBACK = 20
IMPACT_COEFFICIENT = 0.10
MAX_IMPACT_RATE = 0.01
TRADING_DAYS = 252
IN_SAMPLE_END = pd.Timestamp('2024-12-31')
OOS_START = pd.Timestamp('2025-01-01')
HOLDOUT_START = pd.Timestamp('2026-08-01')
HOLDOUT_END = pd.Timestamp('2026-10-01')
RESEARCH_END = HOLDOUT_START - pd.Timedelta(days=1)
RC_TRIGGER = 0.05
WEIGHT_TRIGGER = 0.03
CHECK_INTERVAL = 15

print(
    f'Configuration loaded: formal one-way cost={COST:.2%}; '
    f'realistic costs are a separate scenario; Top N={TOP_N}; '
    'all signals execute no earlier than the next trading-day close.'
)

