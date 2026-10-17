# MFE5209 Global Multi-Asset Adaptive Allocation

## Global Momentum Adaptive Defense and Growth (GMI-ADG)

This project develops an automatically updateable global multi-asset allocation system measured in Singapore dollars (SGD). It downloads ETF, Bitcoin, FX, VIX and benchmark data from Yahoo Finance, processes daily SORA published by the Monetary Authority of Singapore (MAS), and performs data validation, momentum selection, risk estimation, portfolio construction, delayed execution, cost accounting, performance evaluation, stress testing, robustness analysis and attribution.

The submitted product is **Global Momentum Adaptive Defense and Growth (GMI-ADG)**. It selects the ten strongest assets from an 18-asset universe using 12-1 momentum, constructs a core portfolio with an EWMA covariance matrix and a fully adaptive two-level risk budget, and allocates 0%–30% to a diversified offensive sleeve according to trend breadth, volatility shocks, VIX conditions and realized drawdown.

The objective is not simply to maximize historical return. The model balances return, risk-adjusted performance, drawdown, turnover, global diversification and practical tradability. All results are historical research and do not constitute investment advice.

This is the **only project README**. It documents both the complete research archive in `explore/` and the frozen final product in `code/`.

## 1. Final Strategy Logic

### 1.1 SGD-denominated investment universe

| Ticker | Instrument | Class | Market | Currency | Role |
|---|---|---|---|---|---|
| SPY | S&P 500 | Equities | US | USD | US large caps |
| QQQ | Nasdaq 100 | Equities | US | USD | US growth and technology |
| IWM | Russell 2000 | Equities | US | USD | US small caps |
| EXS1.DE | DAX | Equities | Germany | EUR | European equities |
| ISF.L | FTSE 100 | Equities | UK | GBP | UK equities |
| 1308.T | TOPIX | Equities | Japan | JPY | Japanese equities |
| 510300.SS | CSI 300 | Equities | China | CNY | Chinese large caps |
| 2800.HK | Hang Seng Index | Equities | Hong Kong | HKD | Hong Kong equities |
| ES3.SI | Straits Times Index | Equities | Singapore | SGD | Singapore equities |
| IEF | 7–10 Year US Treasury | Bonds | US | USD | Government bonds |
| 511010.SS | China 5-Year Treasury | Bonds | China | CNY | Chinese government bonds |
| USO | Crude Oil | Commodities | Global | USD | Energy |
| DBB | Industrial Metals | Commodities | Global | USD | Industrial metals |
| GLD | Gold | Commodities | Global | USD | Defensive precious metal |
| UUP | US Dollar Index | FX | US | USD | US-dollar exposure |
| FXY | Japanese Yen | FX | Japan | USD | Yen exposure |
| FXE | Euro | FX | Euro area | USD | Euro exposure |
| BTC-USD | Bitcoin | Crypto | Global | USD | Crypto exposure |

Local prices are converted to SGD. The research valuation grid uses global weekdays and forward-fills an already observed price for no more than five business days; it never fills a past gap with a future price. The common sample starts only after all 18 assets have valid prices and then requires 252 valuation days of warm-up. The formal evaluator additionally checks unfilled quotes against exchange calendars.

### 1.2 Top-10 momentum selection

```text
Momentum(i,t) = Price(i,t-21) / Price(i,t-252) - 1
```

- `LOOKBACK = 252`: approximately 12 months;
- `MOMENTUM_SKIP = 21`: excludes the latest month to reduce reversal effects;
- `TOP_N = 10`: selects the ten highest-ranked assets;
- signals read data only through decision-date close; and
- orders execute at the next available valuation close.

New weights earn returns only after execution. The signal-day return is never assigned to a position that did not yet exist.

### 1.3 EWMA risk estimation

```text
Sigma(t) = lambda * Sigma(t-1) + (1-lambda) * r(t-1)r(t-1)'
```

The frozen decay is `EWMA_LAMBDA = 0.94`. EWMA reacts to recent volatility and correlation changes, while matrix symmetrization and regularization improve solver stability. Sample covariance, downside second moments and GARCH(1,1) remain as research comparisons.

### 1.4 Fully adaptive two-level risk budget

Risk is allocated first across equities, bonds, commodities, FX and crypto, and then among securities within each class. Both levels use momentum rank and trend information:

- rank multiplier: `0.75 + 0.50 * rank percentile`;
- positive six-month return and price above MA200: multiplier 1.50;
- only one trend condition satisfied: multiplier 1.00; and
- neither condition satisfied: multiplier 0.50.

The resulting target risk budget is solved against the EWMA covariance matrix. The budget defines desired risk shares; the covariance matrix estimates current risk. They are not interchangeable.

### 1.5 Dynamic 0%–30% offensive sleeve

```text
Final Weight = (1 - Offensive Share) * Adaptive EWMA Core Weight
             + Offensive Share * Diversified Offensive Weight
```

Candidates must be in the Top 10, have positive six-month momentum, trade above MA200, and have 20-day volatility no greater than 1.5 times 126-day volatility.

| Qualified trend breadth | Base offensive share |
|---:|---:|
| Below 30% | 0% |
| 30%–50% | 10% |
| 50%–70% | 20% |
| Above 70% | 30% |

Drawdown at −5%, −8% and −12% caps the sleeve at 20%, 10% and 0%, respectively. VIX above 30 activates crisis mode; VIX below 25 releases it. Crisis mode sets the sleeve to zero. One eligible candidate caps the sleeve at 15%; no candidate sets it to zero.

Candidates are ranked by a 50/50 combination of six-month and 12-1 momentum divided by 63-day annualized volatility. At most three are held, with preference for different asset classes and absolute 60-day pairwise correlation below 0.75.

### 1.6 Turnover and execution

- Review every 15 valuation days.
- Fully implement the new target when Top-10 membership changes.
- When membership is unchanged, trade only 50% of the distance to target.
- Ignore within-set changes below two percentage points.
- Do not impose a universal turnover cap that could block crisis exits.
- Combine core and offensive targets before calculating one account-level net trade.
- Remain long-only; residual cash earns aligned SORA.

## 2. Project Architecture

```text
project/
├── README.md
├── requirements.txt
├── Final_Product.ipynb
├── product_evaluation.py
├── code/                    # frozen final product
├── data/raw/raw sora.csv
├── data/processed/
├── data/cache/
├── output/
├── explore/                 # complete research archive
└── tests/test_regressions.py
```

### 2.1 Complete research workflow: `explore/`

```text
config -> data -> models -> strategies -> reporting -> stress
       -> benchmarks -> robustness -> attribution -> visuals
```

| File | Responsibility | Key output |
|---|---|---|
| `config.py` | Dates, parameters, names, colors and research costs | Shared configuration |
| `data_pipeline.py` | SORA, prices, FX, VIX and benchmarks | `prices_sgd`, `returns` |
| `models.py` | Momentum, EWMA, downside, GARCH, ERC and dynamic budgets | Target-weight functions |
| `strategies.py` | Twelve strategies, delayed execution, holdings and trades | `results`, `strategy_returns` |
| `reporting.py` | Metrics, returns, drawdowns and holdings | Formal figures and tables |
| `stress_tests.py` | Historical/simulated shocks, costs, delay, bootstrap and VaR/ES | Stress evidence |
| `benchmarks.py` | SGD P&L, FIFO trades and market comparisons | Alpha/beta and trade statistics |
| `robustness.py` | Sensitivity, BTC ablation, random Top 10, walk-forward and DSR | Robustness evidence |
| `attribution.py` | Asset, class, module, factor, FX and friction attribution | Four-layer attribution |
| `visual_diagnostics.py` | Mechanism and comparison figures | PNGs and source CSVs |
| `main.py` | Runs the stages in order in one shared context | Complete research context |
| `sync_notebook.py` | Synchronizes modules into the research notebook | Consistent notebook/code |

Dependent modules are not independent entry points. Use `explore/main.py`.

### 2.2 Frozen final product: `code/`

| File | Purpose | Main interfaces |
|---|---|---|
| `main.py` | Instructor-facing formal evaluation | `cli()` |
| `product_evaluation.py` | Loading, replay, comparison, export and safe publication | `evaluate_final_product()`, `evaluate_historical_backtest()` |
| `config.py` | Frozen rules, dates, fee and visual style | Configuration constants |
| `assets.py` | 18-asset metadata | `asset_rows` |
| `data.py` | Raw MAS parser and publication-aware SORA alignment | `process_sora()`, `load_sora_daily()` |
| `download.py` | Request-specific Yahoo cache | `download_market_data()` |
| `market_data.py` | SGD prices, returns, FX, VIX and benchmarks | Shared market context |
| `validation.py` | Observed quote and exchange-session checks | `validate_quote_coverage()` |
| `strategy.py` | Momentum, EWMA, budgets and offensive rules | Signal and allocation functions |
| `backtest.py` | Historical replay and next-close execution | `run_dynamic_backtest()`, `run_dynamic_attack_backtest()` |
| `evaluation.py` | Metrics, drawdowns and FIFO ledger | `compute_metrics()`, `build_fifo_trade_ledger()` |
| `plots.py` | Returns, wealth, drawdown and allocation charts | `plot_comparison()` |
| `export.py` | 18-asset closing-price export | `export_asset_closing_prices()` |

The two public entry points intentionally differ:

- `python code/backtest.py`: fixed historical product backtest, SGD 1,000,000, 2015-09-07 to 2026-07-31;
- `python code/main.py`: user-specified formal out-of-sample evaluation.

Historical results go to `output/historical_backtest/` and never overwrite formal output.

## 3. Frozen Parameters

### 3.1 Data, signal and execution

| Parameter | Value | Meaning |
|---|---:|---|
| `DOWNLOAD_START` | 1990-01-01 | Earliest requested Yahoo history |
| `LOOKBACK` | 252 | Momentum/covariance window |
| `VOL_LOOKBACK` | 63 | Standard volatility window |
| `MOMENTUM_SKIP` | 21 | 12-1 skip period |
| `MOMENTUM_6M` | 126 | Six-month trend window |
| `MA_WINDOW` | 200 | Long-term moving average |
| `EWMA_LAMBDA` | 0.94 | EWMA decay |
| `TOP_N` | 10 | Selected assets |
| `CHECK_INTERVAL` | 15 | Review interval |
| `RC_TRIGGER` | 5% | Risk-contribution trigger |
| `WEIGHT_TRIGGER` | 3% | Weight-deviation trigger |
| `INITIAL_CAPITAL_SGD` | 1,000,000 | Fixed historical capital |
| `COST` | 0.0005 | Formal one-way fee |

### 3.2 Final-strategy controls

| Parameter | Value | Purpose |
|---|---:|---|
| `DYNAMIC_ATTACK_MAX_SHARE` | 30% | Maximum offensive allocation |
| `DYNAMIC_ATTACK_VOL_SHOCK_MULTIPLIER` | 1.50 | Volatility-shock threshold |
| `DYNAMIC_ATTACK_NO_TRADE_BAND` | 2% | Within-set no-trade band |
| `DYNAMIC_ATTACK_PARTIAL_EXECUTION` | 50% | Partial adjustment rate |
| `ATTACK_MAX_ASSETS` | 3 | Maximum offensive assets |
| `ATTACK_CORRELATION_LIMIT` | 0.75 | Pairwise correlation limit |
| `VIX_CASH_HIGH` | 30 | Enter crisis state |
| `VIX_CASH_LOW` | 25 | Exit crisis state |

### 3.3 Research and evaluation dates

| Period | Date | Purpose |
|---|---|---|
| In-sample end | 2024-12-31 | Initial research |
| Historical validation | 2025-01-01 to 2026-07-31 | Forward validation evidence |
| Frozen research end | 2026-07-31 | End of completed research |
| Formal course evaluation | 2026-10-01 to 2026-10-31 | Instructor evaluation |

Historical validation was inspected during development and is not described as an untouched final holdout.

## 4. Formal and Realistic Transaction Costs

All formal strategies use a fixed **0.05% one-way fee on actual executed notional**. The research archive also contains a separate realistic-cost scenario for GMI-ADG; it never overwrites formal results.

| Market | Commission/fees | Spread/slippage |
|---|---:|---:|
| US | 0.020% | 0.030% |
| Germany | 0.080% | 0.080% |
| UK | 0.050% | 0.080% |
| Japan | 0.050% | 0.060% |
| Mainland China | 0.030% | 0.100% |
| Hong Kong | 0.080% | 0.080% |
| Singapore | 0.060% | 0.080% |
| Global/euro-area proxy | 0.040% | 0.050% |
| Bitcoin override | 0.150% | 0.150% |

| Currency | One-way FX cost |
|---|---:|
| SGD | 0.000% |
| USD | 0.100% |
| EUR, GBP, JPY, HKD | 0.150% |
| CNY | 0.200% |

These are transparent research assumptions, not broker quotes. The FX scenario conservatively assumes each non-SGD trade requires conversion and does not model foreign-currency cash netting.

Dynamic impact is modeled as:

```text
Impact(i,t) = min(1%, 0.10 * Vol20(i,t-1) * sqrt(OrderValue(i,t) / ADV20(i,t-1)))
```

All inputs are lagged. The scenario also applies an asset-level annual withholding-tax drag proxy. Its daily order of operations is: pre-return holdings and cash return, tax drag, executed commission/FX/spread/impact, compounded net return, then updated closing holdings.

## 5. Important Functions

- `load_sora_daily()` / `align_sora_returns()`: load and align publication-available SORA.
- `momentum_12_1()` / `select_top10()`: calculate momentum and select assets.
- `ewma_covariance()`: estimate annualized EWMA covariance.
- `strict_risk_budget()`: solve long-only fully invested risk-budget weights.
- `fully_dynamic_class_budget()` / `fully_dynamic_budget()`: allocate risk across and within classes.
- `dynamic_attack_state()`: determine the offensive share.
- `select_dynamic_attack_assets()`: impose ranking, class and correlation rules.
- `run_dynamic_attack_backtest()`: replay holdings, reviews, next-close orders and costs.
- `apply_realistic_cost_model()`: create the independent realistic-cost result.
- `compute_metrics()`: calculate return, volatility, Sharpe, Sortino, drawdown, Calmar and turnover.
- `build_fifo_trade_ledger()`: reconstruct post-fee quantities and FIFO matches, including inherited opening holdings.

### Final-product API

```python
from product_evaluation import evaluate_final_product

report = evaluate_final_product(
    start='2026-10-01',
    end='2026-10-31',
    initial_capital=10_000,
    allow_partial=False,
    show_plots=False,
    export_prices=True,
    output_dir=None,
    raw_sora=None,
    refresh=False,
)
```

| Argument | Meaning |
|---|---|
| `start`, `end` | Measurement dates; they do not retune rules |
| `initial_capital` | Positive finite SGD capital |
| `allow_partial` | Explicitly permit a labelled incomplete result |
| `show_plots` | Display chart windows after saving |
| `export_prices` | Export all 18 assets' close files |
| `output_dir` | Optional separate output directory |
| `raw_sora` | Optional replacement MAS raw file |
| `refresh` | Bypass compatible Yahoo caches |
| `strict_quote_coverage` | Internal control; keep `True` for formal out-of-sample evaluation. The fixed historical wrapper alone uses the disclosed research fill convention. |

The product replays all earlier holdings, pending orders, review timing, drawdown state and VIX state before measuring the requested interval. It therefore evaluates investment in an operating strategy rather than restarting from cash.

`evaluate_historical_backtest()` accepts no date or capital parameters and runs the frozen historical product comparison.

## 6. Twelve Research Strategies

| Acronym | Full name | Role |
|---|---|---|
| GMI-IV | Global Momentum Inverse Volatility | Risk-input comparison |
| GMI-CRP | Global Momentum Classic Risk Parity | Classic risk parity |
| GMI-ERP | Global Momentum EWMA Risk Parity | EWMA comparison |
| GMI-DRP | Global Momentum Downside Risk Parity | Downside-risk comparison |
| GMA-EW | Global Multi-Asset Equal Weight | Main 18-asset benchmark |
| GMI-DRB | Global Momentum Dynamic Risk Budget | Single-level ablation |
| GMI-SHRB | Global Momentum Static Hierarchical Risk Budget | Hierarchy ablation |
| GMI-GRP | Global Momentum GARCH Risk Parity | GARCH comparison |
| GMI-AHRB | Global Momentum Adaptive Hierarchical Risk Budget | Final core |
| GMI-FOA | Global Momentum Fixed Offensive Allocation | Fixed offensive comparison |
| **GMI-ADG** | **Global Momentum Adaptive Defense and Growth** | **Final submitted strategy** |
| GMA-CRP | Global Multi-Asset Classic Risk Parity | Full-universe risk parity |

Charts and tables use acronyms; the report can introduce each full name once.

## 7. Timing and Look-Ahead Protection

```text
data through close t
  -> calculate signals, covariance, budgets and target
  -> execute at the next valuation close
  -> new holdings earn subsequent interval returns
```

Signal dates must precede execution dates. Momentum, MA200, EWMA, correlation, volume and impact use only lagged information. Evaluation inherits true opening holdings rather than resetting to cash. Returns use pre-return holdings; closing weights reflect completed trades. Formal evaluation checks observed Close for all 18 assets, VT, VIX/VIX3M, VIX Open and FX proxies against exchange sessions. Interior gaps fail rather than being hidden by valuation filling.

Historical research retains the documented maximum five-business-day fill convention for isolated provider gaps, and the historical status file explicitly discloses this difference.

## 8. Metrics, Stress Tests, Robustness and Attribution

### Metrics

- cumulative return: compounded daily net return;
- annualized return: 252-day compounding;
- volatility: sample daily standard deviation times sqrt(252);
- Sharpe: annualized daily return in excess of aligned SORA;
- Sortino: excess return relative to downside deviation;
- maximum drawdown: worst fall from running NAV peak;
- Calmar: annualized return divided by absolute drawdown;
- positive-day ratio: profitable days, not trade win rate;
- trade win rate: profitable completed FIFO matches;
- two-way turnover: sum of absolute weight changes; and
- SGD P&L: profit/loss from stated capital.

FIFO quantities use post-fee NAV and closing weights. Recorded fees are allocated across executed legs rather than charged twice. Fee-funding-only sales remain for reconciliation but are excluded from ordinary completed-trade statistics.

### Stress testing

The archive includes historical crisis replay, interpretable joint shocks, static current-holding shocks, one-day execution delay, 0.5x–2.0x cost sensitivity, ordinary and crisis-weighted block bootstrap, reverse stress tests, VaR and expected shortfall. Stress tests diagnose risk and do not reselect parameters. Bootstrap reorders realized net returns and is a conditional distribution test, not an exact future-price simulation.

### Robustness and overfitting

Tests cover Top N, momentum horizon, EWMA lambda, review interval, BTC ablation, momentum versus random Top 10, model-layer ablation, annual walk-forward performance, multi-factor alpha, Deflated Sharpe Ratio and realistic-cost sensitivity. Parameter neighborhoods test stability; they are not used to select a new ex-post optimum.

### Four-layer attribution

1. asset and asset-class contribution;
2. momentum, EWMA, dynamic budget and offensive-module contribution, including Shapley analysis;
3. market/factor contribution and residual alpha; and
4. gross-to-net commission, FX, spread/slippage, impact and tax friction.

Reconciliation assertions link contribution results to portfolio returns. Factor regression describes exposure and association, not causal proof of alpha.

## 9. Raw SORA Processing

Download daily SORA from [MAS Domestic Interest Rates](https://eservices.mas.gov.sg/statistics/dir/) and save the unedited export as:

```text
data/raw/raw sora.csv
```

Do not remove MAS headings or footnotes. Process it with:

```powershell
python code/main.py --process-sora
```

`process_sora()` identifies the value/publication-date table, reconstructs grouped date labels, validates annual percentage rates and publication dates, and writes processed data and audits. `1.50` means 1.50% per year. Cash accrual uses:

```text
SORA / 100 * actual calendar days / 365
```

The product selects the latest rate published by the return date. It never backfills unpublished SORA or silently substitutes a fixed risk-free rate. A rate older than seven calendar days fails formal evaluation.

## 10. Installation and Running

Use 64-bit CPython 3.10:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Fixed historical product backtest

```powershell
python code/backtest.py
```

Fixed capital: SGD 1,000,000. Fixed period: 2015-09-07 to 2026-07-31. Output: `output/historical_backtest/`.

### Formal instructor evaluation

```powershell
python code/main.py
```

or:

```powershell
python code/main.py --start 2026-10-01 --end 2026-10-31 --capital 10000
```

Options: `--allow-partial`, `--show-plots`, `--refresh`, `--no-price-export`, and `--output-dir output/my_run`. An unfinished interval returns `PENDING` unless a partial result is explicitly permitted. Future observations are never manufactured.

### Complete research

```powershell
python explore/main.py --through visuals
```

Available stopping stages include `data`, `strategies`, `reporting`, `stress`, `benchmarks`, `robustness`, `attribution` and `visuals`.

### Notebooks

`Final_Product.ipynb` is the short product demonstration. The long notebook under `explore/` contains the full research narrative and numbered Markdown sections.

```powershell
python -m ipykernel install --user --name gmi-project --display-name "GMI Project"
python -m nbconvert --execute --to notebook --inplace `
  --ExecutePreprocessor.kernel_name=gmi-project `
  --ExecutePreprocessor.timeout=1200 Final_Product.ipynb
```

## 11. Outputs and Safe Publication

Formal figures are `growth.png`, `wealth.png`, `drawdown.png` and `allocation.png`. Tables contain metrics, daily returns, NAV, P&L, drawdowns, weights, FIFO matches, trade legs, costs, SORA alignment and quote coverage. CSVs retain calculation precision; charts and console output use two decimals.

The 18-asset export includes asset metadata, unadjusted local closes, adjusted local closes, adjusted SGD closes and data notes. Holidays and pre-listing dates remain missing. Weekend Bitcoin uses preceding FX, never a future rate. VT is an extra benchmark, not a nineteenth strategy asset.

Outputs are built in a temporary sibling folder and published only after every required artifact succeeds. Failed and pending runs do not relabel earlier successful results. Always inspect `evaluation_status.json` for actual dates, status, assumptions and source hashes.

## 12. Reproduction Checklist

- [ ] 18 formal assets and 12 formal strategies.
- [ ] GMI-ADG is the submitted strategy.
- [ ] Research ends on 2026-07-31.
- [ ] October evaluation uses the independent final-product path.
- [ ] SORA loads without a fixed-rate fallback.
- [ ] Every signal date precedes execution.
- [ ] Holdings and cash reconcile to 100%.
- [ ] Top-10 membership, holdings and trades reconcile.
- [ ] Formal cost equals 0.05% of actual executed notional.
- [ ] Gross return, tax, costs and net return reconcile.
- [ ] Historical and formal outputs remain separate.
- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] Both notebooks execute without errors before submission.

## 13. Known Limitations

- Yahoo data are not institutional tick-level execution data.
- Cost assumptions are proxies, not broker-specific historical fills.
- Square-root impact cannot reproduce every algorithm or order book.
- FX cash-account netting is not modeled.
- Tax depends on investor identity, domicile and treaties.
- Asynchronous international closes can distort short-horizon relationships.
- Bitcoin shortens the common sample and may affect early results.
- A short formal test cannot prove persistent alpha.
- Multi-strategy research retains data-mining risk.
- Historical, stress and bootstrap results do not predict future returns.

## 14. Current Validation Status

The project has been checked by direct execution:

- fixed `code/backtest.py` completed for 2015-09-07 to 2026-07-31;
- formal `code/main.py` completed its currently available interim path;
- 26 regression tests passed;
- all historical tables and four figures were generated; and
- successful product outputs include current source hashes.

As of 17 October 2026, the requested 1–31 October interval is unfinished. Any currently available formal result is labelled `PARTIAL` and ends on the latest common date supported by observed quotes. After month-end, update `data/raw/raw sora.csv`, use `--refresh`, and require a non-partial completion before reporting final October performance.

## 15. Updating the Formal Evaluation Period

Do not change research parameters or the frozen historical dates to evaluate a new formal period. Pass the new dates to the product entry point:

```powershell
python code/main.py --start 2026-10-01 --end 2026-11-30 --capital 10000 --refresh
```

The Yahoo end date is handled internally. The evaluator reloads the full warm-up history and replays the strategy state before measuring the requested dates. Do not extend an old chart, splice future returns onto an old NAV series, or reset the strategy to cash at the new start date.

Before running a completed period:

1. download an updated unedited MAS SORA CSV;
2. replace `data/raw/raw sora.csv` while retaining its exact filename;
3. use `--refresh` to request current market data;
4. do not use `--allow-partial` for the final submitted measurement;
5. confirm `evaluation_status.json` says `COMPLETED` and contains the requested actual end date; and
6. inspect `observed_quote_coverage.csv` before quoting results.

If the period has not ended, the program returns `PENDING`. If an explicitly requested interim observation is shortened because some markets have not published their latest quote, it returns `PARTIAL` and records the reason. Neither status may be presented as a completed monthly result.

## 16. Research Notebook Synchronization and Visual Diagnostics

The long research notebook is a readable record of the modular `explore/` source. Strategy logic should not be edited independently in two places. After changing research modules, use the archive's synchronization utility and then execute the notebook from a clean kernel. The final product notebook is separate and must also be rerun after product changes.

The research visual-diagnostics stage is read-only: it does not alter signals, trades, costs or returns. It produces the research map, seven-step construction path, return/drawdown/turnover comparison, annual heatmaps, offensive-state diagnostics, capital-versus-risk comparisons, risk-budget snapshots, regime-conditional results and common drawdown-recovery windows. Outputs and their source CSVs are stored under `explore/report_work/visual_diagnostics/`.

Risk snapshots compare target risk budgets, estimated risk contribution and actual capital weight using the covariance information available on the signal date. Offensive share is the model's executed target state and is not misrepresented as an independently funded sub-account. Market regimes are descriptive ex-post groupings, not additional trading signals.

To export report materials after executing and saving the research notebook:

```powershell
python explore/extract_report_materials.py
```

This populates `explore/report_work/assets/` and `explore/report_work/tables/`. Word-generation scripts are optional editorial tools and are not backtest entry points.

## 17. Portability and Troubleshooting

- **Missing module:** install `requirements.txt` with the same Python executable used for the script or notebook kernel.
- **Yahoo returns no data:** verify internet/proxy access and provider availability; use `--refresh` only when a live request is intended.
- **Missing or stale SORA:** replace `data/raw/raw sora.csv`, then run `python code/main.py --process-sora`.
- **PENDING result:** the requested period has not finished; this is an evaluation state, not a calculated performance result.
- **No plot window:** PNG files are saved regardless. `--show-plots` requires a graphical desktop.
- **Different fonts:** Matplotlib can fall back to DejaVu Sans on another computer without changing calculations.
- **Read-only directory:** move the project to a writable folder because processed data, caches and outputs are created locally.
- **A direct internal-module launch:** use `code/main.py`, `code/backtest.py`, or `explore/main.py`; other modules are dependencies rather than user entry points.

All operational paths are resolved relative to the project files. No author-specific username, drive letter or Conda environment is required. Relative `--output-dir` and `--raw-sora` arguments are interpreted from the terminal's current directory. Existing JSON files may record the machine that generated them, but those paths are audit information rather than runtime dependencies.

The Windows portability check used a clean 64-bit CPython 3.10 environment, installed only the root requirements, parsed the raw MAS export, ran the final product, generated all four figures and the 18-asset export, executed the final notebook, and ran the complete research workflow. This supports reproducibility on the tested environment; it does not guarantee every operating system, future provider response or optional editorial utility.
