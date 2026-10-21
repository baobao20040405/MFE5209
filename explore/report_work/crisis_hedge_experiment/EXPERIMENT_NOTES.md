# Crisis sleeve feasibility screen

Source: `explore/crisis_hedge_experiment.py`; frozen product context and bundled
market/SORA inputs. SGD 1,000,000, 2015-09-07 through 2026-07-31. No October
observations or changes to submitted product code. Source hashes are recorded
in `methodology.json`. Four synthetic accounting/timing tests passed.

Entry: VT below its 200-session moving average AND negative 63-session return,
with at least three of five equal-represented asset classes having negative
63-session returns. Correlation variant additionally requires average 60-session
off-diagonal class correlation >=0.30 at entry. Exit when the VT downtrend
condition ceases OR at most one class has a negative 63-session return.
Close-t signals execute at close t+1; changed exposure earns returns from t+2.

Variants: baseline; crisis target 70% baseline fund + cash; same reduction with
10% or 20% short SPY proxy; correlation-filtered 20% short. Between transitions,
exposures drift. Short-sale proceeds and 50% additional margin earn zero interest.
Other free cash earns aligned SORA. Base borrow fee is 3% annually; 1% and 10%
sensitivities are exported. The short total-return proxy already includes dividend
compensation. No extra dividend charge is imposed.

Results: fixed-fee baseline annual return/Sharpe/max DD =16.19%/1.32/-14.98%;
SHORT-20=14.70%/1.21/-15.70%. With realistic core costs, baseline=
14.61%/1.20/-15.44%; SHORT-20=12.81%/1.06/-17.35%.
No tested short variant improves both historical annual return and maximum
drawdown. CASH-30 modestly reduces full-history DD, but reduces returns and
does not improve the historical validation window. Correlation filtering makes
the sleeve much less active (62 versus 477 crisis days) without adding value
over the baseline in this screen.

In the fixed-fee COVID rebound window (2020-03-24 through 2020-06-30), baseline
returns +0.42%, SHORT-20 -5.32%. This demonstrates a real cost of delayed exit;
it does not establish that every possible crisis hedge fails.

Limitations: this is an independent fund-level overlay around a frozen underlying
GMI-ADG path, not a full reconstruction with core signals reacting to overlay NAV.
The SPY SGD total-return short is a proxy, not certified borrow availability,
currency funding or broker execution. Realistic overlay core resizing uses a
0.15% basket cost proxy; SPY trades additionally use lagged liquidity impact.
No forced recall, changing margin requirement or actual stock-loan quotes are
modeled. Do not label it a production-ready short strategy. Historical validation
2025-01-01 through 2026-07-31 has been viewed previously, not untouched holdout.
All predefined variants are retained, including unsuccessful ones; no rule was
retuned after observing the results.
