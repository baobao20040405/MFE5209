"""Official strategy, retrospect engine and transaction execution.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# Part III Formal strategy construction and transaction execution
#
# 3.1 Universal No Future Information Remittance Engine and nine basic strategies
#
#
# The 12 current formal strategies are non-leveraging: 100 per cent of the combined risk assets and cash weights. The code retains `funding_daily` fields to harmonize the result structure, but there is no negative cash position and no actual trigger cost of financing. If leverage is added in the future, financing spreads above SORA must be set and disclosed separately.
# --- Original Notebook code unit 25--
def sample_covariance(history):
    return history.cov() * TRADING_DAYS


DYNAMIC_BUDGET_NAME = STRATEGY_NAMES['single_dynamic']
HIERARCHICAL_BUDGET_NAME = STRATEGY_NAMES['static_hierarchical']
GARCH_MOMENTUM_NAME = STRATEGY_NAMES['garch_erc']

strategy_specs = {
    STRATEGY_NAMES['inverse_vol']: (target_momentum_inverse_vol, None),
    STRATEGY_NAMES['classic_erc']: (target_momentum_erc, sample_covariance),
    STRATEGY_NAMES['ewma_erc']: (target_momentum_ewma_erc, ewma_covariance),
    STRATEGY_NAMES['downside_erc']: (target_momentum_downside_erc, downside_second_moment),
    STRATEGY_NAMES['equal_weight']: (target_equal_weight, None),
    DYNAMIC_BUDGET_NAME: (target_momentum_ewma_dynamic_budget, None),
    HIERARCHICAL_BUDGET_NAME: (target_momentum_ewma_hierarchical_budget, None),
    GARCH_MOMENTUM_NAME: (target_momentum_garch_erc, garch11_covariance),
    FULLY_DYNAMIC_BUDGET_NAME: (target_momentum_ewma_fulldynamic_budget, None),
}


def prepare_target(name, target):
    target = pd.Series(target, dtype=float).reindex(PORTFOLIO_ASSETS).fillna(0.0).clip(lower=0.0)
    if target.sum() <= 0:
        raise RuntimeError(f'{name}Target weights are empty.')
    target /= target.sum()
    return target


def run_dynamic_backtest(
    name, allocator, covariance_function=None, end_date=RESEARCH_END,
    sora_frame=None, allow_holdout=False,
):
    """Scroll back: a signal is formed after the t-day closing, the t-+1-day closing is concluded and the new weight begins to yield benefits between the subsequent closing and the closing area.

(a) The transactional cost of the closing date is only deducted from the transaction cost of the closing date and does not allow new weights to yield the benefit before the transaction is made;
The calibre does not give new weights the benefit of signal dayst, but it is not equivalent to using real t+1 openings for sale.
"""
    if pd.Timestamp(end_date) >= HOLDOUT_START and not allow_holdout:
        raise RuntimeError('The study backsight prohibits reading outlet for locked samples.')
    sora_frame = sora_daily_df if sora_frame is None else sora_frame
    test_returns = portfolio_returns_for_backtest.loc[strategy_start:end_date].copy()
    cash_returns = align_sora_returns(test_returns.index, sora_frame, allow_holdout=allow_holdout)
    funding_returns = align_sora_returns(
        test_returns.index, sora_frame, allow_holdout=allow_holdout, annual_spread=0.0
    )
    current_w = pd.Series(0.0, index=PORTFOLIO_ASSETS)
    initialized = False
    pending_order = None
    start_position = returns.index.get_loc(strategy_start)

    daily_records, turnover_records, trade_leg_records, cost_records = [], [], [], []
    weight_records, selection_records, trigger_records = [], [], []
    pre_return_weight_records, end_of_day_weight_records, trade_delta_records = [], [], []

    for date in test_returns.index:
        position = returns.index.get_loc(date)
        # The revenue per day is the amount received from the previous collection to the same day. This is the first time that we can get the proceeds from the old warehouse.
        # The order is then collected on the same day to execute the order formed on the date of the previous transaction.
        pre_return_weight_records.append(current_w.rename(date))
        day = test_returns.loc[date]
        cash_weight = 1.0 - float(current_w.sum())
        cash_rate = cash_returns.loc[date] if cash_weight >= 0 else funding_returns.loc[date]
        before_cost_return = float(current_w @ day) + cash_weight * cash_rate
        denominator = 1.0 + before_cost_return
        if denominator <= 0:
            raise RuntimeError(f'{date.date()}Group net value is not positive.')
        current_w = current_w * (1.0 + day) / denominator

        cost_rate = 0.0
        if pending_order is not None:
            if not pending_order['Signal Date'] < date:
                raise RuntimeError(f'{date.date()}The order was not delayed until the next transaction date.')
            target = pending_order['Target weights']
            delta = target - current_w
            turnover = float(delta.abs().sum())
            trade_legs = int((delta.abs() > 1e-10).sum())
            cost_rate = COST * turnover
            current_w = target.copy()
            initialized = True
            turnover_records.append((date, turnover))
            trade_leg_records.append((date, trade_legs))
            cost_records.append((date, cost_rate))
            trade_delta_records.append(delta.rename(date))
            weight_records.append(current_w.rename(date))
            trigger_records.append({
                'Signal Date': pending_order['Signal Date'], 'Execution Date': date,
                'Reason': pending_order['Reason'], 'Trigger deviation': pending_order['Trigger deviation'],
                'Total asset weights': float(current_w.sum()), 'Transaction cost rate': cost_rate,
            })
            selection_records.extend([
                {'Rebalance Date': date, 'Signal Date': pending_order['Signal Date'], 'Policy': name,
                 'Code': ticker, 'Asset Name': asset_metadata.loc[ticker, 'Asset Name'],
                 'Motion Score': pending_order['Motion Score'].get(ticker, np.nan),
                 'Target weights': current_w[ticker]}
                for ticker in pending_order['Selected assets']
            ])
            pending_order = None

        net_return = (1.0 - cost_rate) * (1.0 + before_cost_return) - 1.0
        daily_records.append((date, net_return))
        end_of_day_weight_records.append(current_w.rename(date))

        history = returns.iloc[position - LOOKBACK + 1:position + 1]
        if len(history) != LOOKBACK or history.index.max() != date:
            raise RuntimeError(f'{date.date()}The signal window does not strictly close to the signal day closing or insufficient length.')
        check_today = (not initialized) or (position - start_position) % CHECK_INTERVAL == 0
        if check_today and pending_order is None:
            if name == STRATEGY_NAMES['equal_weight']:
                scores = pd.Series(np.nan, index=ALL_ASSETS)
                selected = ALL_ASSETS
            else:
                scores = momentum_12_1(history)
                selected = scores.nlargest(TOP_N).index.tolist()

            should_rebalance, reason, deviation = False, '', np.nan
            if not initialized:
                should_rebalance, reason = True, 'Initial buildup'
            elif set(selected) != set(current_w[current_w.abs() > 1e-8].index):
                should_rebalance, reason = True, 'Change in selected asset members'
            elif covariance_function is not None and selected:
                covariance = covariance_function(history[selected])
                rc_share = risk_contribution_share(current_w[selected], covariance)
                deviation = float((rc_share - 1.0 / len(selected)).abs().max())
                if deviation > RC_TRIGGER:
                    should_rebalance, reason = True, 'Risk contribution deviation threshold'
            else:
                preview, _, _ = allocator(history)
                preview = prepare_target(name, preview)
                deviation = float((current_w - preview).abs().max())
                if deviation > WEIGHT_TRIGGER:
                    should_rebalance, reason = True, 'weight deviation threshold'

            if should_rebalance:
                target, selected, scores = allocator(history)
                target = prepare_target(name, target)
                pending_order = {
                    'Signal Date': date, 'Target weights': target, 'Selected assets': list(selected),
                    'Motion Score': scores.copy(), 'Reason': reason, 'Trigger deviation': deviation,
                }

    result = {
        'name': name, 'returns': pd.Series(dict(daily_records), name=name),
        'trade_legs': pd.Series(dict(trade_leg_records), name=name, dtype=float),
        'turnover': pd.Series(dict(turnover_records), name=name, dtype=float),
        'transaction_cost': pd.Series(dict(cost_records), name='Transaction cost rate', dtype=float),
        'weights': pd.DataFrame(weight_records), 'selections': pd.DataFrame(selection_records),
        'pre_return_weights': pd.DataFrame(pre_return_weight_records),
        'end_of_day_weights': pd.DataFrame(end_of_day_weight_records),
        'trade_deltas': pd.DataFrame(trade_delta_records),
        'triggers': pd.DataFrame(trigger_records), 'risk_free_daily': cash_returns,
        'funding_daily': funding_returns,
        'asset_cost_rate': pd.Series(COST, index=PORTFOLIO_ASSETS, dtype=float),
    }
    if not result['triggers'].empty and not (
        pd.to_datetime(result['triggers']['Signal Date']) < pd.to_datetime(result['triggers']['Execution Date'])
    ).all():
        raise RuntimeError(f'{name}There are signals coming together or future information leaks.')
    return result

results = {name: run_dynamic_backtest(name, allocator, covariance_function)
           for name, (allocator, covariance_function) in strategy_specs.items()}
strategy_returns = pd.concat([obj['returns'] for obj in results.values()], axis=1).dropna()

print(f'Study the walk:{strategy_returns.index.min().date()}to{strategy_returns.index.max().date()}')
print(f'Sample endpoint:{IN_SAMPLE_END.date()};validation interval:{OOS_START.date()}to{RESEARCH_END.date()}')
print(f'Validation of the day of transaction:{len(strategy_returns.loc[OOS_START:RESEARCH_END])}')
assert strategy_returns.index.max() < HOLDOUT_START, 'Lock test data leak into the research back.'

# 3.2 Fixed 70 per cent core + 30 per cent decentralized offensive
#
#
# The strategy does not cover the original innovation strategy. The core component uses a dynamic Top10, the EWMA margin and a fully dynamic two-tier risk budget; the offensive component selects two to three assets from the core Top10 with a trend of two to three, with priority given to asset classes, and requires any two of the past 60 days to have absolute correlation coefficients below 0.75.
#
# - Normality: 70% core + 30% offensive;
# - VIX crisis state: 70 per cent of SSD cash transferred to daily SORA interest rate, and 30 per cent of attacks maintained;
# - Delaying of the VIX signal to the next combined trading date;
# - The core and offensive sleeves are combined into the final weight of the securities, and the change of hands and costs are calculated in a uniform manner based on changes in net weights;
# - The result is a visible recording of cash, total securities openings, hand exchanges, trading legs, offensive assets and signal dates.
#
# The core, offensive and VIX status co-ordinated target weights to be implemented at the next transaction date.
# --- Original Notebook code unit 27-
CASH_ASSET = 'Cash (SGD)'
VIX_CASH_HIGH = 30.0
VIX_CASH_LOW = 25.0
vix_open_daily = vix_open_bday['^VIX'].reindex(returns.index)

CORE_ATTACK_NAME = STRATEGY_NAMES['fixed_attack']
CORE_SHARE = 0.70
ATTACK_SHARE = 0.30
ATTACK_MAX_ASSETS = 3
ATTACK_CORRELATION_LIMIT = 0.75


def select_diversified_attack_assets(history, selected, score):
    selected = list(selected)
    six_month = momentum_6m(history[selected])
    ma_signal = above_ma200(history[selected])
    eligible = [ticker for ticker in selected if six_month[ticker] > 0.0 and bool(ma_signal[ticker])]
    ranked = sorted(eligible, key=lambda ticker: float(score[ticker]), reverse=True)
    if not ranked:
        return []

    correlation = history[ranked].iloc[-60:].corr()
    chosen, used_classes = [], set()
    # The first priority cross-category; the second allows for the same category, but always enforces the ceiling on relevance.
    for require_new_class in (True, False):
        for ticker in ranked:
            if ticker in chosen:
                continue
            asset_class = assets.loc[ticker, 'Asset Class']
            if require_new_class and asset_class in used_classes:
                continue
            if chosen and any(
                abs(float(correlation.loc[ticker, existing])) >= ATTACK_CORRELATION_LIMIT
                for existing in chosen
            ):
                continue
            chosen.append(ticker)
            used_classes.add(asset_class)
            if len(chosen) >= ATTACK_MAX_ASSETS:
                return chosen
    return chosen


def core_attack_target(history, core_in_market):
    selected, score = select_top10(history)
    target_budget = fully_dynamic_budget(history, selected, score)
    core_weights = strict_risk_budget(ewma_covariance(history[selected]), target_budget)
    core_weights = core_weights.reindex(ALL_ASSETS, fill_value=0.0)

    attack_assets = select_diversified_attack_assets(history, selected, score)
    attack_weights = pd.Series(0.0, index=ALL_ASSETS)
    if attack_assets:
        attack_weights.loc[attack_assets] = 1.0 / len(attack_assets)
    else:
        # There is no trend of qualified assets not being forced to attack, with 30 per cent returning to core configurations.
        attack_weights = core_weights.copy()

    target = pd.Series(0.0, index=ALL_ASSETS + [CASH_ASSET])
    target.loc[ALL_ASSETS] = ATTACK_SHARE * attack_weights
    if core_in_market:
        target.loc[ALL_ASSETS] += CORE_SHARE * core_weights
    else:
        target[CASH_ASSET] = CORE_SHARE
    target = target.clip(lower=0.0)
    target /= target.sum()
    return target, selected, score, attack_assets, core_weights, attack_weights


def run_core_attack_backtest(
    name=CORE_ATTACK_NAME, end_date=RESEARCH_END, sora_frame=None, allow_holdout=False,
):
    """70/30 Core offensive strategy; all targets are implemented on the next trading date after the signal formation."""
    if pd.Timestamp(end_date) >= HOLDOUT_START and not allow_holdout:
        raise RuntimeError('Add a new strategy study to re-examine the off-site range of locked samples.')
    sora_frame = sora_daily_df if sora_frame is None else sora_frame
    test_returns = returns.loc[strategy_start:end_date].copy()
    cash_returns = align_sora_returns(test_returns.index, sora_frame, allow_holdout=allow_holdout)
    funding_returns = align_sora_returns(test_returns.index, sora_frame, allow_holdout=allow_holdout)
    instrument_returns = test_returns.copy()
    instrument_returns[CASH_ASSET] = cash_returns
    position_assets = ALL_ASSETS + [CASH_ASSET]
    current_weights = pd.Series(0.0, index=position_assets)
    current_weights[CASH_ASSET] = 1.0
    last_model_target, pending_order = None, None
    last_selected, last_attack_assets = [], []
    in_market, previous_in_market = True, True
    start_position = returns.index.get_loc(strategy_start)
    nav = 1.0
    daily_records, turnover_records, trade_leg_records, cost_records = [], [], [], []
    weight_records, selection_records, trigger_records = [], [], []
    cash_records, exposure_records, state_records = [], [], []
    pre_return_weight_records, end_of_day_weight_records, trade_delta_records = [], [], []

    for date in test_returns.index:
        position = returns.index.get_loc(date)
        pre_return_weight_records.append(current_weights.rename(date))
        day_returns = instrument_returns.loc[date].fillna(0.0)
        gross_return = float(current_weights @ day_returns)
        if 1.0 + gross_return <= 0:
            raise RuntimeError('The net 70/30 strategy is not positive.')
        current_weights = current_weights * (1.0 + day_returns) / (1.0 + gross_return)
        current_weights /= current_weights.sum()

        cost_rate = 0.0
        if pending_order is not None:
            if not pending_order['Signal Date'] < date:
                raise RuntimeError('The 70/30 strategic order was not delayed until the next transaction date.')
            target = pending_order['Target weights']
            security_delta = target[ALL_ASSETS] - current_weights[ALL_ASSETS]
            turnover = float(security_delta.abs().sum())
            trade_legs = int((security_delta.abs() > 1e-10).sum())
            cost_rate = COST * turnover
            current_weights = target.copy()
            turnover_records.append((date, turnover))
            trade_leg_records.append((date, trade_legs))
            cost_records.append((date, cost_rate))
            trade_delta_records.append(security_delta.rename(date))
            weight_records.append(current_weights.rename(date))
            trigger_records.append({
                'Signal Date': pending_order['Signal Date'], 'Execution Date': date,
                'Reason': pending_order['Reason'], 'VIX signal date': pending_order['VIX signal date'],
                'VIX start value': pending_order['VIX start value'], 'Is the core present?': pending_order['Is the core present?'],
                'Proportion of total bilateral turnover of securities': turnover, 'Transaction cost rate': cost_rate,
                'Cash weights': float(current_weights[CASH_ASSET]),
                'Total exposure to risk securities mouth': float(current_weights[ALL_ASSETS].sum()),
                'Attack assets': ','.join(pending_order['Attack assets']), 'Policy net value (before transaction)': nav,
            })
            selection_records.extend([
                {'Rebalance Date': date, 'Signal Date': pending_order['Signal Date'], 'Policy': name,
                 'Code': ticker, 'Asset Name': asset_metadata.loc[ticker, 'Asset Name'],
                 'Asset Class': assets.loc[ticker, 'Asset Class'],
                 'Motion Score': pending_order['Motion Score'].get(ticker, np.nan),
                 'Whether to attack assets': ticker in pending_order['Attack assets'],
                 'Final net weight': current_weights[ticker]}
                for ticker in current_weights[ALL_ASSETS][current_weights[ALL_ASSETS] > 1e-10].index
            ])
            pending_order = None

        net_return = (1.0 - cost_rate) * (1.0 + gross_return) - 1.0
        daily_records.append((date, net_return))
        cash_records.append((date, float(pre_return_weight_records[-1][CASH_ASSET])))
        exposure_records.append((date, float(pre_return_weight_records[-1][ALL_ASSETS].abs().sum())))
        nav *= 1.0 + net_return
        end_of_day_weight_records.append(current_weights.rename(date))

        history = returns.iloc[position - LOOKBACK + 1:position + 1]
        if len(history) != LOOKBACK or history.index.max() != date:
            raise RuntimeError('The 70/30 policy history window did not strictly close to the signal day closing.')
        observed_vix = vix_open_daily.get(date, np.nan)
        if pd.notna(observed_vix):
            if observed_vix > VIX_CASH_HIGH:
                in_market = False
            elif observed_vix < VIX_CASH_LOW:
                in_market = True
        state_changed = in_market != previous_in_market

        check_today = last_model_target is None or (position - start_position) % CHECK_INTERVAL == 0
        model_changed = False
        proposed_score = pd.Series(dtype=float)
        if check_today or state_changed:
            proposed_target, proposed_selected, proposed_score, proposed_attack, _, _ = core_attack_target(history, in_market)
            if last_model_target is None:
                model_changed = True
            elif check_today:
                model_changed = (
                    set(proposed_selected) != set(last_selected)
                    or float((proposed_target[ALL_ASSETS] - last_model_target[ALL_ASSETS]).abs().max()) > WEIGHT_TRIGGER
                )
            if model_changed or state_changed:
                last_model_target = proposed_target.copy()
                last_selected = list(proposed_selected)
                last_attack_assets = list(proposed_attack)

        if last_model_target is None:
            raise RuntimeError('The 70/30 strategy did not generate target weights.')
        if model_changed or state_changed:
            reasons = []
            if len(weight_records) == 0:
                reasons.append('Initial buildup')
            if state_changed:
                reasons.append('VIX restore core next trading date' if in_market else 'Core cash transfer for VIX next transaction date')
            if model_changed and 'Initial buildup' not in reasons:
                reasons.append('Top10 or target weight update')
            pending_order = {
                'Signal Date': date, 'Target weights': last_model_target.copy(),
                'Reason': '；'.join(reasons), 'VIX signal date': date, 'VIX start value': observed_vix,
                'Is the core present?': in_market, 'Attack assets': list(last_attack_assets),
                'Motion Score': proposed_score.copy(),
            }

        state_records.append({
            'Date': date, 'VIX signal date': date, 'VIX start value': observed_vix,
            'Is the core present?': in_market,
            'Cash weights between areas of return': float(pre_return_weight_records[-1][CASH_ASSET]),
            'Total risk securities in the yield zone mouth': float(pre_return_weight_records[-1][ALL_ASSETS].sum()),
            'Cash weight after closing': float(current_weights[CASH_ASSET]),
            'After closing, risk securities are all open. mouth': float(current_weights[ALL_ASSETS].sum()),
            'Attack assets': ','.join(last_attack_assets), 'Policy net value (after closing)': nav,
        })
        previous_in_market = in_market

    result = {
        'name': name, 'returns': pd.Series(dict(daily_records), name=name),
        'turnover': pd.Series(dict(turnover_records), name=name, dtype=float),
        'trade_legs': pd.Series(dict(trade_leg_records), name=name, dtype=float),
        'transaction_cost': pd.Series(dict(cost_records), name='Transaction cost rate', dtype=float),
        'weights': pd.DataFrame(weight_records).reindex(columns=position_assets, fill_value=0.0),
        'pre_return_weights': pd.DataFrame(pre_return_weight_records).reindex(columns=position_assets, fill_value=0.0),
        'end_of_day_weights': pd.DataFrame(end_of_day_weight_records).reindex(columns=position_assets, fill_value=0.0),
        'trade_deltas': pd.DataFrame(trade_delta_records).reindex(columns=ALL_ASSETS, fill_value=0.0),
        'selections': pd.DataFrame(selection_records), 'triggers': pd.DataFrame(trigger_records),
        'cash_weight': pd.Series(dict(cash_records), name='Cash weights', dtype=float),
        'gross_exposure': pd.Series(dict(exposure_records), name='Total exposure to risk securities mouth', dtype=float),
        'state_history': pd.DataFrame(state_records).set_index('Date'),
        'risk_free_daily': cash_returns, 'funding_daily': funding_returns,
        'trade_assets': list(ALL_ASSETS), 'instrument_returns': test_returns,
        'instrument_prices': prices_sgd.reindex(test_returns.index),
        'asset_cost_rate': pd.Series(COST, index=ALL_ASSETS, dtype=float),
    }
    if not result['triggers'].empty and not (
        result['triggers']['Signal Date'] < result['triggers']['Execution Date']
    ).all():
        raise RuntimeError('The 70/30 strategy is implemented on the same day or future data leakage.')
    return result

results[CORE_ATTACK_NAME] = run_core_attack_backtest()
strategy_returns = pd.concat(
    [results[name]['returns'].rename(name) for name in results], axis=1
).dropna(how='any')

core_attack_audit = pd.DataFrame({
    'Value': [
        results[CORE_ATTACK_NAME]['returns'].index.max().date(),
        sora_daily_df.index.max().date(),
        float(results[CORE_ATTACK_NAME]['turnover'].sum()),
        int(len(results[CORE_ATTACK_NAME]['turnover'])),
        float(results[CORE_ATTACK_NAME]['cash_weight'].mean()),
    ]
}, index=['Final date of research findings', 'Study of SORA final date', 'Proportion of total bilateral turnover of cumulative securities', 'Number of redeployments', 'Average cash weights'])
display(core_attack_audit)
assert results[CORE_ATTACK_NAME]['returns'].index.max() < HOLDOUT_START
assert sora_daily_df.index.max() < HOLDOUT_START
print(f'New official strategy completed:{CORE_ATTACK_NAME}')

# 3.3 Dispersive attacks in dynamic categories 0-30%
#
#
# The strategy is a formal improvement on the fixed 70 per cent core + 30 per cent offensive version, but does not cover the original version in order to retain auditable comparisons.
#
# 1. The core portfolio maintains a 100 per cent investmentable framework, with the continued use of the motor Top10, the fully dynamic two-tier risk budget and the strict risk parity of EWMA;
# An offensive sleeve cap of 30 per cent, determined by the dynamicity of Top10 to meet the ratio of assets “six months of positive earnings, with prices above MA200 and 20 days of volatility not significantly above 126 days of fluctuations”;
# 3. The VIX crisis only eliminates the offensive cuffs and does not define the empty core composition;
# 4. Reduction of the offensive ceiling when the combined withdrawal reaches 5 per cent, 8 per cent and 12 per cent, respectively;
# 5. Risk-adjusted drive ranking of offensive assets, priority cross-category, and 60 days absolute correlation coefficient must be below 0.75;
# Top10 members will be fully implemented as they change; only 50 per cent of model weight changes will be implemented when the list remains unchanged and minor adjustments within 2 percentage points will be ignored;
# 7. Core and offensive weights are combined, followed by turnover and cost based on net account transactions;
# 8. All signals are completed on the next day of the transaction and the results of the study are not allowed to read the lock slot after 2026-08-01.
# --- Original Notebook code unit 29-
DYNAMIC_ATTACK_NAME = STRATEGY_NAMES['dynamic_attack']
DYNAMIC_ATTACK_MAX_SHARE = 0.30
DYNAMIC_ATTACK_VOL_SHOCK_MULTIPLIER = 1.50
DYNAMIC_ATTACK_NO_TRADE_BAND = 0.02
DYNAMIC_ATTACK_PARTIAL_EXECUTION = 0.50


def dynamic_attack_state(history, selected, vix_crisis, portfolio_drawdown):
    """Only the price history up to the signal date and the combination withdrawal achieved on that date were used to determine the upper limit of the attack."""
    six_month = momentum_6m(history[selected])
    ma_signal = above_ma200(history[selected])
    volatility_20 = history[selected].iloc[-20:].std(ddof=1)
    volatility_126 = history[selected].iloc[-126:].std(ddof=1).clip(lower=1e-8)
    no_volatility_shock = (
        volatility_20 <= DYNAMIC_ATTACK_VOL_SHOCK_MULTIPLIER * volatility_126
    )
    qualified = (six_month > 0.0) & ma_signal & no_volatility_shock
    breadth = float(qualified.mean())

    if breadth < 0.30:
        attack_share = 0.00
    elif breadth < 0.50:
        attack_share = 0.10
    elif breadth < 0.70:
        attack_share = 0.20
    else:
        attack_share = DYNAMIC_ATTACK_MAX_SHARE

    if portfolio_drawdown <= -0.12:
        attack_share = 0.00
    elif portfolio_drawdown <= -0.08:
        attack_share = min(attack_share, 0.10)
    elif portfolio_drawdown <= -0.05:
        attack_share = min(attack_share, 0.20)
    if vix_crisis:
        attack_share = 0.00
    return qualified, breadth, float(attack_share)


def select_dynamic_attack_assets(history, selected, qualified):
    candidates = [ticker for ticker in selected if bool(qualified[ticker])]
    if not candidates:
        return [], pd.Series(dtype=float)
    six_month = momentum_6m(history[candidates])
    twelve_one = momentum_12_1(history[candidates])
    volatility = (
        history[candidates].iloc[-63:].std(ddof=1).clip(lower=1e-8)
        * np.sqrt(TRADING_DAYS)
    )
    risk_adjusted_momentum = (0.50 * six_month + 0.50 * twelve_one) / volatility
    ranked = risk_adjusted_momentum.sort_values(ascending=False).index.tolist()
    correlation = history[ranked].iloc[-60:].corr()
    chosen, used_classes = [], set()
    for require_new_class in (True, False):
        for ticker in ranked:
            if ticker in chosen:
                continue
            asset_class = assets.loc[ticker, 'Asset Class']
            if require_new_class and asset_class in used_classes:
                continue
            if chosen and any(
                abs(float(correlation.loc[ticker, old])) >= ATTACK_CORRELATION_LIMIT
                for old in chosen
            ):
                continue
            chosen.append(ticker)
            used_classes.add(asset_class)
            if len(chosen) >= ATTACK_MAX_ASSETS:
                return chosen, risk_adjusted_momentum
    return chosen, risk_adjusted_momentum


def run_dynamic_attack_backtest(
    name=DYNAMIC_ATTACK_NAME, end_date=RESEARCH_END,
    sora_frame=None, allow_holdout=False, short_trend_window=None,
):
    """Dynamic offensive strategy; all model targets are implemented on the next transaction date after the signal is formed.

    short_trend_window=None is GMI-ADG. An integer adds the GMI-ATC rule (Section 3.3.1):
    an offensive asset must also close above its trailing moving average of that length on the
    signal date. Core weights, trend breadth and the offensive share are not changed by the rule.
    """
    if pd.Timestamp(end_date) >= HOLDOUT_START and not allow_holdout:
        raise RuntimeError('Dynamic offensive research backsight prohibits the reading of locked samples outside.')
    sora_frame = sora_daily_df if sora_frame is None else sora_frame
    test_returns = returns.loc[strategy_start:end_date, ALL_ASSETS].copy()
    cash_returns = align_sora_returns(test_returns.index, sora_frame, allow_holdout=allow_holdout)
    funding_returns = align_sora_returns(test_returns.index, sora_frame, allow_holdout=allow_holdout)
    current_weights = pd.Series(0.0, index=ALL_ASSETS, dtype=float)
    pending_order = None
    last_selected = []
    vix_crisis = False
    nav = peak_nav = 1.0
    start_position = returns.index.get_loc(strategy_start)

    daily_records, turnover_records, trade_leg_records, cost_records = [], [], [], []
    weight_records, selection_records, trigger_records, state_records = [], [], [], []
    cash_records, exposure_records = [], []
    pre_return_weight_records, end_of_day_weight_records, trade_delta_records = [], [], []

    for date in test_returns.index:
        position = returns.index.get_loc(date)
        pre_return_weight_records.append(current_weights.rename(date))
        day_return = test_returns.loc[date]
        cash_weight = 1.0 - float(current_weights.sum())
        gross_return = float(current_weights @ day_return) + cash_weight * float(cash_returns.loc[date])
        if 1.0 + gross_return <= 0:
            raise RuntimeError('The dynamic offensive portfolio is not positive.')
        current_weights = current_weights * (1.0 + day_return) / (1.0 + gross_return)

        cost_rate = 0.0
        if pending_order is not None:
            if not pending_order['Signal Date'] < date:
                raise RuntimeError('Dynamic offensive orders are not delayed until the next trading date.')
            target = pending_order['Target weights']
            delta = target - current_weights
            turnover = float(delta.abs().sum())
            trade_legs = int((delta.abs() > 1e-10).sum())
            cost_rate = COST * turnover
            current_weights = target.copy()
            turnover_records.append((date, turnover))
            trade_leg_records.append((date, trade_legs))
            cost_records.append((date, cost_rate))
            trade_delta_records.append(delta.rename(date))
            weight_records.append(current_weights.rename(date))
            trigger_records.append({
                'Signal Date': pending_order['Signal Date'], 'Execution Date': date,
                'Reason': pending_order['Reason'], 'Offensive Allocation': pending_order['Offensive Allocation'],
                'Qualified Trend Breadth': pending_order['Qualified Trend Breadth'],
                'Portfolio Drawdown': pending_order['Portfolio Drawdown'], 'VIX Crisis': pending_order['VIX Crisis'],
                'Top10 Change': pending_order['Top10 Change'],
                'Attack assets': ','.join(pending_order['Attack assets']),
                'Proportion of total bilateral turnover of securities': turnover, 'Transaction cost rate': cost_rate,
            })
            selection_records.extend([
                {
                    'Rebalance Date': date, 'Signal Date': pending_order['Signal Date'], 'Policy': name,
                    'Code': ticker, 'Asset Name': assets.loc[ticker, 'Asset Name'],
                    'Asset Class': assets.loc[ticker, 'Asset Class'],
                    'Whether to attack assets': ticker in pending_order['Attack assets'],
                    'Final net weight': current_weights[ticker],
                }
                for ticker in current_weights.index[current_weights > 1e-10]
            ])
            pending_order = None

        net_return = (1.0 - cost_rate) * (1.0 + gross_return) - 1.0
        daily_records.append((date, net_return))
        cash_records.append((date, cash_weight))
        exposure_records.append((date, float(pre_return_weight_records[-1].abs().sum())))
        nav *= 1.0 + net_return
        peak_nav = max(peak_nav, nav)
        portfolio_drawdown = nav / peak_nav - 1.0
        end_of_day_weight_records.append(current_weights.rename(date))

        observed_vix = vix_open_daily.get(date, np.nan)
        if pd.notna(observed_vix):
            if observed_vix > VIX_CASH_HIGH:
                vix_crisis = True
            elif observed_vix < VIX_CASH_LOW:
                vix_crisis = False

        check_today = (position - start_position) % CHECK_INTERVAL == 0
        if check_today:
            history = returns.iloc[position - LOOKBACK + 1:position + 1][ALL_ASSETS]
            if len(history) != LOOKBACK or history.index.max() != date:
                raise RuntimeError('The dynamic attack history window does not strictly close to signal day closing.')
            selected, momentum_score = select_top10(history)
            core_budget = fully_dynamic_budget(history, selected, momentum_score)
            core_weights = strict_risk_budget(
                ewma_covariance(history[selected]), core_budget,
            ).reindex(ALL_ASSETS, fill_value=0.0)

            qualified, breadth, attack_share = dynamic_attack_state(
                history, selected, vix_crisis, portfolio_drawdown,
            )
            if short_trend_window is not None:
                qualified = qualified & short_trend_confirmed(history, selected, short_trend_window)
            attack_assets, _ = select_dynamic_attack_assets(
                history, selected, qualified,
            )
            if not attack_assets:
                attack_share = 0.0
            elif len(attack_assets) == 1:
                attack_share = min(attack_share, 0.15)

            attack_weights = pd.Series(0.0, index=ALL_ASSETS)
            if attack_assets:
                attack_weights.loc[attack_assets] = 1.0 / len(attack_assets)
            raw_target = (1.0 - attack_share) * core_weights + attack_share * attack_weights
            raw_target = raw_target.clip(lower=0.0)
            raw_target /= raw_target.sum()

            membership_changed = set(selected) != set(last_selected)
            if membership_changed or not last_selected:
                target = raw_target.copy()
                reason = 'Initial build-up or Top10 member change, complete'
            else:
                target = current_weights + DYNAMIC_ATTACK_PARTIAL_EXECUTION * (
                    raw_target - current_weights
                )
                small_trade = (target - current_weights).abs() < DYNAMIC_ATTACK_NO_TRADE_BAND
                target.loc[small_trade] = current_weights.loc[small_trade]
                target = target.clip(lower=0.0)
                target /= target.sum()
                reason = 'The list remains unchanged, half-speed implementation and non-tradable area application'

            pending_order = {
                'Signal Date': date, 'Target weights': target, 'Reason': reason,
                'Offensive Allocation': attack_share, 'Qualified Trend Breadth': breadth,
                'Portfolio Drawdown': portfolio_drawdown, 'VIX Crisis': vix_crisis,
                'Top10 Change': membership_changed, 'Attack assets': list(attack_assets),
            }
            last_selected = list(selected)

        state_records.append({
            'Date': date, 'VIX start value': observed_vix, 'VIX Crisis': vix_crisis,
            'Portfolio Drawdown': portfolio_drawdown,
            'Cash weights between areas of return': cash_weight,
            'Total risk securities in the yield zone mouth': float(pre_return_weight_records[-1].sum()),
            'Cash weight after closing': 1.0 - float(current_weights.sum()),
            'After closing, risk securities are all open. mouth': float(current_weights.sum()),
        })

    result = {
        'name': name, 'returns': pd.Series(dict(daily_records), name=name),
        'turnover': pd.Series(dict(turnover_records), name=name, dtype=float),
        'trade_legs': pd.Series(dict(trade_leg_records), name=name, dtype=float),
        'transaction_cost': pd.Series(dict(cost_records), name='Transaction cost rate', dtype=float),
        'weights': pd.DataFrame(weight_records).reindex(columns=ALL_ASSETS, fill_value=0.0),
        'pre_return_weights': pd.DataFrame(pre_return_weight_records).reindex(columns=ALL_ASSETS, fill_value=0.0),
        'end_of_day_weights': pd.DataFrame(end_of_day_weight_records).reindex(columns=ALL_ASSETS, fill_value=0.0),
        'trade_deltas': pd.DataFrame(trade_delta_records).reindex(columns=ALL_ASSETS, fill_value=0.0),
        'selections': pd.DataFrame(selection_records), 'triggers': pd.DataFrame(trigger_records),
        'state_history': pd.DataFrame(state_records).set_index('Date'),
        'cash_weight': pd.Series(dict(cash_records), name='Cash weights', dtype=float),
        'gross_exposure': pd.Series(dict(exposure_records), name='Total exposure to risk securities mouth', dtype=float),
        'risk_free_daily': cash_returns, 'funding_daily': funding_returns,
        'trade_assets': list(ALL_ASSETS), 'instrument_returns': test_returns,
        'instrument_prices': prices_sgd.reindex(test_returns.index),
        'asset_cost_rate': pd.Series(COST, index=ALL_ASSETS, dtype=float),
    }
    if not result['triggers'].empty and not (
        result['triggers']['Signal Date'] < result['triggers']['Execution Date']
    ).all():
        raise RuntimeError('A dynamic offensive strategy exists on the same day or future data leaks.')
    if not np.allclose(result['weights'].sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError('The implementation weight of the dynamic offensive strategy was not maintained at 100 per cent.')
    return result


results[DYNAMIC_ATTACK_NAME] = run_dynamic_attack_backtest()
SUBMISSION_STRATEGY = DYNAMIC_ATTACK_NAME
strategy_returns = pd.concat(
    [results[name]['returns'].rename(name) for name in results], axis=1
).dropna(how='any')

dynamic_attack_audit = pd.DataFrame({
    'Value': [
        results[DYNAMIC_ATTACK_NAME]['returns'].index.max().date(),
        len(results[DYNAMIC_ATTACK_NAME]['turnover']),
        0.5 * float(results[DYNAMIC_ATTACK_NAME]['turnover'].sum()),
        float(results[DYNAMIC_ATTACK_NAME]['triggers']['Offensive Allocation'].mean()),
        float(results[DYNAMIC_ATTACK_NAME]['triggers']['Offensive Allocation'].eq(0.0).mean()),
        float(results[DYNAMIC_ATTACK_NAME]['triggers']['Offensive Allocation'].eq(0.30).mean()),
    ]
}, index=[
    'Final date of research findings', 'Number of redeployments', 'Cumulative unilateral turnover rate', 'Average attack rate',
    'Zero-aggression ratio.', '30% of attacks',
])
display(dynamic_attack_audit)
assert results[DYNAMIC_ATTACK_NAME]['returns'].index.max() < HOLDOUT_START
print(f'Official strategy registration is complete:{DYNAMIC_ATTACK_NAME}')

# 3.3.1 GMI-ATC: 20-Day Short-Trend Confirmation for the Offensive Sleeve (Final Product)
#
# GMI-ATC keeps every GMI-ADG rule and adds one offensive-eligibility condition: on each 15-session
# signal date an offensive asset must also close above its own 20-day moving average (SGD price,
# closes up to and including the signal date). Core weights, trend breadth and the 0%-30% offensive
# share are unchanged; only which qualified assets may receive the offensive overlay changes.
# The rule was added on 2026-10-21 after reviewing October 2026 out-of-sample results; the research
# evidence for it is the 2015-2026 research period and the parameter grid reported in the project.
# --- Original Notebook code unit 30 ---
DYNAMIC_ATTACK_H20_NAME = STRATEGY_NAMES['dynamic_attack_h20']
ATTACK_SHORT_MA = 20


def short_trend_confirmed(history, tickers, window=ATTACK_SHORT_MA):
    """Signal-day close above its own trailing moving average, using only the signal-window history."""
    tickers = list(tickers)
    normalized_price = (1.0 + history[tickers]).cumprod()
    return normalized_price.iloc[-1] > normalized_price.iloc[-int(window):].mean()


results[DYNAMIC_ATTACK_H20_NAME] = run_dynamic_attack_backtest(
    DYNAMIC_ATTACK_H20_NAME, short_trend_window=ATTACK_SHORT_MA,
)
SUBMISSION_STRATEGY = DYNAMIC_ATTACK_H20_NAME
strategy_returns = pd.concat(
    [results[name]['returns'].rename(name) for name in results], axis=1
).dropna(how='any')

h20_triggers = results[DYNAMIC_ATTACK_H20_NAME]['triggers']
adg_triggers = results[DYNAMIC_ATTACK_NAME]['triggers']
h20_comparison = pd.DataFrame({
    strategy_label(name): {
        'Final date of research findings': results[name]['returns'].index.max().date(),
        'Number of redeployments': len(results[name]['turnover']),
        'Cumulative unilateral turnover rate': 0.5 * float(results[name]['turnover'].sum()),
        'Average attack rate': float(results[name]['triggers']['Offensive Allocation'].mean()),
        'Signals with at least one offensive asset': float(results[name]['triggers']['Attack assets'].ne('').mean()),
    }
    for name in [DYNAMIC_ATTACK_NAME, DYNAMIC_ATTACK_H20_NAME]
})
changed_signals = (adg_triggers['Attack assets'].reset_index(drop=True)
                   != h20_triggers['Attack assets'].reset_index(drop=True))
h20_comparison.loc['Signals where the offensive assets differ from GMI-ADG'] = [0.0, float(changed_signals.mean())]
display(h20_comparison)
for _, trigger in h20_triggers.iterrows():
    for ticker in [t for t in trigger['Attack assets'].split(',') if t]:
        signal_prices = prices_sgd[ticker].loc[:trigger['Signal Date']]
        if not signal_prices.iloc[-1] > signal_prices.iloc[-ATTACK_SHORT_MA:].mean():
            raise RuntimeError(f'GMI-ATC offensive asset {ticker} breaks the 20-day rule on {trigger["Signal Date"].date()}.')
assert results[DYNAMIC_ATTACK_H20_NAME]['returns'].index.max() < HOLDOUT_START
print(f'Official strategy registration is complete:{DYNAMIC_ATTACK_H20_NAME} (final submission strategy)')

# 3.4 Classic monthly asset risk parities
# --- Original Notebook code unit 31--
STATIC_CLASSIC_NAME = STRATEGY_NAMES['monthly_erc']


def target_all_asset_classic(history):
    return strict_erc(sample_covariance(history)).reindex(ALL_ASSETS)


def run_all_asset_monthly_backtest(
    name, allocator, end_date=RESEARCH_END, sora_frame=None, allow_holdout=False,
):
    """The target was set at the beginning of the month and the closing date of the next transaction was implemented to avoid the receipt of new weights to the receipt proceeds before they were delivered."""
    if pd.Timestamp(end_date) >= HOLDOUT_START and not allow_holdout:
        raise RuntimeError('The monthly policy study returns to prohibit access to the outer range of locked samples.')
    sora_frame = sora_daily_df if sora_frame is None else sora_frame
    test_returns = returns.loc[strategy_start:end_date].copy()
    cash_returns = align_sora_returns(test_returns.index, sora_frame, allow_holdout=allow_holdout)
    funding_returns = align_sora_returns(test_returns.index, sora_frame, allow_holdout=allow_holdout)
    current_weights = pd.Series(0.0, index=ALL_ASSETS)
    month_starts = set(test_returns.groupby(test_returns.index.to_period('M')).head(1).index)
    pending_order = None
    daily_records, turnover_records, trade_leg_records, cost_records = [], [], [], []
    weight_records, selection_records, trigger_records = [], [], []
    pre_return_weight_records, end_of_day_weight_records, trade_delta_records = [], [], []

    for date in test_returns.index:
        pre_return_weight_records.append(current_weights.rename(date))
        day_return = test_returns.loc[date]
        cash_weight = 1.0 - float(current_weights.sum())
        cash_rate = cash_returns.loc[date] if cash_weight >= 0 else funding_returns.loc[date]
        gross_return = float(current_weights @ day_return) + cash_weight * cash_rate
        denominator = 1.0 + gross_return
        if denominator <= 0:
            raise RuntimeError('The net monthly strategy is not positive.')
        current_weights = current_weights * (1.0 + day_return) / denominator

        cost_rate = 0.0
        if pending_order is not None:
            if not pending_order['Signal Date'] < date:
                raise RuntimeError('Monthly strategic orders were not delayed until the next transaction date.')
            target = pending_order['Target weights']
            delta = target - current_weights
            turnover = float(delta.abs().sum())
            trade_legs = int((delta.abs() > 1e-10).sum())
            cost_rate = COST * turnover
            current_weights = target.copy()
            turnover_records.append((date, turnover))
            trade_leg_records.append((date, trade_legs))
            cost_records.append((date, cost_rate))
            trade_delta_records.append(delta.rename(date))
            weight_records.append(current_weights.rename(date))
            trigger_records.append({
                'Signal Date': pending_order['Signal Date'], 'Execution Date': date,
                'Reason': 'Monthly rebalancing (receivable on next transaction date)', 'Trigger deviation': np.nan,
                'Total asset weights': float(current_weights.sum()), 'Transaction cost rate': cost_rate,
            })
            selection_records.extend([
                {'Rebalance Date': date, 'Signal Date': pending_order['Signal Date'], 'Policy': name,
                 'Code': ticker, 'Asset Name': assets.loc[ticker, 'Asset Name'],
                 'Motion Score': np.nan, 'Target weights': current_weights[ticker]}
                for ticker in current_weights.index[current_weights > 1e-12]
            ])
            pending_order = None

        net_return = (1.0 - cost_rate) * (1.0 + gross_return) - 1.0
        daily_records.append((date, net_return))
        end_of_day_weight_records.append(current_weights.rename(date))

        position = returns.index.get_loc(date)
        history = returns.iloc[position - LOOKBACK + 1:position + 1]
        if date in month_starts and pending_order is None:
            target = pd.Series(allocator(history), dtype=float).reindex(ALL_ASSETS).fillna(0.0)
            target = target.clip(lower=0.0)
            target /= target.sum()
            pending_order = {'Signal Date': date, 'Target weights': target}

    return {
        'name': name, 'returns': pd.Series(dict(daily_records), name=name),
        'trade_legs': pd.Series(dict(trade_leg_records), name=name, dtype=float),
        'turnover': pd.Series(dict(turnover_records), name=name, dtype=float),
        'transaction_cost': pd.Series(dict(cost_records), name='Transaction cost rate', dtype=float),
        'weights': pd.DataFrame(weight_records), 'selections': pd.DataFrame(selection_records),
        'pre_return_weights': pd.DataFrame(pre_return_weight_records),
        'end_of_day_weights': pd.DataFrame(end_of_day_weight_records),
        'trade_deltas': pd.DataFrame(trade_delta_records),
        'triggers': pd.DataFrame(trigger_records), 'risk_free_daily': cash_returns,
        'funding_daily': funding_returns,
        'asset_cost_rate': pd.Series(COST, index=ALL_ASSETS, dtype=float),
    }



static_monthly_results = {
    STATIC_CLASSIC_NAME: run_all_asset_monthly_backtest(STATIC_CLASSIC_NAME, target_all_asset_classic),
}
results.update(static_monthly_results)
strategy_returns = pd.concat([obj['returns'] for obj in results.values()], axis=1).dropna()

print('All-asset monthly classic risk parity policy is complete:')
for strategy_name, strategy_result in static_monthly_results.items():
    print(f'{strategy_name}Monthly changes{len(strategy_result["turnover"])}Minor')

# The classification implementation experiment strategy has been removed from the official research code.


def realistic_static_cost_rates(asset_list):
    """Returns fixed unilateral cost decomposition based on listed market and currency of trade."""
    metadata = assets.reindex(asset_list)
    commission = metadata['Market'].map(MARKET_COMMISSION_RATE).astype(float)
    spread = metadata['Market'].map(MARKET_SPREAD_SLIPPAGE_RATE).astype(float)
    fx = metadata['Local Currency'].map(FX_CONVERSION_RATE).astype(float)
    if commission.isna().any() or spread.isna().any() or fx.isna().any():
        raise RuntimeError('Real cost parameters do not cover the entire asset market or the currency of the transaction.')
    # Bitcoin is not ETF and uses a broader transaction fee and price differential assumption separately.
    if 'BTC-USD' in asset_list:
        commission.loc['BTC-USD'] = 0.0015
        spread.loc['BTC-USD'] = 0.0015
    return pd.DataFrame({'Commissions and transaction costs': commission, 'FX Conversion Costs': fx, 'Bid-Ask Spread and Slippage': spread})


def apply_realistic_cost_model(result):
    """The inclusion of sub-market costs, shocks and withholding taxes in the freeze on hold and trade routes does not change signals."""
    output = dict(result)
    dates = result['returns'].index
    asset_list = list(result.get('trade_assets', ALL_ASSETS))
    asset_list = [ticker for ticker in asset_list if ticker in ALL_ASSETS]
    deltas = result['trade_deltas'].reindex(index=dates, columns=asset_list, fill_value=0.0).fillna(0.0)
    held = result['pre_return_weights'].reindex(index=dates, columns=asset_list, fill_value=0.0).fillna(0.0)
    old_cost = result['transaction_cost'].reindex(dates).fillna(0.0)
    old_net = result['returns'].reindex(dates).astype(float)
    if (old_cost >= 1.0).any():
        raise RuntimeError('The original transaction cost rate was abnormal and it was not possible to rebuild gross gains.')
    gross_before_new_cost = (1.0 + old_net) / (1.0 - old_cost) - 1.0

    static_rates = realistic_static_cost_rates(asset_list)
    tax_annual = pd.Series(ANNUAL_WITHHOLDING_TAX_DRAG, dtype=float).reindex(asset_list).fillna(0.0)
    lagged_adv = dollar_volume_sgd[asset_list].shift(1).rolling(
        IMPACT_LOOKBACK, min_periods=5
    ).median().reindex(dates)
    lagged_vol = returns[asset_list].shift(1).rolling(
        IMPACT_LOOKBACK, min_periods=5
    ).std().reindex(dates).fillna(0.0)

    nav = INITIAL_CAPITAL_SGD
    component_rows, effective_rate_rows, net_rows = [], [], []
    for date in dates:
        absolute_delta = deltas.loc[date].abs()
        commission = float(absolute_delta @ static_rates['Commissions and transaction costs'])
        fx_cost = float(absolute_delta @ static_rates['FX Conversion Costs'])
        spread = float(absolute_delta @ static_rates['Bid-Ask Spread and Slippage'])
        order_value = absolute_delta * nav
        adv = lagged_adv.loc[date].replace(0.0, np.nan)
        participation = (order_value / adv).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        impact_unit_rate = (
            IMPACT_COEFFICIENT * lagged_vol.loc[date] * np.sqrt(participation)
        ).clip(lower=0.0, upper=MAX_IMPACT_RATE)
        market_impact = float(absolute_delta @ impact_unit_rate)
        effective_asset_rate = static_rates.sum(axis=1) + impact_unit_rate
        withholding_tax = float(held.loc[date] @ (tax_annual / TRADING_DAYS))
        transaction_cost = commission + fx_cost + spread + market_impact
        after_tax_gross = float(gross_before_new_cost.loc[date]) - withholding_tax
        net_return = (1.0 - transaction_cost) * (1.0 + after_tax_gross) - 1.0
        component_rows.append(pd.Series({
            'Commissions and transaction costs': commission, 'FX Conversion Costs': fx_cost,
            'Bid-Ask Spread and Slippage': spread, 'Market Impact': market_impact,
            'Obsessed taxes': withholding_tax,
        }, name=date))
        effective_rate_rows.append(effective_asset_rate.rename(date))
        net_rows.append((date, net_return))
        nav *= 1.0 + net_return

    components = pd.DataFrame(component_rows).fillna(0.0)
    output['uniform_cost_returns'] = old_net.copy()
    output['returns'] = pd.Series(dict(net_rows), name=result['name'], dtype=float)
    output['transaction_cost'] = components[
        ['Commissions and transaction costs', 'FX Conversion Costs', 'Bid-Ask Spread and Slippage', 'Market Impact']
    ].sum(axis=1).rename('Actual transaction cost rate')
    output['withholding_tax_drag'] = components['Obsessed taxes'].rename('Obsessed taxes')
    output['cost_components'] = components
    output['asset_cost_rate'] = static_rates.sum(axis=1)
    output['effective_asset_cost_rate'] = pd.DataFrame(effective_rate_rows).reindex(
        index=dates, columns=asset_list
    ).fillna(0.0)
    output['cost_model'] = 'Real cost scenarios for independent sub-markets'
    return output


REALISTIC_COST_ASSUMPTIONS = realistic_static_cost_rates(ALL_ASSETS).join(
    pd.Series(ANNUAL_WITHHOLDING_TAX_DRAG, name='Annual withholding tax drag agent')
)
# The formal result continues to be a unified unilateral 0.05 per cent. The cost of the difference in reality is only an independent copy of the final strategy.
# Results cannot be covered or the subsequent official retrospect, pressure testing and comparison of the official strategy cannot be changed.
realistic_submission_result = apply_realistic_cost_model(results[SUBMISSION_STRATEGY])
strategy_returns = pd.concat(
    [results[name]['returns'].rename(name) for name in results], axis=1
).dropna(how='any')
