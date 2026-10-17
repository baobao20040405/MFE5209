"""Final-product functions extracted from the completed research.

Loaded by the product runner into an explicit shared execution context.
"""

def prepare_target(name, target):
    target = pd.Series(target, dtype=float).reindex(PORTFOLIO_ASSETS).fillna(0.0).clip(lower=0.0)
    if target.sum() <= 0:
        raise RuntimeError(f'{name}Target weights are empty.')
    target /= target.sum()
    return target


def run_dynamic_backtest(
    name, allocator, covariance_function=None, end_date=None,
    sora_frame=None, allow_holdout=False,
):
    """Scroll back: a signal is formed after the t-day closing, the t-+1-day closing is concluded and the new weight begins to yield benefits between the subsequent closing and the closing area.

(a) The transactional cost of the closing date is only deducted from the transaction cost of the closing date and does not allow new weights to yield the benefit before the transaction is made;
The calibre does not give new weights the benefit of signal dayst, but it is not equivalent to using real t+1 openings for sale.
"""
    end_date = RESEARCH_END if end_date is None else pd.Timestamp(end_date)
    if end_date >= HOLDOUT_START and not allow_holdout:
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


def run_dynamic_attack_backtest(
    name=None, end_date=None,
    sora_frame=None, allow_holdout=False,
):
    """Dynamic offensive strategy; all model targets are implemented on the next transaction date after the signal is formed."""
    name = DYNAMIC_ATTACK_NAME if name is None else name
    end_date = RESEARCH_END if end_date is None else pd.Timestamp(end_date)
    if end_date >= HOLDOUT_START and not allow_holdout:
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


if __name__ == '__main__':
    from product_evaluation import evaluate_historical_backtest
    report = evaluate_historical_backtest()
    print(report['metrics'].to_string())
    print('Historical period:', report['status']['actual_start'], 'to', report['status']['actual_end'])
    print('Initial capital: SGD 1,000,000.00')
    print('Output:', report['status']['output_dir'])
