"""Motion signals, coupling models, ERC and dynamic risk budgets.

This document is broken down by function by the original Notebook. Please run in the given order through Main.py and do not execute independently of the document that relies on a prior study.
"""

# 2.1 Dynamic and trend variables
#
# Define a six-month drive, 12-1 momentum and MA200 trend state for asset screening and dynamic risk budgeting.
# --- Original Notebook code module 11--
def momentum_6m(history):
    return (1.0 + history.iloc[-MOMENTUM_6M:]).prod() - 1.0


def momentum_12_1(history):
    return (1.0 + history.iloc[-LOOKBACK:-MOMENTUM_SKIP]).prod() - 1.0


def above_ma200(history):
    normalized_price = (1.0 + history).cumprod()
    return normalized_price.iloc[-1] > normalized_price.iloc[-MA_WINDOW:].mean()

# 2.2 EWMA and Downward Risk Model
#
# The EWMA Annual Difference Matrix and the following line zero yield are estimated to be the threshold of the annualized bottom-up matrix.
# --- Original Notebook code module 13--
def ewma_covariance(history, decay=EWMA_LAMBDA):
    x = history.astype(float)
    raw_weights = decay ** np.arange(len(x) - 1, -1, -1)
    weights = raw_weights / raw_weights.sum()
    mean = np.average(x.values, axis=0, weights=weights)
    centered = x.values - mean
    denominator = max(1.0 - np.sum(weights ** 2), 1e-12)
    covariance = (centered * weights[:, None]).T @ centered / denominator
    return pd.DataFrame(covariance * TRADING_DAYS, index=x.columns, columns=x.columns)


def downside_second_moment(history):
    downside = np.minimum(history.astype(float).values, 0.0)
    matrix = downside.T @ downside / len(downside)
    return pd.DataFrame(matrix * TRADING_DAYS, index=history.columns, columns=history.columns)

# 2.3 Strict ERC solver
#
# Calculated the share of asset risk contribution and verified errors in target contribution through a rigorous risk budget.
# --- Original Notebook code unit 15--
def risk_contribution_share(weights, covariance):
    w = pd.Series(weights, dtype=float).reindex(covariance.index).fillna(0.0)
    variance = float(w.values @ covariance.values @ w.values)
    if variance <= 0 or not np.isfinite(variance):
        return pd.Series(np.nan, index=covariance.index)
    return w * (covariance @ w) / variance


def strict_risk_budget(covariance, budget=None, tol=1e-8, max_iter=5000):
    cov = pd.DataFrame(covariance, dtype=float).copy()
    cov = (cov + cov.T) / 2
    n = len(cov)
    if budget is None:
        budget = pd.Series(1.0 / n, index=cov.index)
    budget = pd.Series(budget, dtype=float).reindex(cov.index)
    budget = budget / budget.sum()
    if n == 1:
        return pd.Series(1.0, index=cov.index)

    eig_min = np.linalg.eigvalsh(cov.values).min()
    if eig_min <= 1e-10:
        cov.values[np.diag_indices_from(cov.values)] += 1e-10 - eig_min

    sigma = cov.values
    b = budget.values
    x = np.sqrt(b) / np.sqrt(np.maximum(np.diag(sigma), 1e-12))
    x = np.maximum(x, 1e-12)

    for _ in range(max_iter):
        old = x.copy()
        sx = sigma @ x
        for i in range(n):
            cross = sx[i] - sigma[i, i] * x[i]
            x_new = (-cross + np.sqrt(cross * cross + 4 * sigma[i, i] * b[i])) / (2 * sigma[i, i])
            delta = x_new - x[i]
            x[i] = max(x_new, 1e-12)
            sx += sigma[:, i] * delta
        if np.max(np.abs(x - old)) < tol:
            break

    weights = pd.Series(x / x.sum(), index=cov.index)
    error = float((risk_contribution_share(weights, cov) - budget).abs().max())
    if not np.isfinite(error) or error > 1e-4:
        raise RuntimeError(f'Strict risk budget verification failed, maximum error ={error:.4%}')
    return weights


def strict_erc(covariance):
    return strict_risk_budget(covariance)

# 2.4 Top10 Screening and Basic Target Rights Heavy
#
# Top10 is selected at 12-1 momentum and gives rise to a negative impact of volatility, classic ERC, ZXQ0QZ ZXQ1QZ, downline risk ERC and full assets.
# --- Original Notebook code module 17--
def select_top10(history):
    score = momentum_12_1(history)
    return score.nlargest(TOP_N).index.tolist(), score


def target_momentum_inverse_vol(history):
    selected, score = select_top10(history)
    vol = history[selected].iloc[-VOL_LOOKBACK:].std(ddof=1) * np.sqrt(TRADING_DAYS)
    weights = (1.0 / vol.clip(lower=1e-8))
    weights /= weights.sum()
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score


def target_momentum_erc(history):
    selected, score = select_top10(history)
    weights = strict_erc(history[selected].cov() * TRADING_DAYS)
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score


def target_momentum_ewma_erc(history):
    selected, score = select_top10(history)
    weights = strict_erc(ewma_covariance(history[selected]))
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score


def target_momentum_downside_erc(history):
    selected, score = select_top10(history)
    weights = strict_erc(downside_second_moment(history[selected]))
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score


def target_equal_weight(history):
    return pd.Series(1.0 / len(ALL_ASSETS), index=ALL_ASSETS), ALL_ASSETS, pd.Series(np.nan, index=ALL_ASSETS)

# 2.5 Bilayered dynamic risk budget for single and static categories
#
# An asset-level dynamic risk budget is first defined, followed by a two-tiered risk budget with a fixed category budget and a dynamic distribution within the class.
# --- Original Notebook code module 19--
def dynamic_risk_budget(history, selected, score):
    six_month = momentum_6m(history[selected])
    ma_signal = above_ma200(history[selected])
    rank_multiplier = 0.75 + 0.50 * score[selected].rank(pct=True)
    trend_multiplier = pd.Series(
        np.select([((six_month > 0) & ma_signal), ((six_month > 0) | ma_signal)],
                  [1.50, 1.00], default=0.50),
        index=selected,
    )
    raw_budget = rank_multiplier * trend_multiplier
    return raw_budget / raw_budget.sum()


def target_momentum_ewma_dynamic_budget(history):
    selected, score = select_top10(history)
    budget = dynamic_risk_budget(history, selected, score)
    weights = strict_risk_budget(ewma_covariance(history[selected]), budget)
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score

CLASS_RISK_BUDGET = pd.Series({
    'Equities': 0.40,
    'Bonds': 0.35,
    'Commodities': 0.15,
    'FX': 0.05,
    'Crypto': 0.05,
})


def hierarchical_dynamic_risk_budget(history, selected, score):
    """The base budget is allocated between categories and is distributed internally according to dynamics and trends."""
    selected = list(selected)
    final_budget = pd.Series(0.0, index=selected, dtype=float)
    selected_classes = assets.loc[selected, 'Asset Class']
    active_classes = [asset_class for asset_class in CLASS_RISK_BUDGET.index
                      if (selected_classes == asset_class).any()]
    active_class_budget = CLASS_RISK_BUDGET.loc[active_classes]
    active_class_budget = active_class_budget / active_class_budget.sum()

    for asset_class, class_budget in active_class_budget.items():
        class_assets = selected_classes.index[selected_classes == asset_class].tolist()
        six_month = momentum_6m(history[class_assets])
        ma_signal = above_ma200(history[class_assets])
        rank_multiplier = 0.75 + 0.50 * score[class_assets].rank(pct=True)
        trend_multiplier = pd.Series(
            np.select(
                [((six_month > 0) & ma_signal), ((six_month > 0) | ma_signal)],
                [1.50, 1.00],
                default=0.50,
            ),
            index=class_assets,
            dtype=float,
        )
        internal_budget = rank_multiplier * trend_multiplier
        internal_budget = internal_budget / internal_budget.sum()
        final_budget.loc[class_assets] = float(class_budget) * internal_budget

    final_budget = final_budget / final_budget.sum()
    return final_budget


def target_momentum_ewma_hierarchical_budget(history):
    selected, score = select_top10(history)
    budget = hierarchical_dynamic_risk_budget(history, selected, score)
    covariance = ewma_covariance(history[selected])
    weights = strict_risk_budget(covariance, budget)
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score

# 2.6 GARCH (1,1) Risk model
#
# A single variable, GARCH (1,1) is used to forecast the rate of fluctuations for the next period for each asset and is combined with a historically relevant matrix into an annual reconciliation matrix.
# --- Original Notebook code unit 21--
GARCH_VOL_CACHE = {}
GARCH_FIT_DIAGNOSTICS = []


def garch11_forecast_volatility(history):
    """The proceeds from the collection up to the signal date are to be matched to GARCH (1,1) and return to the rate of fluctuations in the conditions of the following day."""
    key = (pd.Timestamp(history.index[-1]), tuple(history.columns), len(history))
    if key in GARCH_VOL_CACHE:
        return GARCH_VOL_CACHE[key].copy()

    forecasts = {}
    for ticker in history.columns:
        series = history[ticker].dropna().astype(float) * 100.0
        model = arch_model(
            series, mean='Constant', vol='GARCH', p=1, o=0, q=1,
            dist='normal', rescale=False,
        )
        fitted = model.fit(disp='off', show_warning=False, options={'maxiter': 500})
        convergence_flag = int(fitted.convergence_flag)
        forecast_variance = np.nan
        if convergence_flag == 0:
            forecast_variance = float(
                fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
            ) / 10000.0
        used_fallback = convergence_flag != 0 or not np.isfinite(forecast_variance) or forecast_variance <= 0
        if used_fallback:
            # The numerical optimization of individual windows may fail. The comparison strategy should not interrupt the entire study.
            # Therefore, the EWMA-day margin reserve for the same signal day is used only for failed assets.
            forecast_variance = float(
                ewma_covariance(history[[ticker]]).iloc[0, 0] / TRADING_DAYS
            )
        if not np.isfinite(forecast_variance) or forecast_variance <= 0:
            raise RuntimeError(f'{ticker}The difference between the GARCH and the EWMA reserve projection is not valid.')
        forecasts[ticker] = np.sqrt(forecast_variance)
        GARCH_FIT_DIAGNOSTICS.append({
            'Estimated date': history.index[-1], 'Assets': ticker,
            'Depress marks': convergence_flag, 'Use EWMA backup': used_fallback,
        })

    result = pd.Series(forecasts, index=history.columns, dtype=float)
    GARCH_VOL_CACHE[key] = result.copy()
    return result


def garch11_covariance(history):
    daily_volatility = garch11_forecast_volatility(history)
    correlation = history.corr().reindex(index=history.columns, columns=history.columns)
    correlation = correlation.fillna(0.0)
    correlation = ((correlation + correlation.T) / 2.0).copy()
    correlation_array = correlation.to_numpy(copy=True)
    np.fill_diagonal(correlation_array, 1.0)
    correlation = pd.DataFrame(correlation_array, index=history.columns, columns=history.columns)
    daily_covariance = correlation.mul(daily_volatility, axis=0).mul(daily_volatility, axis=1)
    return daily_covariance * TRADING_DAYS


def target_momentum_garch_erc(history):
    selected, score = select_top10(history)
    covariance = garch11_covariance(history[selected])
    weights = strict_erc(covariance)
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score

# 2.7 A fully dynamic two-tier risk budget
#
# The allocation of risk budgets between and within categories is based on the dynamics of momentum, trends and rankings, followed by the use of the EWMA agreement to resolve strict risk budget weights.
# --- Original Notebook code module 23--
FULLY_DYNAMIC_BUDGET_NAME = STRATEGY_NAMES['fully_dynamic']


def fully_dynamic_class_budget(history, selected, score):
    """Outer layer: Class risk budgets vary according to category dynamic + category trends (as distinct from static CLASS_RISK_BUDGET)."""
    selected = list(selected)
    selected_classes = assets.loc[selected, 'Asset Class']
    active_classes = list(dict.fromkeys(selected_classes.tolist()))
    class_momentum = {}
    class_trend = {}
    for asset_class in active_classes:
        class_assets = selected_classes.index[selected_classes == asset_class].tolist()
        class_momentum[asset_class] = momentum_12_1(history[class_assets]).mean()
        six_month = momentum_6m(history[class_assets])
        ma_signal = above_ma200(history[class_assets])
        class_trend[asset_class] = float(((six_month > 0) & ma_signal).mean())
    momentum_series = pd.Series(class_momentum)
    rank_multiplier = 0.75 + 0.50 * momentum_series.rank(pct=True)
    trend_series = pd.Series(class_trend)
    trend_multiplier = pd.Series(
        np.select([trend_series >= 0.5, trend_series >= 0.25], [1.50, 1.00], default=0.50),
        index=trend_series.index,
    )
    raw = rank_multiplier * trend_multiplier
    return raw / raw.sum()


def fully_dynamic_budget(history, selected, score):
    """Full dynamic double layer: budgets within dynamic categories x budgets within dynamic categories."""
    selected = list(selected)
    class_budget = fully_dynamic_class_budget(history, selected, score)
    selected_classes = assets.loc[selected, 'Asset Class']
    final_budget = pd.Series(0.0, index=selected, dtype=float)
    for asset_class, class_weight in class_budget.items():
        class_assets = selected_classes.index[selected_classes == asset_class].tolist()
        six_month = momentum_6m(history[class_assets])
        ma_signal = above_ma200(history[class_assets])
        rank_multiplier = 0.75 + 0.50 * score[class_assets].rank(pct=True)
        trend_multiplier = pd.Series(
            np.select([((six_month > 0) & ma_signal), ((six_month > 0) | ma_signal)],
                      [1.50, 1.00], default=0.50),
            index=class_assets,
        )
        internal = rank_multiplier * trend_multiplier
        internal = internal / internal.sum()
        final_budget.loc[class_assets] = float(class_weight) * internal
    return final_budget / final_budget.sum()


def target_momentum_ewma_fulldynamic_budget(history):
    selected, score = select_top10(history)
    budget = fully_dynamic_budget(history, selected, score)
    weights = strict_risk_budget(ewma_covariance(history[selected]), budget)
    return weights.reindex(ALL_ASSETS, fill_value=0.0), selected, score

