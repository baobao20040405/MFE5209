"""Final-product functions extracted from the completed research.

Loaded by the product runner into an explicit shared execution context.
"""

def momentum_6m(history):
    return (1.0 + history.iloc[-MOMENTUM_6M:]).prod() - 1.0


def momentum_12_1(history):
    return (1.0 + history.iloc[-LOOKBACK:-MOMENTUM_SKIP]).prod() - 1.0


def above_ma200(history):
    normalized_price = (1.0 + history).cumprod()
    return normalized_price.iloc[-1] > normalized_price.iloc[-MA_WINDOW:].mean()


def ewma_covariance(history, decay=None):
    decay = EWMA_LAMBDA if decay is None else float(decay)
    x = history.astype(float)
    raw_weights = decay ** np.arange(len(x) - 1, -1, -1)
    weights = raw_weights / raw_weights.sum()
    mean = np.average(x.values, axis=0, weights=weights)
    centered = x.values - mean
    denominator = max(1.0 - np.sum(weights ** 2), 1e-12)
    covariance = (centered * weights[:, None]).T @ centered / denominator
    return pd.DataFrame(covariance * TRADING_DAYS, index=x.columns, columns=x.columns)


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


def select_top10(history):
    score = momentum_12_1(history)
    return score.nlargest(TOP_N).index.tolist(), score


def target_equal_weight(history):
    return pd.Series(1.0 / len(ALL_ASSETS), index=ALL_ASSETS), ALL_ASSETS, pd.Series(np.nan, index=ALL_ASSETS)


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


def short_trend_confirmed(history, tickers, window=None):
    """GMI-ATC eligibility: signal-day close above its own trailing moving average.

    Uses only the signal-window history (closes up to and including the signal date).
    """
    window = ATTACK_SHORT_MA if window is None else int(window)
    tickers = list(tickers)
    normalized_price = (1.0 + history[tickers]).cumprod()
    return normalized_price.iloc[-1] > normalized_price.iloc[-window:].mean()


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
