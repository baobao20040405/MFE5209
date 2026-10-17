"""Validate observed quotes separately from forward-filled valuation prices."""
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd


class QuoteCoverageError(ValueError):
    def __init__(self, message, *, safe_end=None):
        super().__init__(message)
        self.safe_end = safe_end


@lru_cache(maxsize=64)
def _calendar(name, year):
    return xcals.get_calendar(name, start=f'{year}-01-01', end=f'{year}-12-31')


def _exchange(ticker):
    if ticker.endswith('=X'):
        # OTC FX has no single exchange calendar. London sessions provide a
        # conservative minimum-coverage check, not a claim of all FX sessions.
        return 'XLON'
    if ticker == 'BTC-USD':
        return '24/7'
    for suffix, exchange in [('.DE', 'XETR'), ('.L', 'XLON'), ('.T', 'XTKS'),
                             ('.SS', 'XSHG'), ('.HK', 'XHKG'), ('.SI', 'XSES')]:
        if ticker.endswith(suffix):
            return exchange
    return 'XNYS'


def validate_quote_coverage(context, start, end):
    """Reject missing exchange-session quotes in the evaluation interval.

    Holidays are allowed. Filled valuation rows cannot prove observations exist.
    Checks all 18 instruments, VT and the VIX inputs, even for unheld assets.
    This checks calendar dates, not intraday execution/liquidity or warm-up data.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError('No completed observations in the requested interval.')
    raw = context['raw']
    if not raw.index.is_monotonic_increasing or not raw.index.is_unique:
        raise ValueError('Observed market dates must be unique and increasing.')
    fields = [('Close', ticker) for ticker in [*context['ALL_ASSETS'], 'VT', '^VIX', '^VIX3M']]
    fields.append(('Open', '^VIX'))
    fields.extend(('Close', ticker) for ticker in context.get('FX_DOWNLOAD', []))
    rows, errors, cutoffs = [], [], []
    interior_gap = False
    for field, ticker in fields:
        name = _exchange(ticker)
        # Bound each calendar by year; do not rely on its rolling default range.
        expected = pd.DatetimeIndex([])
        for year in range(start.year, end.year + 1):
            first, last = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
            calendar = _calendar(name, year)
            first = max(first, calendar.first_session.tz_localize(None))
            last = min(last, calendar.last_session.tz_localize(None))
            if first <= last:
                expected = expected.union(calendar.sessions_in_range(first, last))
        if (field, ticker) not in raw:
            raise ValueError(f'Missing required market field: {field} {ticker}.')
        series = raw[(field, ticker)]
        values = series.reindex(expected)
        missing = values.index[~np.isfinite(values) | values.le(0)]
        if len(missing):
            errors.append(f'{ticker} {field}: {len(missing)} missing/invalid session(s), e.g. {missing[0].date()}')
            valid = series.loc[:end]
            valid = valid.loc[np.isfinite(valid) & valid.gt(0)]
            if len(valid) and valid.index[-1] < missing[0]:
                cutoffs.append(missing[0] - pd.Timedelta(days=1))
            else:
                interior_gap = True
        actual = series.loc[start:end].dropna()
        rows.append({'Ticker': ticker, 'Field': field,
            'Calendar': f'{name} (FX minimum-coverage proxy)' if ticker.endswith('=X') else name,
            'Checked Start': str(start.date()), 'Checked End': str(end.date()),
            'Expected Sessions': len(expected), 'Missing Sessions': len(missing),
            'Last Observed Date': str(actual.index.max().date()) if len(actual) else None})
    if errors:
        raise QuoteCoverageError(
            'Observed quote coverage is incomplete; forward-filled prices '
            'cannot establish completion. Refresh market data. ' + '; '.join(errors),
            safe_end=min(cutoffs) if cutoffs and not interior_gap else None)
    return pd.DataFrame(rows)
