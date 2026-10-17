"""Parse unedited MAS SORA exports and align only published rates to returns."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RATE_COLUMN = 'Annualized SORA interest rate available for retrospect (%)'


def process_sora(raw_file=None, output_dir=None):
    """Convert MAS's grouped year/month/day CSV into validated tidy observations.

    Preserve the annual percentage and BOTH value/publication dates. Parsing is
    structural; metadata/footer lines are excluded, never interpreted as data.
    """
    path = Path(raw_file or ROOT / 'data' / 'raw' / 'raw sora.csv')
    if not path.exists():
        raise FileNotFoundError(f'Place the unedited MAS export at {path}')
    rows = list(csv.reader(path.read_text(encoding='utf-8-sig').splitlines()))
    header = next((i for i, row in enumerate(rows)
                   if row and row[0].strip() == 'SORA Value Date'), None)
    if header is None:
        raise ValueError('Expected MAS SORA Value Date / SORA Publication Date header.')
    year = month = None
    records = []
    for row in rows[header + 1:]:
        if len(row) < 5:
            continue
        y, m, day, publication, rate = [value.strip() for value in row[:5]]
        if not day.isdigit():
            continue
        if y:
            year = y
        if m:
            month = m
        if year is None or month is None:
            raise ValueError('SORA date appears before its year/month group.')
        try:
            value_date = pd.to_datetime(f'{year}-{month}-{day}', format='%Y-%b-%d')
            published = pd.to_datetime(publication, format='%d %b %Y')
            value = float(rate.replace(',', ''))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid SORA data row: {row}') from exc
        if not np.isfinite(value) or value <= -100 or published <= value_date:
            raise ValueError(f'Invalid rate or publication ordering: {row}')
        records.append((value_date, published, value))
    frame = pd.DataFrame(records, columns=['Date', 'Publication Date', 'SORA']).sort_values('Date')
    if frame.empty or frame['Date'].duplicated().any() or frame['Publication Date'].duplicated().any():
        raise ValueError('SORA observations are empty or contain duplicate dates.')
    out = Path(output_dir or ROOT / 'data' / 'processed')
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'sora_publication_audit.csv', index=False)
    frame[['Date', 'SORA']].to_csv(out / 'SORA_daily.csv', index=False)
    (out / 'sora_processing.json').write_text(json.dumps({
        'source': 'https://eservices.mas.gov.sg/statistics/dir/',
        'raw_file': str(path.resolve()), 'observations': len(frame),
        'first_value_date': str(frame['Date'].min().date()),
        'last_value_date': str(frame['Date'].max().date()),
        'last_publication_date': str(frame['Publication Date'].max().date()),
        'unit': 'Annual percentage; 1.50 means 1.50%, not 150%.',
        'alignment': 'Last observation published on or before the return date; never future-filled.',
        'day_count': 'Simple ACT/365 over each valuation interval. This is a daily cash-rate proxy, not the official compounded SORA Index.',
    }, indent=2), encoding='utf-8')
    return frame


def load_sora_daily(max_date, raw_file=None):
    frame = process_sora(raw_file)
    frame = frame.loc[frame['Publication Date'] <= pd.Timestamp(max_date)].copy()
    if frame.empty:
        raise ValueError('No SORA observations published by the requested endpoint.')
    frame[RATE_COLUMN] = frame['SORA']
    return frame.set_index('Publication Date').sort_index()


def align_sora_returns(index, sora_frame, allow_holdout=False, annual_spread=0.0):
    """Use the last published prior-value-date rate, with actual calendar gaps.

    MAS publishes the previous Singapore business day's rate by 9am. Publication
    date, rather than a second arbitrary shift, establishes availability here.
    """
    dates = pd.DatetimeIndex(index)
    if dates.empty:
        return pd.Series(dtype=float, index=dates)
    source = sora_frame.sort_index()
    aligned = source.reindex(dates, method='ffill')
    if aligned[RATE_COLUMN].isna().any():
        raise ValueError('SORA coverage starts after the first required return date.')
    positions = source.index.get_indexer(dates, method='ffill')
    publications = source.index[positions]
    if (publications > dates).any() or (aligned['Date'].to_numpy() >= dates.to_numpy()).any():
        raise ValueError('SORA alignment would use an unpublished or same-value-date rate.')
    age = (dates - publications).days
    if (age > 7).any():
        raise ValueError('SORA is stale by more than 7 calendar days. Update data/raw/raw sora.csv.')
    days = dates.to_series().diff().dt.days.astype(float)
    days.iloc[0] = 1.0
    if (days <= 0).any():
        raise ValueError('Return dates must be unique and increasing.')
    values = (aligned[RATE_COLUMN].to_numpy() / 100 + annual_spread) * days.to_numpy() / 365
    return pd.Series(values, index=dates, name='Daily Risk-Free Return')


def sora_alignment_audit(index, frame):
    dates = pd.DatetimeIndex(index)
    aligned = frame.reindex(dates, method='ffill')
    positions = frame.index.get_indexer(dates, method='ffill')
    return pd.DataFrame({
        'SORA Value Date': aligned['Date'].to_numpy(),
        'SORA Publication Date': frame.index[positions],
        'Annual SORA (%)': aligned[RATE_COLUMN].to_numpy(),
        'Daily Risk-Free Return': align_sora_returns(dates, frame).to_numpy(),
    }, index=dates).rename_axis('Return Date')
