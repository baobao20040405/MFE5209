"""Yahoo Finance downloads with a parameter-specific local snapshot."""
from pathlib import Path
import hashlib
import json

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent


def download_market_data(tickers, *, refresh=False, **kwargs):
    cache = ROOT / 'data' / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    parameters = {'tickers': list(tickers), **kwargs}
    key = hashlib.sha256(json.dumps(parameters, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = cache / f'market_{key}.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    frame = yf.download(tickers, **kwargs)
    if frame.empty:
        raise RuntimeError('Market download is empty. Check network access and ticker availability.')
    frame.to_csv(path, index_label='Date')
    (cache / f'market_{key}.json').write_text(json.dumps(parameters, indent=2, default=str), encoding='utf-8')
    return frame
