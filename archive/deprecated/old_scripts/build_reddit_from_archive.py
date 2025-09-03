#!/usr/bin/env python3
"""
Build a daily Reddit dataset from archive-3 yearly CSVs (2021-2023) for WSB.

Output schema matches the loader expectation minimally:
- date: YYYY-MM-DD
- score: daily proxy of WSB mentions (sum over target tickers)
- num_comments: set to 0 (unknown)

This extends coverage beyond the single-year raw file by synthesizing a
daily aggregate that downstream code can ingest.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List
import sys

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_year(file_path: Path, tickers: List[str]) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Archive file not found: {file_path}")
    df = pd.read_csv(file_path)
    # Ensure ticker column exists
    if 'ticker' not in df.columns:
        raise ValueError(f"Unexpected schema in {file_path}: missing 'ticker'")
    # Keep only desired tickers
    df = df[df['ticker'].isin(tickers)].copy()
    if df.empty:
        # No target tickers in this file
        return pd.DataFrame(columns=['date', 'mentions'])

    # Melt daily columns: they are like '1/1/21', '1/2/21', ...
    value_cols = [c for c in df.columns if c not in ('ticker', 'overall_rank', 'total')]
    melted = df.melt(id_vars=['ticker'], value_vars=value_cols, var_name='date_str', value_name='mentions')
    melted = melted.dropna(subset=['date_str'])
    # Some files might use different date headers; filter plausible date strings
    melted = melted[melted['date_str'].astype(str).str.contains('/')]
    # Convert to datetime; coerce errors to NaT then drop
    melted['date'] = pd.to_datetime(melted['date_str'], errors='coerce')
    melted = melted.dropna(subset=['date'])
    # Mentions to numeric
    melted['mentions'] = pd.to_numeric(melted['mentions'], errors='coerce').fillna(0).astype(int)
    # Aggregate across tickers for the day
    daily = melted.groupby('date', as_index=False)['mentions'].sum()
    return daily


def build_daily_wsbreddit(tickers: List[str] = None) -> pd.DataFrame:
    if tickers is None:
        tickers = ['GME', 'AMC', 'BB']

    archive_root = DATA_DIR / 'raw' / 'archive-3'
    years = ['2021', '2022', '2023']
    all_daily = []
    for y in years:
        fp = archive_root / y / f'wallstreetbets_{y}.csv'
        try:
            daily = _load_year(fp, tickers)
            if not daily.empty:
                all_daily.append(daily)
        except FileNotFoundError:
            # Skip missing years gracefully
            continue

    if not all_daily:
        raise RuntimeError("No archive-3 yearly files found for any specified years.")

    combined = pd.concat(all_daily, ignore_index=True)
    combined = combined.groupby('date', as_index=False)['mentions'].sum()
    # Synthesize required fields for downstream loader
    combined = combined.sort_values('date').reset_index(drop=True)
    combined['score'] = combined['mentions']
    combined['num_comments'] = 0
    combined['total_engagement'] = combined['score'] + combined['num_comments']
    combined['title_length'] = 0
    combined['word_count'] = 0
    combined['is_weekend'] = combined['date'].dt.dayofweek.isin([5, 6]).astype(int)
    out = combined[['date', 'score', 'num_comments', 'total_engagement', 'title_length', 'word_count', 'is_weekend']].copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    return out


def main():
    out_file = DATA_DIR / 'raw' / 'reddit_wsb.csv'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = build_daily_wsbreddit()
    df.to_csv(out_file, index=False)
    print(f"✅ Built daily Reddit WSB dataset: {out_file} ({len(df)} rows)")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == '__main__':
    sys.exit(main())


