#!/usr/bin/env python3
"""
Process Reddit archive data for Machine Learning/Deep Learning applications.

Generates multiple formats optimized for different use cases:
1. Parquet format - Efficient storage and fast filtering
2. Full CSV - Complete granular data for custom preprocessing  
3. ML-ready CSV formats - Pre-processed features for immediate use

Includes additional features useful for DL:
- Time-based features (day of week, month, quarter)
- Lag features (previous day mentions)
- Rolling statistics (7-day, 30-day averages)
- Mention velocity and acceleration
- Cross-subreddit correlation features
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Tuple

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Target subreddits
SUBREDDITS = [
    'wallstreetbets',
    'cryptocurrency', 
    'stocks',
    'investing',
    'options',
    'stockmarket',
    'pennystocks'
]

# Target tickers
MEME_TICKERS = ['GME', 'AMC', 'BB', 'BBBY', 'KOSS']
CRYPTO_TICKERS = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'BTC', 'ETH']
ALL_TICKERS = MEME_TICKERS + CRYPTO_TICKERS


def load_all_archive_data(years: List[str], subreddits: List[str]) -> pd.DataFrame:
    """Load all archive data into a single DataFrame."""
    all_data = []
    
    for year in years:
        for subreddit in subreddits:
            archive_path = DATA_DIR / 'raw' / 'archive' / 'archive-3' / year / f'{subreddit}_{year}.csv'
            
            if not archive_path.exists():
                print(f"⚠️  Missing: {archive_path}")
                continue
            
            try:
                df = pd.read_csv(archive_path)
                print(f"📂 Loading {archive_path} ({len(df)} tickers)")
                
                if 'ticker' not in df.columns:
                    continue
                
                # Filter for target tickers
                df = df[df['ticker'].isin(ALL_TICKERS)].copy()
                if df.empty:
                    continue
                
                # Get date columns
                meta_cols = ['ticker', 'overall_rank', 'total']
                date_cols = [c for c in df.columns if c not in meta_cols and '/' in str(c)]
                
                # Melt to long format
                id_vars = ['ticker']
                if 'overall_rank' in df.columns:
                    id_vars.append('overall_rank')
                if 'total' in df.columns:
                    id_vars.append('total')
                
                melted = df.melt(
                    id_vars=id_vars,
                    value_vars=date_cols,
                    var_name='date_str',
                    value_name='mentions'
                )
                
                # Parse dates
                melted['date'] = pd.to_datetime(melted['date_str'], format='%m/%d/%y', errors='coerce')
                melted = melted.dropna(subset=['date'])
                
                # Clean data
                melted['mentions'] = pd.to_numeric(melted['mentions'], errors='coerce').fillna(0).astype(int)
                melted['subreddit'] = subreddit
                melted['year'] = int(year)
                
                all_data.append(melted)
                print(f"✅ Processed {len(melted)} records from {subreddit}_{year}")
                
            except Exception as e:
                print(f"❌ Error processing {archive_path}: {e}")
                continue
    
    if not all_data:
        raise RuntimeError("No data loaded!")
    
    # Combine all data
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Add ticker classification
    full_df['ticker_type'] = full_df['ticker'].apply(
        lambda x: 'meme_stock' if x in MEME_TICKERS else 'crypto'
    )
    
    # Sort by date, ticker, subreddit
    full_df = full_df.sort_values(['date', 'ticker', 'subreddit']).reset_index(drop=True)
    
    return full_df


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features useful for ML/DL models."""
    print("\n🔧 Adding ML features...")
    
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Sort for time series operations
    df = df.sort_values(['ticker', 'subreddit', 'date']).reset_index(drop=True)
    
    # Lag features (previous mentions)
    for lag in [1, 3, 7, 14, 30]:
        df[f'mentions_lag_{lag}'] = df.groupby(['ticker', 'subreddit'])['mentions'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'mentions_ma_{window}'] = df.groupby(['ticker', 'subreddit'])['mentions'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'mentions_std_{window}'] = df.groupby(['ticker', 'subreddit'])['mentions'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        df[f'mentions_max_{window}'] = df.groupby(['ticker', 'subreddit'])['mentions'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
    
    # Mention velocity (rate of change)
    df['mention_velocity'] = df.groupby(['ticker', 'subreddit'])['mentions'].diff()
    df['mention_acceleration'] = df.groupby(['ticker', 'subreddit'])['mention_velocity'].diff()
    
    # Normalized features (z-score within ticker-subreddit)
    df['mentions_zscore'] = df.groupby(['ticker', 'subreddit'])['mentions'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # Percentage of total daily mentions
    daily_total = df.groupby(['date', 'subreddit'])['mentions'].transform('sum')
    df['mention_share'] = (df['mentions'] / (daily_total + 1e-8)) * 100
    
    # Ticker momentum score (mentions relative to 30-day average)
    df['momentum_score'] = df['mentions'] / (df['mentions_ma_30'] + 1e-8)
    
    # Binary spike detection (mentions > 2 * 7-day average)
    df['is_spike'] = (df['mentions'] > 2 * df['mentions_ma_7']).astype(int)
    
    # Rank features
    df['daily_rank_in_subreddit'] = df.groupby(['date', 'subreddit'])['mentions'].rank(
        method='dense', ascending=False
    )
    df['is_top_5'] = (df['daily_rank_in_subreddit'] <= 5).astype(int)
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def create_ml_datasets(full_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create various ML-ready datasets."""
    datasets = {}
    
    # 1. Time series format (one row per ticker-date)
    ts_df = full_df.groupby(['date', 'ticker', 'ticker_type']).agg({
        'mentions': 'sum',
        'subreddit': 'count',  # number of subreddits mentioning
        'overall_rank': 'mean',
        'is_weekend': 'first',
        'day_of_week': 'first',
        'month': 'first',
        'quarter': 'first'
    }).reset_index()
    ts_df.rename(columns={'subreddit': 'subreddit_count'}, inplace=True)
    datasets['timeseries'] = ts_df
    
    # 2. Wide format (subreddits as columns)
    pivot_df = full_df.pivot_table(
        index=['date', 'ticker', 'ticker_type'],
        columns='subreddit',
        values='mentions',
        fill_value=0
    ).reset_index()
    pivot_df['total_mentions'] = pivot_df[SUBREDDITS].sum(axis=1)
    datasets['wide_format'] = pivot_df
    
    # 3. Sequence format for LSTM/Transformer (windows of data)
    # Create 30-day sequences
    sequence_data = []
    tickers = full_df['ticker'].unique()
    dates = sorted(full_df['date'].unique())
    
    for ticker in tickers:
        ticker_df = full_df[full_df['ticker'] == ticker].copy()
        
        for i in range(30, len(dates)):
            # Get 30-day window
            window_dates = dates[i-30:i]
            window_df = ticker_df[ticker_df['date'].isin(window_dates)]
            
            if len(window_df) > 0:
                # Aggregate features for the window
                seq_features = {
                    'ticker': ticker,
                    'end_date': dates[i-1],
                    'target_date': dates[i] if i < len(dates) else None,
                    'mentions_mean_30d': window_df['mentions'].mean(),
                    'mentions_std_30d': window_df['mentions'].std(),
                    'mentions_max_30d': window_df['mentions'].max(),
                    'mentions_sum_30d': window_df['mentions'].sum(),
                    'active_days_30d': window_df[window_df['mentions'] > 0]['date'].nunique(),
                    'active_subreddits_30d': window_df[window_df['mentions'] > 0]['subreddit'].nunique(),
                    'momentum_30d': window_df['mentions'].iloc[-7:].mean() / (window_df['mentions'].mean() + 1e-8)
                }
                sequence_data.append(seq_features)
    
    if sequence_data:
        datasets['sequences'] = pd.DataFrame(sequence_data)
    
    # 4. Daily aggregate with all features
    daily_features = full_df.groupby('date').agg({
        'mentions': ['sum', 'mean', 'std', 'max'],
        'ticker': 'nunique',
        'subreddit': 'nunique',
        'is_spike': 'sum',
        'mention_velocity': 'mean',
        'momentum_score': 'mean'
    }).reset_index()
    daily_features.columns = ['_'.join(col).strip('_') for col in daily_features.columns]
    datasets['daily_aggregate'] = daily_features
    
    return datasets


def save_all_formats(full_df: pd.DataFrame, ml_datasets: Dict[str, pd.DataFrame], 
                     output_dir: Path, timestamp: str, year_range: str):
    """Save data in all formats."""
    
    # 1. Save Parquet format (most efficient)
    parquet_path = output_dir / f'reddit_mentions_full_{year_range}_{timestamp}.parquet'
    
    # Create table with metadata
    table = pa.Table.from_pandas(full_df)
    metadata = {
        b'created_date': datetime.now().isoformat().encode(),
        b'ticker_count': str(full_df['ticker'].nunique()).encode(),
        b'date_range': f"{full_df['date'].min()} to {full_df['date'].max()}".encode(),
        b'total_records': str(len(full_df)).encode()
    }
    table = table.replace_schema_metadata(metadata)
    
    pq.write_table(table, parquet_path, compression='snappy')
    print(f"📦 Saved Parquet: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 2. Save full CSV (complete data)
    csv_path = output_dir / f'reddit_mentions_full_{year_range}_{timestamp}.csv'
    full_df.to_csv(csv_path, index=False)
    print(f"📄 Saved Full CSV: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 3. Save ML-ready datasets
    for name, df in ml_datasets.items():
        ml_path = output_dir / f'reddit_ml_{name}_{year_range}_{timestamp}.csv'
        df.to_csv(ml_path, index=False)
        print(f"🤖 Saved ML {name}: {ml_path} ({len(df)} rows)")
    
    # 4. Save sample data for quick inspection
    sample_path = output_dir / f'reddit_sample_{year_range}_{timestamp}.csv'
    
    # Get sample: top 5 tickers, last 30 days
    top_tickers = full_df.groupby('ticker')['mentions'].sum().nlargest(5).index
    recent_dates = sorted(full_df['date'].unique())[-30:]
    sample_df = full_df[
        (full_df['ticker'].isin(top_tickers)) & 
        (full_df['date'].isin(recent_dates))
    ].copy()
    sample_df.to_csv(sample_path, index=False)
    print(f"📋 Saved Sample: {sample_path} ({len(sample_df)} rows)")
    
    # 5. Save metadata
    metadata = {
        'processing_info': {
            'script': 'process_archive_reddit_data_ml.py',
            'created': datetime.now().isoformat(),
            'years': year_range.split('_'),
            'tickers': {
                'meme_stocks': MEME_TICKERS,
                'crypto': CRYPTO_TICKERS
            }
        },
        'data_stats': {
            'total_records': len(full_df),
            'unique_dates': full_df['date'].nunique(),
            'date_range': [
                full_df['date'].min().strftime('%Y-%m-%d'),
                full_df['date'].max().strftime('%Y-%m-%d')
            ],
            'total_mentions': int(full_df['mentions'].sum()),
            'file_sizes_mb': {
                'parquet': round(parquet_path.stat().st_size / 1024 / 1024, 2),
                'full_csv': round(csv_path.stat().st_size / 1024 / 1024, 2)
            }
        },
        'ticker_stats': {
            ticker: {
                'total_mentions': int(full_df[full_df['ticker'] == ticker]['mentions'].sum()),
                'active_days': int(full_df[full_df['ticker'] == ticker]['date'].nunique()),
                'peak_day': full_df[full_df['ticker'] == ticker].nlargest(1, 'mentions').iloc[0]['date'].strftime('%Y-%m-%d'),
                'peak_mentions': int(full_df[full_df['ticker'] == ticker]['mentions'].max())
            } for ticker in sorted(full_df['ticker'].unique())
        },
        'ml_features': {
            'time_features': ['day_of_week', 'month', 'quarter', 'is_weekend'],
            'lag_features': [f'mentions_lag_{i}' for i in [1, 3, 7, 14, 30]],
            'rolling_features': [f'mentions_{stat}_{w}' for stat in ['ma', 'std', 'max'] for w in [7, 14, 30]],
            'derived_features': ['mention_velocity', 'mention_acceleration', 'momentum_score', 'is_spike']
        },
        'output_files': {
            'parquet': parquet_path.name,
            'full_csv': csv_path.name,
            'ml_datasets': {name: f'reddit_ml_{name}_{year_range}_{timestamp}.csv' 
                           for name in ml_datasets.keys()}
        }
    }
    
    meta_path = output_dir / f'reddit_ml_metadata_{year_range}_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📊 Saved Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process Reddit archive data for ML/DL applications'
    )
    parser.add_argument('--years', nargs='+', default=['2021', '2022', '2023'],
                        help='Years to process')
    parser.add_argument('--subreddits', nargs='+', default=SUBREDDITS,
                        help='Subreddits to process')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    print(f"🚀 Processing Reddit Archive Data for ML/DL")
    print(f"   Years: {args.years}")
    print(f"   Subreddits: {args.subreddits}")
    print()
    
    # Load all data
    full_df = load_all_archive_data(args.years, args.subreddits)
    print(f"\n📊 Loaded {len(full_df)} total records")
    print(f"   Date range: {full_df['date'].min()} to {full_df['date'].max()}")
    print(f"   Unique tickers: {full_df['ticker'].nunique()}")
    print(f"   Total mentions: {full_df['mentions'].sum():,}")
    
    # Add ML features
    full_df = add_ml_features(full_df)
    print(f"✅ Added {len(full_df.columns)} total features")
    
    # Create ML datasets
    ml_datasets = create_ml_datasets(full_df)
    print(f"\n📊 Created {len(ml_datasets)} ML-ready datasets:")
    for name, df in ml_datasets.items():
        print(f"   - {name}: {len(df)} rows, {len(df.columns)} columns")
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / 'processed' / 'reddit' / 'ml'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all formats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    year_range = f"{min(args.years)}_{max(args.years)}"
    
    print(f"\n💾 Saving all formats...")
    save_all_formats(full_df, ml_datasets, output_dir, timestamp, year_range)
    
    print(f"\n🎯 Processing complete!")
    print(f"📁 All files saved to: {output_dir}")
    
    # Show sample of the data
    print(f"\n📋 Sample of processed data:")
    print(full_df[['date', 'ticker', 'subreddit', 'mentions', 'mentions_ma_7', 
                   'momentum_score', 'is_spike']].head(10))


if __name__ == '__main__':
    main()