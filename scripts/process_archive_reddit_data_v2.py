#!/usr/bin/env python3
"""
Process Reddit archive data (2021-2023) from multiple subreddits with ticker-level detail.

Enhanced version that preserves ticker-level information:
- Maintains individual ticker mention counts by date
- Includes overall_rank information
- Creates both aggregated and detailed views
- Generates comprehensive metadata

Output files:
- data/processed/reddit/reddit_archive_ticker_daily_YYYY_YYYY_timestamp.csv (detailed)
- data/processed/reddit/reddit_archive_aggregated_daily_YYYY_YYYY_timestamp.csv (summary)
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import argparse
import sys
import json
from datetime import datetime
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Target subreddits (available in archive-3)
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


def load_subreddit_year_detailed(subreddit: str, year: str, tickers: list) -> pd.DataFrame:
    """Load and process one subreddit/year CSV file with full ticker detail."""
    archive_path = DATA_DIR / 'raw' / 'archive' / 'archive-3' / year / f'{subreddit}_{year}.csv'
    
    if not archive_path.exists():
        print(f"⚠️  Missing: {archive_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(archive_path)
        print(f"📂 Loaded {archive_path} ({len(df)} tickers)")
        
        # Ensure required columns
        if 'ticker' not in df.columns:
            print(f"❌ No ticker column in {archive_path}")
            return pd.DataFrame()
        
        # Filter for target tickers
        df = df[df['ticker'].isin(tickers)].copy()
        if df.empty:
            print(f"📋 No target tickers found in {archive_path}")
            return pd.DataFrame()
        
        # Preserve overall_rank if available
        has_rank = 'overall_rank' in df.columns
        
        # Find date columns (exclude meta columns)
        meta_cols = ['ticker', 'overall_rank', 'total']
        date_cols = [c for c in df.columns if c not in meta_cols]
        date_cols = [c for c in date_cols if '/' in str(c)]  # Filter for date-like columns
        
        if not date_cols:
            print(f"❌ No date columns found in {archive_path}")
            return pd.DataFrame()
        
        # Calculate total mentions per ticker if not available
        if 'total' not in df.columns:
            df['total'] = df[date_cols].sum(axis=1)
        
        # Melt from wide to long format
        id_vars = ['ticker']
        if has_rank:
            id_vars.append('overall_rank')
        id_vars.append('total')
        
        melted = df.melt(
            id_vars=id_vars, 
            value_vars=date_cols, 
            var_name='date_str', 
            value_name='mentions'
        )
        
        # Clean and parse dates
        melted = melted.dropna(subset=['date_str'])
        melted['date'] = pd.to_datetime(melted['date_str'], errors='coerce')
        melted = melted.dropna(subset=['date'])
        
        # Clean mentions data
        melted['mentions'] = pd.to_numeric(melted['mentions'], errors='coerce').fillna(0).astype(int)
        
        # Add metadata
        melted['subreddit'] = subreddit
        melted['year'] = year
        
        # Calculate daily rank within subreddit (if not provided)
        if not has_rank:
            melted['overall_rank'] = melted.groupby('date')['mentions'].rank(
                method='dense', ascending=False
            ).astype(int)
        
        # Calculate additional metrics
        melted['ticker_type'] = melted['ticker'].apply(
            lambda x: 'meme_stock' if x in MEME_TICKERS else 'crypto'
        )
        
        print(f"✅ Processed {len(melted)} daily ticker mentions from {subreddit}_{year}")
        
        columns = ['date', 'ticker', 'mentions', 'overall_rank', 'total', 
                   'subreddit', 'year', 'ticker_type']
        return melted[columns]
        
    except Exception as e:
        print(f"❌ Error processing {archive_path}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def create_ticker_aggregations(detailed_df: pd.DataFrame) -> dict:
    """Create various aggregations from detailed ticker data."""
    
    aggregations = {}
    
    # 1. Daily aggregation across all tickers and subreddits
    daily_total = detailed_df.groupby('date').agg({
        'mentions': 'sum',
        'ticker': 'nunique',
        'subreddit': 'nunique'
    }).reset_index()
    
    daily_total.rename(columns={
        'mentions': 'total_mentions',
        'ticker': 'unique_tickers',
        'subreddit': 'active_subreddits'
    }, inplace=True)
    
    aggregations['daily_total'] = daily_total
    
    # 2. Daily ticker aggregation (across all subreddits)
    daily_ticker = detailed_df.groupby(['date', 'ticker', 'ticker_type']).agg({
        'mentions': 'sum',
        'overall_rank': 'mean',  # Average rank across subreddits
        'subreddit': 'count'     # Number of subreddits where mentioned
    }).reset_index()
    
    daily_ticker.rename(columns={
        'subreddit': 'subreddit_count',
        'overall_rank': 'avg_rank'
    }, inplace=True)
    
    # Calculate daily rank across all subreddits
    daily_ticker['daily_rank'] = daily_ticker.groupby('date')['mentions'].rank(
        method='dense', ascending=False
    ).astype(int)
    
    aggregations['daily_ticker'] = daily_ticker
    
    # 3. Ticker summary statistics
    ticker_stats = detailed_df.groupby(['ticker', 'ticker_type']).agg({
        'mentions': ['sum', 'mean', 'std', 'max'],
        'overall_rank': 'mean',
        'date': ['min', 'max', 'count']
    }).round(2)
    
    ticker_stats.columns = ['_'.join(col).strip() for col in ticker_stats.columns.values]
    ticker_stats = ticker_stats.reset_index()
    
    aggregations['ticker_stats'] = ticker_stats
    
    # 4. Top ticker days (highest mention counts)
    top_days = daily_ticker.nlargest(20, 'mentions')[
        ['date', 'ticker', 'mentions', 'daily_rank', 'subreddit_count']
    ].copy()
    
    aggregations['top_ticker_days'] = top_days
    
    # 5. Ticker momentum (7-day moving average)
    daily_ticker_sorted = daily_ticker.sort_values(['ticker', 'date'])
    daily_ticker_sorted['mentions_ma7'] = daily_ticker_sorted.groupby('ticker')['mentions'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    ).round(2)
    
    aggregations['ticker_momentum'] = daily_ticker_sorted
    
    return aggregations


def process_all_archive_data_v2(years: list = None, subreddits: list = None, tickers: list = None) -> tuple:
    """Process all archive data and return both detailed and aggregated datasets."""
    
    years = years or ['2021', '2022', '2023']
    subreddits = subreddits or SUBREDDITS
    tickers = tickers or ALL_TICKERS
    
    print(f"🚀 Processing archive data (Enhanced Version):")
    print(f"   Years: {years}")
    print(f"   Subreddits: {subreddits}")
    print(f"   Tickers: {tickers}")
    print()
    
    all_data = []
    
    # Load all data with full detail
    for year in years:
        for subreddit in subreddits:
            df = load_subreddit_year_detailed(subreddit, year, tickers)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        raise RuntimeError("❌ No data was successfully loaded from any files!")
    
    # Combine all detailed data
    detailed = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined detailed data: {len(detailed)} records")
    
    # Create aggregations
    aggregations = create_ticker_aggregations(detailed)
    
    # Create backward-compatible daily aggregation
    daily_agg = aggregations['daily_total'].copy()
    
    # Add synthetic fields for compatibility
    daily_agg['score'] = daily_agg['total_mentions']
    daily_agg['num_comments'] = 0  # Not available in archive data
    daily_agg['total_engagement'] = daily_agg['total_mentions']
    daily_agg['title_length'] = 0
    daily_agg['word_count'] = 0
    daily_agg['is_weekend'] = daily_agg['date'].dt.dayofweek.isin([5, 6]).astype(int)
    daily_agg['daily_records'] = daily_agg['unique_tickers'] * daily_agg['active_subreddits']
    daily_agg['subreddits_active'] = daily_agg['active_subreddits']
    
    # Sort by date
    daily_agg = daily_agg.sort_values('date').reset_index(drop=True)
    
    # Format dates
    detailed['date_str'] = detailed['date'].dt.strftime('%Y-%m-%d')
    daily_agg['date'] = daily_agg['date'].dt.strftime('%Y-%m-%d')
    
    print(f"\n📈 Processing complete:")
    print(f"   Detailed records: {len(detailed)}")
    print(f"   Daily aggregated records: {len(daily_agg)}")
    print(f"   Date range: {daily_agg['date'].min()} to {daily_agg['date'].max()}")
    print(f"   Total mentions: {daily_agg['score'].sum():,}")
    print(f"   Unique tickers tracked: {detailed['ticker'].nunique()}")
    
    return detailed, daily_agg, aggregations


def save_outputs(detailed_df, daily_agg_df, aggregations, args, output_dir):
    """Save all outputs with proper naming and metadata."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    year_range = f"{min(args.years)}_{max(args.years)}"
    
    saved_files = {}
    
    # 1. Save detailed ticker-level data
    ticker_path = output_dir / f'reddit_archive_ticker_daily_{year_range}_{timestamp}.csv'
    detailed_save = detailed_df.copy()
    detailed_save['date'] = detailed_save['date_str']
    columns_order = ['date', 'ticker', 'ticker_type', 'mentions', 'overall_rank', 
                     'total', 'subreddit', 'year']
    detailed_save[columns_order].to_csv(ticker_path, index=False)
    saved_files['ticker_daily'] = str(ticker_path)
    print(f"   📄 Ticker daily data: {ticker_path}")
    
    # 2. Save backward-compatible aggregated data
    agg_path = output_dir / f'reddit_archive_daily_{year_range}_{timestamp}.csv'
    daily_agg_df.to_csv(agg_path, index=False)
    saved_files['daily_aggregated'] = str(agg_path)
    print(f"   📄 Daily aggregated data: {agg_path}")
    
    # 3. Save ticker-level aggregations
    ticker_summary_path = output_dir / f'reddit_archive_ticker_summary_{year_range}_{timestamp}.csv'
    aggregations['daily_ticker']['date'] = aggregations['daily_ticker']['date'].dt.strftime('%Y-%m-%d')
    aggregations['daily_ticker'].to_csv(ticker_summary_path, index=False)
    saved_files['ticker_summary'] = str(ticker_summary_path)
    print(f"   📄 Ticker summary data: {ticker_summary_path}")
    
    # 4. Save ticker statistics
    stats_path = output_dir / f'reddit_archive_ticker_stats_{year_range}_{timestamp}.csv'
    aggregations['ticker_stats'].to_csv(stats_path, index=False)
    saved_files['ticker_stats'] = str(stats_path)
    print(f"   📄 Ticker statistics: {stats_path}")
    
    # 5. Save top ticker days
    top_days_path = output_dir / f'reddit_archive_top_ticker_days_{year_range}_{timestamp}.csv'
    top_days = aggregations['top_ticker_days'].copy()
    top_days['date'] = top_days['date'].dt.strftime('%Y-%m-%d')
    top_days.to_csv(top_days_path, index=False)
    saved_files['top_ticker_days'] = str(top_days_path)
    print(f"   📄 Top ticker days: {top_days_path}")
    
    # 6. Save comprehensive metadata
    metadata = {
        'processing_info': {
            'source': 'Reddit archive data (archive-3)',
            'version': '2.0',
            'processing_date': datetime.now().isoformat(),
            'script': 'process_archive_reddit_data_v2.py'
        },
        'data_coverage': {
            'years': args.years,
            'subreddits': args.subreddits,
            'tickers': {
                'meme_stocks': MEME_TICKERS,
                'crypto': CRYPTO_TICKERS,
                'total': args.tickers
            },
            'date_range': [daily_agg_df['date'].min(), daily_agg_df['date'].max()],
            'total_days': len(daily_agg_df)
        },
        'statistics': {
            'total_mentions': int(daily_agg_df['score'].sum()),
            'unique_tickers': int(detailed_df['ticker'].nunique()),
            'total_records': len(detailed_df),
            'average_daily_mentions': float(daily_agg_df['score'].mean()),
            'peak_day': {
                'date': daily_agg_df.loc[daily_agg_df['score'].idxmax(), 'date'],
                'mentions': int(daily_agg_df['score'].max())
            }
        },
        'ticker_breakdown': {
            ticker: {
                'total_mentions': int(detailed_df[detailed_df['ticker'] == ticker]['mentions'].sum()),
                'days_active': int(detailed_df[detailed_df['ticker'] == ticker]['date'].nunique()),
                'avg_daily_mentions': float(
                    detailed_df[detailed_df['ticker'] == ticker].groupby('date')['mentions'].sum().mean()
                )
            } for ticker in args.tickers
        },
        'output_files': saved_files,
        'schemas': {
            'ticker_daily': list(detailed_save[columns_order].columns),
            'daily_aggregated': list(daily_agg_df.columns),
            'ticker_summary': list(aggregations['daily_ticker'].columns)
        }
    }
    
    meta_path = output_dir / f'reddit_archive_metadata_{year_range}_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   📋 Metadata: {meta_path}")
    
    return saved_files, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Process Reddit archive data with ticker-level detail preservation'
    )
    parser.add_argument('--years', nargs='+', default=['2021', '2022', '2023'],
                        help='Years to process')
    parser.add_argument('--subreddits', nargs='+', default=SUBREDDITS,
                        help='Subreddits to process')
    parser.add_argument('--tickers', nargs='+', default=ALL_TICKERS,
                        help='Tickers to filter for')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: data/processed/reddit)')
    
    args = parser.parse_args()
    
    # Process the data
    detailed_data, daily_data, aggregations = process_all_archive_data_v2(
        args.years, args.subreddits, args.tickers
    )
    
    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / 'processed' / 'reddit'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all outputs
    print(f"\n✅ Saving processed data:")
    saved_files, metadata = save_outputs(
        detailed_data, daily_data, aggregations, args, output_dir
    )
    
    print(f"\n🎯 Processing complete! Generated {len(saved_files)} output files.")
    print(f"📊 Key insights:")
    print(f"   - Most mentioned ticker: {max(metadata['ticker_breakdown'].items(), key=lambda x: x[1]['total_mentions'])[0]}")
    print(f"   - Peak activity day: {metadata['statistics']['peak_day']['date']}")
    print(f"   - Average daily mentions: {metadata['statistics']['average_daily_mentions']:.1f}")


if __name__ == '__main__':
    main()