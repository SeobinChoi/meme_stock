#!/usr/bin/env python3
"""
Process Reddit archive data (2021-2023) from multiple subreddits.

Creates unified daily aggregated Reddit data that can be used
in place of BigQuery collection. Processes:
- wallstreetbets, cryptocurrency, stocks, investing, options, stockmarket
- Filters for meme stock tickers: GME, AMC, BB, BBBY, KOSS
- Filters for crypto tickers: DOGE, SHIB, PEPE, BONK, BTC, ETH
- Aggregates daily mentions across all subreddits

Output: data/processed/reddit/reddit_archive_daily_2021_2023.csv
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import argparse
import sys
import json
from datetime import datetime

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


def load_subreddit_year(subreddit: str, year: str, tickers: list) -> pd.DataFrame:
    """Load and process one subreddit/year CSV file."""
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
        
        # Find date columns (exclude meta columns)
        date_cols = [c for c in df.columns if c not in ('ticker', 'overall_rank', 'total')]
        date_cols = [c for c in date_cols if '/' in str(c)]  # Filter for date-like columns
        
        if not date_cols:
            print(f"❌ No date columns found in {archive_path}")
            return pd.DataFrame()
        
        # Melt from wide to long format
        melted = df.melt(
            id_vars=['ticker'], 
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
        
        print(f"✅ Processed {len(melted)} daily ticker mentions from {subreddit}_{year}")
        return melted[['date', 'ticker', 'mentions', 'subreddit', 'year']]
        
    except Exception as e:
        print(f"❌ Error processing {archive_path}: {e}")
        return pd.DataFrame()


def process_all_archive_data(years: list = None, subreddits: list = None, tickers: list = None) -> pd.DataFrame:
    """Process all archive data and return aggregated daily dataset."""
    
    years = years or ['2021', '2022', '2023']
    subreddits = subreddits or SUBREDDITS
    tickers = tickers or ALL_TICKERS
    
    print(f"🚀 Processing archive data:")
    print(f"   Years: {years}")
    print(f"   Subreddits: {subreddits}")
    print(f"   Tickers: {tickers}")
    print()
    
    all_data = []
    
    for year in years:
        for subreddit in subreddits:
            df = load_subreddit_year(subreddit, year, tickers)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        raise RuntimeError("❌ No data was successfully loaded from any files!")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined data: {len(combined)} records")
    
    # Aggregate by date (sum mentions across all subreddits and tickers per day)
    daily_agg = combined.groupby('date').agg({
        'mentions': 'sum',
        'ticker': 'count',  # count of ticker-subreddit combinations per day
        'subreddit': 'nunique'  # number of unique subreddits per day
    }).reset_index()
    
    # Rename columns to match existing schema
    daily_agg.rename(columns={
        'mentions': 'score',
        'ticker': 'daily_records',
        'subreddit': 'subreddits_active'
    }, inplace=True)
    
    # Add synthetic fields for compatibility
    daily_agg['num_comments'] = 0  # Not available in archive data
    daily_agg['total_engagement'] = daily_agg['score']
    daily_agg['title_length'] = 0
    daily_agg['word_count'] = 0
    daily_agg['is_weekend'] = daily_agg['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Sort by date
    daily_agg = daily_agg.sort_values('date').reset_index(drop=True)
    
    # Format date as string
    daily_agg['date'] = daily_agg['date'].dt.strftime('%Y-%m-%d')
    
    print(f"📈 Final aggregated dataset: {len(daily_agg)} days")
    print(f"   Date range: {daily_agg['date'].min()} to {daily_agg['date'].max()}")
    print(f"   Total mentions: {daily_agg['score'].sum():,}")
    print(f"   Average daily mentions: {daily_agg['score'].mean():.1f}")
    
    return daily_agg


def main():
    parser = argparse.ArgumentParser(description='Process Reddit archive data')
    parser.add_argument('--years', nargs='+', default=['2021', '2022', '2023'],
                        help='Years to process')
    parser.add_argument('--subreddits', nargs='+', default=SUBREDDITS,
                        help='Subreddits to process')
    parser.add_argument('--tickers', nargs='+', default=ALL_TICKERS,
                        help='Tickers to filter for')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Process the data
    daily_data = process_all_archive_data(args.years, args.subreddits, args.tickers)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = DATA_DIR / 'processed' / 'reddit'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        year_range = f"{min(args.years)}_{max(args.years)}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'reddit_archive_daily_{year_range}_{timestamp}.csv'
    
    # Save the data
    daily_data.to_csv(output_path, index=False)
    
    # Save metadata
    metadata = {
        'source': 'Reddit archive data (archive-3)',
        'years': args.years,
        'subreddits': args.subreddits,
        'tickers': args.tickers,
        'date_range': [daily_data['date'].min(), daily_data['date'].max()],
        'total_days': len(daily_data),
        'total_mentions': int(daily_data['score'].sum()),
        'processing_date': datetime.now().isoformat(),
        'columns': list(daily_data.columns)
    }
    
    meta_path = output_path.with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Saved processed data:")
    print(f"   📄 Data: {output_path}")
    print(f"   📋 Metadata: {meta_path}")
    print(f"\n🎯 Ready for ML pipeline integration!")


if __name__ == '__main__':
    main()
