#!/usr/bin/env python3
"""
Process Reddit archive data (2021-2023) preserving ALL original data.

This script:
- Loads all archive CSV files without aggregation
- Preserves complete ticker x date x subreddit information
- Creates a unified dataset with all mention counts
- Maintains original granularity for detailed analysis

Output: Complete dataset with every ticker mention by date and subreddit
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import argparse
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


def load_and_transform_archive(subreddit: str, year: str) -> pd.DataFrame:
    """Load archive CSV and transform to long format preserving all data."""
    archive_path = DATA_DIR / 'raw' / 'archive' / 'archive-3' / year / f'{subreddit}_{year}.csv'
    
    if not archive_path.exists():
        print(f"⚠️  Missing: {archive_path}")
        return pd.DataFrame()
    
    try:
        # Load the CSV
        df = pd.read_csv(archive_path)
        print(f"📂 Loaded {archive_path} ({len(df)} tickers)")
        
        if 'ticker' not in df.columns:
            print(f"❌ No ticker column in {archive_path}")
            return pd.DataFrame()
        
        # Get all tickers (not just our target list) for complete data
        all_tickers_in_file = df['ticker'].unique()
        print(f"   Found {len(all_tickers_in_file)} unique tickers")
        
        # Identify date columns
        meta_cols = ['ticker', 'overall_rank', 'total']
        date_cols = [c for c in df.columns if c not in meta_cols and '/' in str(c)]
        
        if not date_cols:
            print(f"❌ No date columns found in {archive_path}")
            return pd.DataFrame()
        
        print(f"   Found {len(date_cols)} date columns")
        
        # Transform to long format preserving ALL data
        id_vars = ['ticker']
        if 'overall_rank' in df.columns:
            id_vars.append('overall_rank')
        if 'total' in df.columns:
            id_vars.append('total')
        
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=date_cols,
            var_name='date_str',
            value_name='mentions'
        )
        
        # Parse dates
        long_df['date'] = pd.to_datetime(long_df['date_str'], format='%m/%d/%y', errors='coerce')
        long_df = long_df.dropna(subset=['date'])
        
        # Clean mentions (convert to int, NaN becomes 0)
        long_df['mentions'] = pd.to_numeric(long_df['mentions'], errors='coerce').fillna(0).astype(int)
        
        # Add metadata
        long_df['subreddit'] = subreddit
        long_df['year'] = int(year)
        
        # Add ticker classification
        long_df['ticker_type'] = long_df['ticker'].apply(
            lambda x: 'meme_stock' if x in MEME_TICKERS else 
                     'crypto' if x in CRYPTO_TICKERS else 
                     'other'
        )
        
        # Only keep our target tickers (but could remove this filter to keep ALL)
        target_df = long_df[long_df['ticker'].isin(ALL_TICKERS)].copy()
        
        print(f"✅ Transformed to {len(target_df)} records ({len(long_df)} total)")
        
        return target_df
        
    except Exception as e:
        print(f"❌ Error processing {archive_path}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def create_analysis_views(full_data: pd.DataFrame) -> dict:
    """Create various analysis views from the full dataset."""
    views = {}
    
    # 1. Subreddit x Ticker x Date view (complete granularity)
    views['full_detail'] = full_data.copy()
    
    # 2. Daily ticker totals across all subreddits
    daily_ticker = full_data.groupby(['date', 'ticker', 'ticker_type']).agg({
        'mentions': 'sum',
        'subreddit': 'count',  # Number of subreddits mentioning
        'overall_rank': 'mean'  # Average rank across subreddits
    }).reset_index()
    daily_ticker.rename(columns={'subreddit': 'subreddit_count'}, inplace=True)
    views['daily_ticker_total'] = daily_ticker
    
    # 3. Subreddit x Ticker summary (total over time)
    subreddit_ticker = full_data.groupby(['subreddit', 'ticker', 'ticker_type']).agg({
        'mentions': ['sum', 'mean', 'max'],
        'date': ['min', 'max', 'count']
    }).reset_index()
    subreddit_ticker.columns = ['_'.join(col).strip('_') for col in subreddit_ticker.columns]
    views['subreddit_ticker_summary'] = subreddit_ticker
    
    # 4. Time series by subreddit (which communities discuss which tickers)
    subreddit_daily = full_data.groupby(['date', 'subreddit']).agg({
        'mentions': 'sum',
        'ticker': 'nunique'
    }).reset_index()
    views['subreddit_daily_activity'] = subreddit_daily
    
    # 5. Ticker popularity ranking by period
    # Monthly ranking
    full_data['month'] = full_data['date'].dt.to_period('M')
    monthly_rank = full_data.groupby(['month', 'ticker']).agg({
        'mentions': 'sum'
    }).reset_index()
    monthly_rank['monthly_rank'] = monthly_rank.groupby('month')['mentions'].rank(
        method='dense', ascending=False
    )
    views['monthly_ticker_rank'] = monthly_rank
    
    return views


def main():
    parser = argparse.ArgumentParser(
        description='Process Reddit archive data preserving full granularity'
    )
    parser.add_argument('--years', nargs='+', default=['2021', '2022', '2023'],
                        help='Years to process')
    parser.add_argument('--subreddits', nargs='+', default=SUBREDDITS,
                        help='Subreddits to process')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Process all data
    print(f"🚀 Processing Reddit archive data (Full Preservation Mode)")
    print(f"   Years: {args.years}")
    print(f"   Subreddits: {args.subreddits}")
    print()
    
    all_data = []
    
    for year in args.years:
        for subreddit in args.subreddits:
            df = load_and_transform_archive(subreddit, year)
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        raise RuntimeError("❌ No data loaded!")
    
    # Combine all data
    full_data = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(full_data)} total records")
    
    # Sort by date, subreddit, ticker for clean output
    full_data = full_data.sort_values(['date', 'subreddit', 'ticker']).reset_index(drop=True)
    
    # Create analysis views
    print("\n🔍 Creating analysis views...")
    views = create_analysis_views(full_data)
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / 'processed' / 'reddit'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    year_range = f"{min(args.years)}_{max(args.years)}"
    
    # Save main dataset (full granularity)
    main_path = output_dir / f'reddit_archive_full_{year_range}_{timestamp}.csv'
    full_data.to_csv(main_path, index=False)
    print(f"\n✅ Saved main dataset: {main_path}")
    print(f"   Records: {len(full_data):,}")
    print(f"   Columns: {list(full_data.columns)}")
    
    # Save analysis views
    for view_name, view_df in views.items():
        if view_name == 'full_detail':
            continue  # Already saved as main dataset
        
        view_path = output_dir / f'reddit_archive_{view_name}_{year_range}_{timestamp}.csv'
        
        # Convert period to string for monthly view
        if 'month' in view_df.columns:
            view_df['month'] = view_df['month'].astype(str)
        
        view_df.to_csv(view_path, index=False)
        print(f"   📄 {view_name}: {view_path.name} ({len(view_df)} rows)")
    
    # Generate comprehensive statistics
    stats = {
        'data_overview': {
            'total_records': len(full_data),
            'date_range': [
                full_data['date'].min().strftime('%Y-%m-%d'),
                full_data['date'].max().strftime('%Y-%m-%d')
            ],
            'unique_dates': full_data['date'].nunique(),
            'unique_tickers': full_data['ticker'].nunique(),
            'subreddits': list(full_data['subreddit'].unique()),
            'total_mentions': int(full_data['mentions'].sum())
        },
        'ticker_totals': {
            ticker: {
                'total_mentions': int(full_data[full_data['ticker'] == ticker]['mentions'].sum()),
                'days_active': int(full_data[full_data['ticker'] == ticker]['date'].nunique()),
                'subreddits_active': list(full_data[full_data['ticker'] == ticker]['subreddit'].unique()),
                'peak_day': {
                    'date': full_data[full_data['ticker'] == ticker].groupby('date')['mentions'].sum().idxmax().strftime('%Y-%m-%d'),
                    'mentions': int(full_data[full_data['ticker'] == ticker].groupby('date')['mentions'].sum().max())
                }
            } for ticker in sorted(full_data['ticker'].unique())
        },
        'subreddit_activity': {
            sub: {
                'total_mentions': int(full_data[full_data['subreddit'] == sub]['mentions'].sum()),
                'unique_tickers': int(full_data[full_data['subreddit'] == sub]['ticker'].nunique()),
                'top_ticker': full_data[full_data['subreddit'] == sub].groupby('ticker')['mentions'].sum().idxmax()
            } for sub in full_data['subreddit'].unique()
        }
    }
    
    # Save statistics
    stats_path = output_dir / f'reddit_archive_stats_{year_range}_{timestamp}.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Saved statistics: {stats_path}")
    
    # Print summary
    print(f"\n🎯 Processing complete!")
    print(f"📈 Key insights:")
    print(f"   - Total mentions: {stats['data_overview']['total_mentions']:,}")
    print(f"   - Most mentioned ticker: {max(stats['ticker_totals'].items(), key=lambda x: x[1]['total_mentions'])[0]}")
    print(f"   - Most active subreddit: {max(stats['subreddit_activity'].items(), key=lambda x: x[1]['total_mentions'])[0]}")
    
    # Sample of the data
    print(f"\n📋 Sample data (first 5 rows):")
    print(full_data.head().to_string(index=False))


if __name__ == '__main__':
    main()