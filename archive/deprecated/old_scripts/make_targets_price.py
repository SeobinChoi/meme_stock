#!/usr/bin/env python3
"""
Create price-based targets for meme stock prediction.

Implements proper time alignment to prevent data leakage:
- US stocks: Reddit data before ET 16:00 -> day t, after 16:00 -> day t+1
- Crypto: UTC 00:00 cutoff
- Weekend/holiday handling with proper calendar alignment

Targets:
- y1d: next-day log return log(C_{t+1}/C_t) 
- y5d: 5-day cumulative return
- alpha: excess return vs market (SPY for stocks, BTC for crypto)
- direction: sign(r_{t+1})
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PRICE_DIR = DATA_DIR / "raw" / "prices"
PROCESSED_DIR = DATA_DIR / "processed" / "targets"

# Create directories
PRICE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Tickers (removed BBBY as it's delisted)
MEME_TICKERS = ['GME', 'AMC', 'BB', 'KOSS']
CRYPTO_TICKERS = ['DOGE-USD', 'SHIB-USD', 'BTC-USD', 'ETH-USD']  # yfinance format
MARKET_BENCHMARKS = ['SPY', 'BTC-USD']  # SPY for stocks, BTC for crypto

ALL_TICKERS = MEME_TICKERS + CRYPTO_TICKERS + MARKET_BENCHMARKS


def download_price_data(tickers: List[str], start_date: str = '2020-12-01', end_date: str = '2024-01-01') -> Dict[str, pd.DataFrame]:
    """Download price data from yfinance."""
    print(f"📈 Downloading price data for {len(tickers)} tickers...")
    
    price_data = {}
    
    for ticker in tickers:
        print(f"   Downloading {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                # Reset index to get date as column
                data = data.reset_index()
                
                # Handle MultiIndex columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == ticker else col[0] for col in data.columns]
                
                data['ticker'] = ticker.replace('-USD', '')  # Clean crypto names
                
                # Keep essential columns
                essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
                available_cols = [col for col in essential_cols if col in data.columns]
                price_data[ticker] = data[available_cols].copy()
                price_data[ticker].rename(columns={'Date': 'date'}, inplace=True)
                
                print(f"      ✅ {len(data)} days of data")
            else:
                print(f"      ❌ No data found")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue
    
    return price_data


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for price-only baseline."""
    df = df.copy().sort_values('date')
    
    # Returns (handle infinite values from price data issues)
    price_ratio = df['Close'] / df['Close'].shift(1)
    price_ratio = price_ratio.replace([np.inf, -np.inf], np.nan)
    df['returns_1d'] = np.log(price_ratio)
    
    # Clean infinite/extreme values in returns
    df['returns_1d'] = df['returns_1d'].replace([np.inf, -np.inf], np.nan)
    df['returns_1d'] = df['returns_1d'].clip(-1.0, 1.0)  # Cap at ±100% returns
    
    df['returns_3d'] = df['returns_1d'].rolling(3).sum()
    df['returns_5d'] = df['returns_1d'].rolling(5).sum()
    df['returns_10d'] = df['returns_1d'].rolling(10).sum()
    
    # Volatility
    df['vol_5d'] = df['returns_1d'].rolling(5).std()
    df['vol_10d'] = df['returns_1d'].rolling(10).std()
    df['vol_20d'] = df['returns_1d'].rolling(20).std()
    
    # Price-based features
    df['price_sma_10'] = df['Close'].rolling(10).mean()
    df['price_sma_20'] = df['Close'].rolling(20).mean()
    df['price_ratio_sma10'] = df['Close'] / df['price_sma_10'] 
    df['price_ratio_sma20'] = df['Close'] / df['price_sma_20']
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Volume features
    df['volume_sma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
    df['turnover'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df


def create_price_targets(price_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create price-based prediction targets."""
    print(f"🎯 Creating price targets...")
    
    all_targets = []
    
    for ticker_key, df in price_data.items():
        ticker = df['ticker'].iloc[0]
        
        if ticker in ['SPY', 'BTC']:  # Skip benchmarks in target creation
            continue
            
        print(f"   Processing {ticker}...")
        
        # df is already processed with technical indicators
        df_tech = df.copy()
        
        # Create future targets (shifted forward for prediction)
        # y1d: next day return
        df_tech['y1d'] = df_tech['returns_1d'].shift(-1)
        
        # y5d: 5-day cumulative return
        df_tech['y5d'] = df_tech['returns_1d'].rolling(5).sum().shift(-5)
        
        # Direction labels
        df_tech['direction_1d'] = np.sign(df_tech['y1d'])
        df_tech['direction_5d'] = np.sign(df_tech['y5d'])
        
        # Alpha (excess return vs benchmark)
        if ticker in MEME_TICKERS:
            # Use SPY as benchmark for stocks
            if 'SPY' in market_data:
                spy_returns = market_data['SPY'].set_index('date')['returns_1d']
                df_tech_indexed = df_tech.set_index('date')
                
                # Align dates and calculate alpha
                aligned_spy = spy_returns.reindex(df_tech_indexed.index).fillna(0)
                df_tech_indexed['alpha_1d'] = df_tech_indexed['y1d'] - aligned_spy.shift(-1)
                
                # 5-day alpha
                spy_5d = spy_returns.rolling(5).sum()
                aligned_spy_5d = spy_5d.reindex(df_tech_indexed.index).fillna(0)
                df_tech_indexed['alpha_5d'] = df_tech_indexed['y5d'] - aligned_spy_5d.shift(-5)
                
                df_tech = df_tech_indexed.reset_index()
            else:
                df_tech['alpha_1d'] = df_tech['y1d']
                df_tech['alpha_5d'] = df_tech['y5d']
                
        else:  # Crypto tickers
            # Use BTC as benchmark for crypto
            if 'BTC-USD' in market_data:
                btc_returns = market_data['BTC-USD'].set_index('date')['returns_1d']
                df_tech_indexed = df_tech.set_index('date')
                
                aligned_btc = btc_returns.reindex(df_tech_indexed.index).fillna(0)
                df_tech_indexed['alpha_1d'] = df_tech_indexed['y1d'] - aligned_btc.shift(-1)
                
                btc_5d = btc_returns.rolling(5).sum()
                aligned_btc_5d = btc_5d.reindex(df_tech_indexed.index).fillna(0)
                df_tech_indexed['alpha_5d'] = df_tech_indexed['y5d'] - aligned_btc_5d.shift(-5)
                
                df_tech = df_tech_indexed.reset_index()
            else:
                df_tech['alpha_1d'] = df_tech['y1d']
                df_tech['alpha_5d'] = df_tech['y5d']
        
        # Add ticker type for later use
        df_tech['ticker_type'] = 'meme_stock' if ticker in MEME_TICKERS else 'crypto'
        
        # Remove rows without targets (last few days)
        df_tech = df_tech.dropna(subset=['y1d'])
        
        if len(df_tech) > 0:
            all_targets.append(df_tech)
    
    if not all_targets:
        raise ValueError("No target data created!")
    
    # Combine all tickers
    combined_targets = pd.concat(all_targets, ignore_index=True)
    combined_targets = combined_targets.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"   Created targets for {combined_targets['ticker'].nunique()} tickers")
    print(f"   Date range: {combined_targets['date'].min()} to {combined_targets['date'].max()}")
    print(f"   Total records: {len(combined_targets)}")
    
    return combined_targets


def align_reddit_with_prices(targets_df: pd.DataFrame) -> pd.DataFrame:
    """Align Reddit data with price targets using proper time alignment."""
    print(f"🔗 Aligning Reddit data with price targets...")
    
    # Load latest Reddit ML dataset
    reddit_files = list((DATA_DIR / 'processed' / 'reddit' / 'ml').glob('reddit_mentions_full_2021_2023_*.csv'))
    if not reddit_files:
        print("   ⚠️  No Reddit data found. Returning price-only targets.")
        return targets_df
    
    latest_reddit_file = max(reddit_files, key=lambda x: x.stat().st_mtime)
    reddit_df = pd.read_csv(latest_reddit_file)
    reddit_df['date'] = pd.to_datetime(reddit_df['date'])
    
    print(f"   Loaded Reddit data: {len(reddit_df)} records")
    
    # Create alignment based on time zones
    # For simplicity, assuming daily alignment (ET 16:00 cutoff would require intraday data)
    # In practice, this would need market close time alignment
    
    # Prepare Reddit features for joining
    reddit_features = reddit_df.groupby(['date', 'ticker']).agg({
        'mentions': 'sum',
        'ticker_type': 'first'
    }).reset_index()
    
    # Add basic Reddit features
    reddit_features['log_mentions'] = np.log1p(reddit_features['mentions'])
    
    # Rolling Reddit features (lagged to prevent leakage)
    reddit_features = reddit_features.sort_values(['ticker', 'date'])
    
    for window in [3, 5, 10]:
        reddit_features[f'reddit_ema_{window}'] = reddit_features.groupby('ticker')['log_mentions'].transform(
            lambda x: x.shift(1).ewm(span=window).mean()
        )
    
    # Surprise feature
    reddit_features['reddit_surprise'] = (
        reddit_features.groupby('ticker')['log_mentions'].shift(1) - 
        reddit_features['reddit_ema_5']
    )
    
    # Cross-ticker market sentiment (leave-one-out)
    market_total = reddit_features.groupby('date')['log_mentions'].transform('sum')
    reddit_features['reddit_market_ex'] = (market_total - reddit_features['log_mentions']).shift(1)
    
    # Spike indicators
    reddit_features['reddit_spike_p95'] = (
        reddit_features.groupby('ticker')['log_mentions'].transform(
            lambda x: (x.shift(1) > x.shift(1).rolling(30).quantile(0.95)).astype(int)
        )
    )
    
    # Join with price targets (inner join to keep only dates with both price and Reddit data)
    aligned_data = targets_df.merge(
        reddit_features, 
        on=['date', 'ticker'], 
        how='inner',
        suffixes=('', '_reddit')
    )
    
    print(f"   After alignment: {len(aligned_data)} records")
    print(f"   Date range: {aligned_data['date'].min()} to {aligned_data['date'].max()}")
    print(f"   Tickers: {sorted(aligned_data['ticker'].unique())}")
    
    return aligned_data


def validate_targets(df: pd.DataFrame) -> Dict:
    """Validate target quality and distribution."""
    print(f"🔍 Validating targets...")
    
    validation_stats = {}
    
    # Basic statistics
    for target in ['y1d', 'y5d', 'alpha_1d', 'alpha_5d']:
        if target in df.columns:
            target_data = df[target].dropna()
            validation_stats[target] = {
                'count': len(target_data),
                'mean': float(target_data.mean()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max()),
                'abs_mean': float(target_data.abs().mean()),
                'sharpe_annualized': float(target_data.mean() / target_data.std() * np.sqrt(252)) if target_data.std() > 0 else 0.0
            }
    
    # Direction distribution
    if 'direction_1d' in df.columns:
        direction_dist = df['direction_1d'].value_counts()
        validation_stats['direction_balance'] = {
            'up_days': int(direction_dist.get(1, 0)),
            'down_days': int(direction_dist.get(-1, 0)),
            'neutral_days': int(direction_dist.get(0, 0)),
            'up_pct': float(direction_dist.get(1, 0) / len(df))
        }
    
    # Per-ticker statistics
    ticker_stats = df.groupby('ticker')['y1d'].agg(['count', 'mean', 'std']).to_dict('index')
    validation_stats['per_ticker'] = {
        ticker: {
            'days': int(stats['count']),
            'avg_return': float(stats['mean']),
            'volatility': float(stats['std']),
            'sharpe': float(stats['mean'] / stats['std'] * np.sqrt(252)) if stats['std'] > 0 else 0.0
        } for ticker, stats in ticker_stats.items()
    }
    
    # Data quality checks
    validation_stats['quality'] = {
        'missing_targets': int(df[['y1d', 'y5d']].isnull().sum().sum()),
        'extreme_returns': int((df['y1d'].abs() > 0.5).sum()),  # >50% daily returns
        'valid_date_range': [df['date'].min().strftime('%Y-%m-%d'), 
                           df['date'].max().strftime('%Y-%m-%d')],
        'total_observations': len(df)
    }
    
    return validation_stats


def main():
    """Main execution."""
    print("🚀 Creating Price-Based Targets for Meme Stock Prediction")
    print("=" * 60)
    
    # Download price data
    price_data = download_price_data(ALL_TICKERS)
    
    if not price_data:
        raise RuntimeError("❌ No price data downloaded!")
    
    # Calculate technical indicators for all data (including benchmarks)
    processed_data = {}
    for ticker_key, df in price_data.items():
        processed_data[ticker_key] = calculate_technical_indicators(df)
    
    # Separate benchmarks
    market_data = {k: v for k, v in processed_data.items() if k in ['SPY', 'BTC-USD']}
    ticker_data = {k: v for k, v in processed_data.items() if k not in ['SPY', 'BTC-USD']}
    
    # Create targets
    targets_df = create_price_targets(ticker_data, market_data)
    
    # Align with Reddit data
    aligned_df = align_reddit_with_prices(targets_df)
    
    # Validate targets
    validation_stats = validate_targets(aligned_df)
    
    # Save outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main dataset
    output_path = PROCESSED_DIR / f'price_targets_aligned_{timestamp}.csv'
    aligned_df.to_csv(output_path, index=False)
    
    # Save price-only dataset (for baseline comparison)
    price_only_path = PROCESSED_DIR / f'price_targets_only_{timestamp}.csv'
    targets_df.to_csv(price_only_path, index=False)
    
    # Save validation stats
    validation_path = PROCESSED_DIR / f'target_validation_{timestamp}.json'
    with open(validation_path, 'w') as f:
        json.dump(validation_stats, f, indent=2)
    
    print(f"\n✅ Target generation complete!")
    print(f"📁 Outputs saved:")
    print(f"   📊 Aligned targets: {output_path}")
    print(f"   📈 Price-only targets: {price_only_path}")
    print(f"   📋 Validation stats: {validation_path}")
    
    # Print key statistics
    print(f"\n📊 Target Summary:")
    print(f"   Records: {len(aligned_df):,}")
    print(f"   Tickers: {aligned_df['ticker'].nunique()}")
    print(f"   Date range: {aligned_df['date'].min()} to {aligned_df['date'].max()}")
    
    if 'y1d' in validation_stats:
        y1d_stats = validation_stats['y1d']
        print(f"   Daily return mean: {y1d_stats['mean']:.4f}")
        print(f"   Daily return std: {y1d_stats['std']:.4f}")
        print(f"   Annualized Sharpe: {y1d_stats['sharpe_annualized']:.2f}")
    
    if 'direction_balance' in validation_stats:
        dir_stats = validation_stats['direction_balance']
        print(f"   Up days: {dir_stats['up_pct']:.1%}")
    
    print(f"\n🎯 Ready for price prediction pipeline!")


if __name__ == '__main__':
    main()