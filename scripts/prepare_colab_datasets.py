#!/usr/bin/env python3
"""
Prepare comprehensive datasets for deep learning experiments on Colab.

Creates:
1. Time series sequences for LSTM/Transformer models
2. Feature-engineered tabular data
3. Multiple target horizons (1d, 5d, direction)
4. Train/val/test splits with proper time alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COLAB_DIR = DATA_DIR / "colab_datasets"
PROCESSED_DIR = DATA_DIR / "processed" / "targets"

# Create directories
COLAB_DIR.mkdir(parents=True, exist_ok=True)


def load_price_targets() -> pd.DataFrame:
    """Load the latest price targets dataset."""
    print("📊 Loading price targets for Colab preparation...")
    
    target_files = list(PROCESSED_DIR.glob('price_targets_aligned_*.csv'))
    if not target_files:
        raise FileNotFoundError("No price targets found. Run make_targets_price.py first.")
    
    latest_file = max(target_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean data thoroughly
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['y1d', 'y5d'])
    
    print(f"   Loaded {len(df)} clean records")
    print(f"   Tickers: {sorted(df['ticker'].unique())}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def engineer_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer comprehensive features for deep learning."""
    print("🔧 Engineering comprehensive features for DL...")
    
    df = df.copy().sort_values(['ticker', 'date'])
    
    # Price features (technical indicators)
    price_features = [
        'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
        'vol_5d', 'vol_10d', 'vol_20d',
        'price_ratio_sma10', 'price_ratio_sma20', 'rsi_14',
        'volume_ratio', 'turnover'
    ]
    
    # Basic Reddit features
    reddit_base_features = [
        'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
        'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95'
    ]
    
    # Enhanced Reddit features
    print("   Creating enhanced Reddit features...")
    
    # Momentum features (various windows)
    for window in [3, 7, 14, 21]:
        df[f'reddit_momentum_{window}'] = df.groupby('ticker')['log_mentions'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=max(1, window//2)).mean() - 
                     x.shift(window+1).rolling(window, min_periods=max(1, window//2)).mean()
        )
    
    # Volatility of mentions
    for window in [5, 10, 20]:
        df[f'reddit_vol_{window}'] = df.groupby('ticker')['log_mentions'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=max(1, window//2)).std()
        )
    
    # Relative attention (percentile within date)
    df['reddit_percentile'] = df.groupby('date')['log_mentions'].transform(
        lambda x: x.shift(1).rank(pct=True)
    )
    
    # Attention regimes
    df['reddit_high_regime'] = df.groupby('ticker')['log_mentions'].transform(
        lambda x: (x.shift(1) > x.shift(1).rolling(30, min_periods=15).quantile(0.8)).astype(int)
    )
    
    df['reddit_low_regime'] = df.groupby('ticker')['log_mentions'].transform(
        lambda x: (x.shift(1) < x.shift(1).rolling(30, min_periods=15).quantile(0.2)).astype(int)
    )
    
    # Cross-asset correlation proxy
    total_market_mentions = df.groupby('date')['log_mentions'].sum()
    df['market_mentions_total'] = df['date'].map(total_market_mentions)
    df['market_sentiment'] = df['market_mentions_total'].shift(1)
    
    # Interaction features
    df['price_reddit_momentum'] = df['returns_1d'] * df['reddit_momentum_7']
    df['vol_reddit_attention'] = df['vol_5d'] * df['log_mentions'].shift(1)
    
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_weekend_effect'] = ((df['day_of_week'] == 0) | (df['day_of_week'] == 4)).astype(int)
    
    # Market state features
    df['market_vol_regime'] = df.groupby('ticker')['vol_10d'].transform(
        lambda x: (x > x.rolling(60, min_periods=30).quantile(0.7)).astype(int)
    )
    
    print(f"   Total features: {len(df.columns)}")
    
    return df


def create_time_series_sequences(df: pd.DataFrame, sequence_length: int = 20) -> Dict:
    """Create time series sequences for LSTM/Transformer models."""
    print(f"📈 Creating time series sequences (length={sequence_length})...")
    
    sequences_data = {}
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy().sort_values('date').reset_index(drop=True)
        
        if len(ticker_data) < sequence_length + 5:  # Need minimum data
            continue
        
        # Feature columns for sequences (exclude metadata and targets)
        feature_cols = [col for col in ticker_data.columns 
                       if col not in ['date', 'ticker', 'ticker_type', 'y1d', 'y5d', 'alpha_1d', 'alpha_5d',
                                    'direction_1d', 'direction_5d']]
        
        # Create sequences
        sequences = []
        targets_1d = []
        targets_5d = []
        directions_1d = []
        dates = []
        
        for i in range(sequence_length, len(ticker_data) - 1):  # Leave room for targets
            # Sequence of features (looking back)
            seq = ticker_data[feature_cols].iloc[i-sequence_length:i].values
            
            # Targets (looking forward)
            target_1d = ticker_data['y1d'].iloc[i]
            target_5d = ticker_data['y5d'].iloc[i]
            direction_1d = np.sign(target_1d)
            
            if not (np.isnan(target_1d) or np.isnan(target_5d)):
                sequences.append(seq)
                targets_1d.append(target_1d)
                targets_5d.append(target_5d)
                directions_1d.append(direction_1d)
                dates.append(ticker_data['date'].iloc[i])
        
        if len(sequences) > 0:
            sequences_data[ticker] = {
                'sequences': np.array(sequences),
                'targets_1d': np.array(targets_1d),
                'targets_5d': np.array(targets_5d),
                'directions_1d': np.array(directions_1d),
                'dates': dates,
                'feature_names': feature_cols
            }
    
    print(f"   Created sequences for {len(sequences_data)} tickers")
    for ticker, data in sequences_data.items():
        print(f"     {ticker}: {len(data['sequences'])} sequences")
    
    return sequences_data


def create_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create final tabular dataset with all engineered features."""
    print("📊 Creating tabular features dataset...")
    
    # Select final feature set for tabular models
    price_features = [
        'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
        'vol_5d', 'vol_10d', 'vol_20d',
        'price_ratio_sma10', 'price_ratio_sma20', 'rsi_14',
        'volume_ratio', 'turnover'
    ]
    
    reddit_features = [
        'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
        'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
        'reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14', 'reddit_momentum_21',
        'reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20',
        'reddit_percentile', 'reddit_high_regime', 'reddit_low_regime',
        'market_sentiment', 'price_reddit_momentum', 'vol_reddit_attention'
    ]
    
    calendar_features = [
        'day_of_week', 'month', 'is_monday', 'is_friday', 'is_weekend_effect'
    ]
    
    market_features = ['market_vol_regime']
    
    # Metadata and targets
    meta_features = ['date', 'ticker', 'ticker_type']
    target_features = ['y1d', 'y5d', 'alpha_1d', 'alpha_5d', 'direction_1d', 'direction_5d']
    
    # Combine all features
    all_features = (meta_features + price_features + reddit_features + 
                   calendar_features + market_features + target_features)
    
    # Filter to available columns
    available_features = [col for col in all_features if col in df.columns]
    tabular_df = df[available_features].copy()
    
    # Remove rows with NaN targets
    tabular_df = tabular_df.dropna(subset=['y1d', 'y5d'])
    
    print(f"   Tabular dataset: {len(tabular_df)} samples, {len(available_features)} features")
    
    return tabular_df


def create_train_val_test_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create train/validation/test splits with proper time alignment."""
    print("📅 Creating train/val/test splits...")
    
    # Sort by date
    df_sorted = df.sort_values('date')
    
    # Time-based splits to prevent data leakage
    dates = df_sorted['date'].unique()
    dates = np.sort(dates)
    
    n_dates = len(dates)
    
    # 70% train, 15% val, 15% test
    train_end_idx = int(0.70 * n_dates)
    val_end_idx = int(0.85 * n_dates)
    
    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:]
    
    # Create splits
    train_df = df_sorted[df_sorted['date'].isin(train_dates)].copy()
    val_df = df_sorted[df_sorted['date'].isin(val_dates)].copy()
    test_df = df_sorted[df_sorted['date'].isin(test_dates)].copy()
    
    splits = {
        'train': train_df,
        'val': val_df, 
        'test': test_df
    }
    
    print(f"   Train: {len(train_df)} samples ({train_dates[0]} to {train_dates[-1]})")
    print(f"   Val: {len(val_df)} samples ({val_dates[0]} to {val_dates[-1]})")
    print(f"   Test: {len(test_df)} samples ({test_dates[0]} to {test_dates[-1]})")
    
    return splits


def save_datasets_for_colab(tabular_splits: Dict, sequences_data: Dict):
    """Save all datasets in formats suitable for Colab."""
    print("💾 Saving datasets for Colab...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save tabular data
    for split_name, split_df in tabular_splits.items():
        filename = COLAB_DIR / f'tabular_{split_name}_{timestamp}.csv'
        split_df.to_csv(filename, index=False)
        print(f"   Saved tabular {split_name}: {filename}")
    
    # Save sequences data
    sequences_filename = COLAB_DIR / f'sequences_{timestamp}.npz'
    
    # Prepare sequences data for saving
    sequences_dict = {}
    for ticker, data in sequences_data.items():
        for key, value in data.items():
            if key != 'dates':  # Save dates separately as strings
                sequences_dict[f'{ticker}_{key}'] = value
            else:
                sequences_dict[f'{ticker}_dates'] = [d.strftime('%Y-%m-%d') for d in value]
    
    np.savez_compressed(sequences_filename, **sequences_dict)
    print(f"   Saved sequences: {sequences_filename}")
    
    # Save feature metadata
    feature_info = {
        'timestamp': timestamp,
        'tabular_features': list(tabular_splits['train'].columns),
        'sequence_features': sequences_data[list(sequences_data.keys())[0]]['feature_names'],
        'tickers': list(sequences_data.keys()),
        'dataset_info': {
            'train_samples': len(tabular_splits['train']),
            'val_samples': len(tabular_splits['val']),
            'test_samples': len(tabular_splits['test']),
            'sequence_length': sequences_data[list(sequences_data.keys())[0]]['sequences'].shape[1],
            'n_sequence_features': sequences_data[list(sequences_data.keys())[0]]['sequences'].shape[2]
        }
    }
    
    metadata_filename = COLAB_DIR / f'dataset_metadata_{timestamp}.json'
    with open(metadata_filename, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"   Saved metadata: {metadata_filename}")
    
    return timestamp


def main():
    """Main execution for dataset preparation."""
    
    print("🚀 Preparing Comprehensive Datasets for Colab Deep Learning")
    print("=" * 60)
    
    # Load data
    df = load_price_targets()
    
    # Engineer comprehensive features
    df_enhanced = engineer_comprehensive_features(df)
    
    # Create time series sequences
    sequences_data = create_time_series_sequences(df_enhanced, sequence_length=20)
    
    # Create tabular features dataset
    tabular_df = create_tabular_features(df_enhanced)
    
    # Create train/val/test splits
    tabular_splits = create_train_val_test_splits(tabular_df)
    
    # Save datasets for Colab
    timestamp = save_datasets_for_colab(tabular_splits, sequences_data)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 All datasets saved in: {COLAB_DIR}")
    print(f"🏷️  Timestamp: {timestamp}")
    
    # Summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Tabular data: {len(tabular_df)} samples × {len(tabular_df.columns)} features")
    print(f"   Time series: {sum(len(v['sequences']) for v in sequences_data.values())} sequences")
    print(f"   Sequence length: 20 timesteps")
    print(f"   Tickers: {len(sequences_data)}")
    
    print(f"\n🎯 Targets available:")
    print(f"   y1d: Next-day returns")
    print(f"   y5d: 5-day cumulative returns")  
    print(f"   direction_1d: Binary direction classification")
    print(f"   alpha_1d/5d: Market-adjusted returns")
    
    print(f"\n🧠 Ready for deep learning experiments on Colab!")
    print(f"   Use tabular data for: MLP, TabNet, CatBoost")
    print(f"   Use sequence data for: LSTM, GRU, Transformer")
    
    return timestamp


if __name__ == '__main__':
    main()