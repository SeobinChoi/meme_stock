#!/usr/bin/env python3
"""
Execute the 2021 Meme Stock Analysis and Generate Training Dataset
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os
from pathlib import Path

print("🚀 **Starting 2021 Meme Stock Analysis Execution**")
print("="*55)

# Load unified dataset
data_dir = Path('data')
unified_path = data_dir / 'processed' / 'unified_dataset.csv'

if unified_path.exists():
    print(f"✅ Loading unified dataset: {unified_path}")
    unified_df = pd.read_csv(unified_path)
    print(f"   Shape: {unified_df.shape}")
    
    # Convert date column to datetime
    unified_df['date'] = pd.to_datetime(unified_df['date'], format='mixed')
    
    # Filter to 2021 only
    data_2021 = unified_df[unified_df['date'].dt.year == 2021].copy()
    
    print(f"🎯 **2021 Dataset:**")
    print(f"   📊 Records: {len(data_2021):,}")
    print(f"   📅 Date range: {data_2021['date'].min()} to {data_2021['date'].max()}")
    print(f"   📈 Features: {len(data_2021.columns)} columns")
    
    # Basic data cleaning
    print(f"\n🧹 **Data Cleaning:**")
    original_shape = data_2021.shape
    
    # Fill missing values
    reddit_cols = [col for col in data_2021.columns if 'reddit' in col.lower()]
    if reddit_cols:
        data_2021[reddit_cols] = data_2021[reddit_cols].fillna(0)
        print(f"   💬 Filled {len(reddit_cols)} Reddit columns with 0")
    
    # Forward fill stock data
    stock_cols = [col for col in data_2021.columns if any(x in col for x in ['_close', '_open', '_high', '_low', '_volume'])]
    if stock_cols:
        data_2021[stock_cols] = data_2021[stock_cols].fillna(method='ffill')
        print(f"   📈 Forward-filled {len(stock_cols)} stock columns")
    
    # Fill remaining with median
    numeric_cols = data_2021.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data_2021[col].isnull().sum() > 0:
            data_2021[col] = data_2021[col].fillna(data_2021[col].median())
    
    print(f"   ✅ Cleaned dataset: {original_shape} → {data_2021.shape}")
    
    # Set date as index
    data_2021_indexed = data_2021.set_index('date').copy()
    
    # Feature Engineering
    print(f"\n⚙️ **Feature Engineering:**")
    features_df = data_2021_indexed.copy()
    original_features = len(features_df.columns)
    
    # Get main stock tickers
    stock_tickers = ['GME', 'AMC', 'BB']
    
    # 1. Volatility and Returns
    for ticker in stock_tickers:
        close_cols = [col for col in features_df.columns if ticker in col and 'close' in col]
        
        if close_cols:
            close_col = close_cols[0]
            
            # Daily returns
            features_df[f'{ticker}_daily_return'] = features_df[close_col].pct_change()
            
            # Volatility
            features_df[f'{ticker}_volatility_7d'] = features_df[f'{ticker}_daily_return'].rolling(7).std()
            features_df[f'{ticker}_volatility_14d'] = features_df[f'{ticker}_daily_return'].rolling(14).std()
            
            # Moving averages
            features_df[f'{ticker}_ma_7'] = features_df[close_col].rolling(7).mean()
            features_df[f'{ticker}_ma_14'] = features_df[close_col].rolling(14).mean()
            
            # Price vs MA
            features_df[f'{ticker}_price_vs_ma7'] = (features_df[close_col] / features_df[f'{ticker}_ma_7'] - 1) * 100
            
            # Momentum
            features_df[f'{ticker}_momentum_3d'] = features_df[close_col].pct_change(3)
            features_df[f'{ticker}_momentum_7d'] = features_df[close_col].pct_change(7)
    
    # 2. Reddit features
    reddit_activity_cols = [col for col in features_df.columns if 'reddit' in col.lower() and 'count' in col.lower()]
    if reddit_activity_cols:
        main_count_col = reddit_activity_cols[0]
        
        # Reddit momentum
        features_df['reddit_activity_momentum_3d'] = features_df[main_count_col].pct_change(3)
        features_df['reddit_activity_momentum_7d'] = features_df[main_count_col].pct_change(7)
        features_df['reddit_activity_volatility'] = features_df[main_count_col].rolling(7).std()
        features_df['reddit_activity_ma7'] = features_df[main_count_col].rolling(7).mean()
    
    # 3. Temporal features
    features_df['is_monday'] = (features_df.index.dayofweek == 0).astype(int)
    features_df['is_friday'] = (features_df.index.dayofweek == 4).astype(int)
    features_df['is_weekend'] = (features_df.index.dayofweek >= 5).astype(int)
    features_df['month'] = features_df.index.month
    features_df['is_january'] = (features_df.index.month == 1).astype(int)
    features_df['day_of_month'] = features_df.index.day
    
    # 4. Target variables
    targets_created = 0
    for ticker in stock_tickers:
        close_cols = [col for col in features_df.columns if ticker in col and 'close' in col]
        
        if close_cols:
            close_col = close_cols[0]
            
            # Future returns
            features_df[f'{ticker}_target_1d_return'] = features_df[close_col].pct_change().shift(-1)
            features_df[f'{ticker}_target_3d_return'] = features_df[close_col].pct_change(3).shift(-3)
            features_df[f'{ticker}_target_7d_return'] = features_df[close_col].pct_change(7).shift(-7)
            
            # Direction
            features_df[f'{ticker}_target_1d_direction'] = (features_df[f'{ticker}_target_1d_return'] > 0).astype(int)
            features_df[f'{ticker}_target_3d_direction'] = (features_df[f'{ticker}_target_3d_return'] > 0).astype(int)
            
            targets_created += 5
    
    new_features = len(features_df.columns) - original_features
    print(f"   ✅ Created {new_features} new features")
    print(f"   🎯 Created {targets_created} target variables")
    
    # Final cleaning
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # Remove highly correlated features (>0.95)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if 'target' not in col]
    
    if len(feature_cols) > 10:
        feature_matrix = features_df[feature_cols].dropna(axis=1, how='all')
        
        if len(feature_matrix.columns) > 5:
            corr_matrix = feature_matrix.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
            
            if high_corr_features:
                features_df = features_df.drop(columns=high_corr_features)
                print(f"   🗑️ Removed {len(high_corr_features)} highly correlated features")
    
    # Final dataset preparation
    final_df = features_df.copy()
    
    # Fill remaining NaNs in features (not targets)
    feature_only_cols = [col for col in final_df.columns if 'target' not in col]
    for col in feature_only_cols:
        if final_df[col].dtype in ['int64', 'float64'] and final_df[col].isnull().sum() > 0:
            final_df[col] = final_df[col].fillna(final_df[col].median())
    
    # Export dataset
    export_df = final_df.reset_index()
    export_path = 'training_data_2021.csv'
    export_df.to_csv(export_path, index=False)
    
    # Final summary
    feature_count = len([col for col in export_df.columns if 'target' not in col])
    target_count = len([col for col in export_df.columns if 'target' in col])
    
    print(f"\n💾 **Export Complete:**")
    print(f"   📁 File: {export_path}")
    print(f"   📊 Records: {len(export_df):,}")
    print(f"   📈 Features: {feature_count}")
    print(f"   🎯 Targets: {target_count}")
    print(f"   💾 Size: {os.path.getsize(export_path) / 1024**2:.1f} MB")
    print(f"   🧹 Missing values: {export_df.isnull().sum().sum():,}")
    
    print(f"\n🎉 **2021 Meme Stock Analysis Complete!**")
    print(f"   ⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🎯 Focus: Strategic 2021 dataset ready for ML training")
    print(f"   ✅ Status: Production-ready dataset generated")

else:
    print("❌ Unified dataset not found")
    print("Please ensure the unified_dataset.csv file exists in data/processed/")