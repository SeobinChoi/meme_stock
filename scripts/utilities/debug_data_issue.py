#!/usr/bin/env python3
"""
Debug Data Issues - 타겟 변수 문제 진단
===============================================
"""

import pandas as pd
import numpy as np

def debug_target_data():
    """타겟 데이터 문제 진단"""
    print("🔍 Debugging Target Data Issues...")
    
    # Load data
    df = pd.read_csv("data/features/advanced_meme_features_dataset.csv")
    print(f"📊 Dataset shape: {df.shape}")
    
    # Check target columns
    target_cols = [col for col in df.columns if 'returns_1d' in col]
    print(f"\n🎯 Found target columns: {target_cols}")
    
    for col in target_cols:
        data = df[col].dropna()
        print(f"\n📈 {col}:")
        print(f"   Non-NaN values: {len(data)}")
        print(f"   Min: {data.min():.6f}")
        print(f"   Max: {data.max():.6f}")
        print(f"   Mean: {data.mean():.6f}")
        print(f"   Std: {data.std():.6f}")
        print(f"   Unique values: {len(data.unique())}")
        print(f"   Zero values: {(data == 0).sum()}")
        print(f"   Distribution:")
        print(f"   {data.describe()}")
    
    # Check if all values are the same
    for col in target_cols:
        data = df[col].dropna()
        if len(data.unique()) <= 1:
            print(f"⚠️ {col} has constant values!")
        elif data.std() < 1e-10:
            print(f"⚠️ {col} has very low variance!")
    
    # Check dates
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"📅 Total days: {len(df)}")
    
    # Look for actual price data in columns
    price_cols = [col for col in df.columns if any(stock in col for stock in ['GME', 'AMC', 'BB']) and 'price' in col.lower()]
    print(f"\n💰 Price-related columns: {len(price_cols)}")
    for col in price_cols[:5]:  # Show first 5
        print(f"   {col}: {df[col].dropna().std():.6f} std")

def check_alternative_data():
    """다른 데이터셋 확인"""
    print("\n🔄 Checking alternative datasets...")
    
    # Check colab datasets
    try:
        train_df = pd.read_csv("data/colab_datasets/tabular_train_20250814_031335.csv")
        print(f"📊 Colab train dataset: {train_df.shape}")
        
        # Check for y1d column
        if 'y1d' in train_df.columns:
            y_data = train_df['y1d'].dropna()
            print(f"🎯 y1d target:")
            print(f"   Non-NaN values: {len(y_data)}")
            print(f"   Min: {y_data.min():.6f}")
            print(f"   Max: {y_data.max():.6f}")
            print(f"   Mean: {y_data.mean():.6f}")
            print(f"   Std: {y_data.std():.6f}")
            print(f"   Unique values: {len(y_data.unique())}")
            
            # Check if this is better
            if y_data.std() > 1e-6:
                print("✅ Colab dataset has proper target variance!")
                return train_df
                
    except Exception as e:
        print(f"❌ Error loading colab dataset: {e}")
    
    return None

def main():
    debug_target_data()
    colab_df = check_alternative_data()
    
    if colab_df is not None:
        print("\n💡 Recommendation: Use colab dataset instead!")
    else:
        print("\n❌ Need to fix data preparation pipeline!")

if __name__ == "__main__":
    main()

