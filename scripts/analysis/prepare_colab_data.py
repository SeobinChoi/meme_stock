#!/usr/bin/env python3
"""
Prepare data for Colab advanced model training
"""

import pandas as pd
import numpy as np
import sys
import os

def prepare_colab_dataset():
    """Prepare the advanced features dataset for Colab training"""
    
    print("🔧 Preparing advanced features dataset for Colab...")
    
    # Load the enhanced dataset; fallback to unified if not available  # [FIX]
    data = None
    try:
        data = pd.read_csv('data/processed/meme_enhanced_data.csv')
        print(f"✅ Loaded enhanced dataset with shape: {data.shape}")
    except FileNotFoundError:
        print("⚠️ Enhanced dataset not found. Falling back to unified_dataset.csv [FIX]")
        try:
            data = pd.read_csv('data/processed/unified_dataset.csv')
            print(f"✅ Loaded unified dataset with shape: {data.shape}")
        except FileNotFoundError:
            print("❌ Neither enhanced nor unified dataset found. Please run data pipeline first.")
            return None
    
    # Remove target variables from features (to prevent data leakage)
    target_patterns = ['direction', 'magnitude', 'returns']
    feature_cols = [col for col in data.columns if not any(pattern in col for pattern in target_patterns)]
    
    # Create clean dataset for Colab
    colab_data = data[feature_cols + ['GME_direction_1d', 'GME_direction_3d', 
                                     'AMC_direction_1d', 'AMC_direction_3d',
                                     'BB_direction_1d', 'BB_direction_3d']].copy()
    
    # Handle missing values
    colab_data = colab_data.fillna(0)
    
    # Save for Colab upload
    output_file = 'colab_advanced_features.csv'
    colab_data.to_csv(output_file, index=False)
    
    print(f"✅ Prepared Colab dataset with shape: {colab_data.shape}")
    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Targets: 6 (direction predictions)")
    print(f"✅ Saved to: {output_file}")
    
    # Print feature categories
    print("\n📊 Feature Categories:")
    feature_categories = {
        'Reddit Features': [col for col in feature_cols if any(x in col.lower() for x in ['reddit', 'sentiment', 'viral', 'social'])],
        'Financial Features': [col for col in feature_cols if any(x in col.lower() for x in ['price', 'volume', 'returns', 'volatility'])],
        'Technical Features': [col for col in feature_cols if any(x in col.lower() for x in ['rsi', 'macd', 'bollinger', 'ma'])],
        'Cross-Modal Features': [col for col in feature_cols if any(x in col.lower() for x in ['correlation', 'interaction', 'cross'])],
        'Temporal Features': [col for col in feature_cols if any(x in col.lower() for x in ['day', 'week', 'month', 'time'])]
    }
    
    for category, features in feature_categories.items():
        print(f"  {category}: {len(features)} features")
    
    print(f"\n📁 Upload '{output_file}' to Colab for advanced model training")
    
    return colab_data

if __name__ == "__main__":
    prepare_colab_dataset() 