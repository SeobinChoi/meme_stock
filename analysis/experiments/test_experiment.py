#!/usr/bin/env python3
"""
Simple test of the ML experiment
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

def test_experiment():
    print("🚀 Testing ML Experiment...")
    
    # Load data
    print("📊 Loading data...")
    train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
    test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Define features
    price_features = [
        'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
        'vol_5d', 'vol_10d', 'vol_20d', 'price_ratio_sma10', 
        'price_ratio_sma20', 'rsi_14', 'volume_ratio', 'turnover',
        'day_of_week', 'month', 'is_monday', 'is_friday', 
        'is_weekend_effect', 'market_vol_regime'
    ]
    
    reddit_features = [
        'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
        'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
        'reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14',
        'reddit_momentum_21', 'reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20',
        'reddit_percentile', 'reddit_high_regime', 'reddit_low_regime',
        'market_sentiment', 'price_reddit_momentum', 'vol_reddit_attention'
    ]
    
    # Check available features
    all_features = set(train_df.columns)
    available_price = [f for f in price_features if f in all_features]
    available_reddit = [f for f in reddit_features if f in all_features]
    
    print(f"   Available price features: {len(available_price)}/{len(price_features)}")
    print(f"   Available reddit features: {len(available_reddit)}/{len(reddit_features)}")
    
    # Test baseline model (price only)
    print("\n🤖 Testing Baseline Model (Price Only)...")
    features_baseline = available_price
    
    X_train = train_df[features_baseline].fillna(0).values
    X_test = test_df[features_baseline].fillna(0).values
    y_train = train_df['y1d'].values
    y_test = test_df['y1d'].values
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge model
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mask = np.isfinite(y_test) & np.isfinite(y_pred)
    y_test_clean = y_test[mask]
    y_pred_clean = y_pred[mask]
    
    ic, _ = pearsonr(y_pred_clean, y_test_clean)
    rank_ic, _ = spearmanr(y_pred_clean, y_test_clean)
    rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
    
    print(f"   Baseline Results:")
    print(f"   IC: {ic:.4f}")
    print(f"   Rank IC: {rank_ic:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Samples: {len(y_test_clean)}")
    
    # Test enhanced model (price + reddit)
    if len(available_reddit) > 0:
        print("\n🚀 Testing Enhanced Model (Price + Reddit)...")
        features_enhanced = available_price + available_reddit
        
        X_train = train_df[features_enhanced].fillna(0).values
        X_test = test_df[features_enhanced].fillna(0).values
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ridge model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mask = np.isfinite(y_test) & np.isfinite(y_pred)
        y_test_clean = y_test[mask]
        y_pred_clean = y_pred[mask]
        
        ic_enhanced, _ = pearsonr(y_pred_clean, y_test_clean)
        rank_ic_enhanced, _ = spearmanr(y_pred_clean, y_test_clean)
        rmse_enhanced = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
        
        print(f"   Enhanced Results:")
        print(f"   IC: {ic_enhanced:.4f}")
        print(f"   Rank IC: {rank_ic_enhanced:.4f}")
        print(f"   RMSE: {rmse_enhanced:.4f}")
        print(f"   Samples: {len(y_test_clean)}")
        
        # Improvement
        ic_improvement = ic_enhanced - ic
        rank_ic_improvement = rank_ic_enhanced - rank_ic
        
        print(f"\n📊 Reddit Data Impact:")
        print(f"   IC Improvement: {ic_improvement:+.4f}")
        print(f"   Rank IC Improvement: {rank_ic_improvement:+.4f}")
        
        if rank_ic_improvement > 0.01:
            print("   ✅ Strong evidence for Reddit data value!")
        elif rank_ic_improvement > 0.005:
            print("   ⚠️ Moderate evidence for Reddit data value")
        else:
            print("   ❌ Limited evidence for Reddit data value")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_experiment()