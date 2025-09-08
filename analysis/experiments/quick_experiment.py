#!/usr/bin/env python3
"""
Quick ML experiment focused on key models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

def quick_experiment():
    print("🚀 Quick Financial ML Experiment")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
    val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
    test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} samples")
    
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
    
    # Filter available features
    all_features = set(train_df.columns)
    price_features = [f for f in price_features if f in all_features]
    reddit_features = [f for f in reddit_features if f in all_features]
    
    print(f"   Using {len(price_features)} price features, {len(reddit_features)} Reddit features")
    
    # Prepare datasets
    results = {}
    experiments = ['Baseline (Price Only)', 'Enhanced (Price + Reddit)']
    
    for exp_name in experiments:
        print(f"\n📊 Running {exp_name}...")
        
        if exp_name == 'Baseline (Price Only)':
            features = price_features
        else:
            features = price_features + reddit_features
        
        print(f"   Features: {len(features)}")
        
        # Prepare data
        X_train = train_df[features].fillna(0).values.astype(np.float32)
        X_val = val_df[features].fillna(0).values.astype(np.float32)
        X_test = test_df[features].fillna(0).values.astype(np.float32)
        
        y_train = train_df['y1d'].values.astype(np.float32)
        y_val = val_df['y1d'].values.astype(np.float32)
        y_test = test_df['y1d'].values.astype(np.float32)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test models
        models = {
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'XGBoost': xgb.XGBRegressor(
                learning_rate=0.1, max_depth=6, n_estimators=50,
                random_state=42, verbosity=0
            ),
            'LightGBM': lgb.LGBMRegressor(
                learning_rate=0.1, num_leaves=31, n_estimators=50,
                random_state=42, verbosity=-1
            )
        }
        
        exp_results = {}
        
        for model_name, model in models.items():
            print(f"   Training {model_name}...")
            
            try:
                # Train model
                if model_name == 'Ridge':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mask = np.isfinite(y_test) & np.isfinite(y_pred)
                y_test_clean = y_test[mask]
                y_pred_clean = y_pred[mask]
                
                ic, _ = pearsonr(y_pred_clean, y_test_clean)
                rank_ic, _ = spearmanr(y_pred_clean, y_test_clean)
                hit_rate = np.mean(np.sign(y_pred_clean) == np.sign(y_test_clean))
                rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
                
                exp_results[model_name] = {
                    'ic': ic if not np.isnan(ic) else 0,
                    'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
                    'hit_rate': hit_rate,
                    'rmse': rmse,
                    'n_samples': len(y_test_clean)
                }
                
                print(f"     IC: {ic:.4f}, Rank IC: {rank_ic:.4f}, Hit Rate: {hit_rate:.4f}")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                exp_results[model_name] = {
                    'ic': 0, 'rank_ic': 0, 'hit_rate': 0.5, 'rmse': np.inf, 'n_samples': 0
                }
        
        results[exp_name] = exp_results
    
    # Analysis
    print("\n📈 Results Analysis")
    print("=" * 50)
    
    # Print comparison table
    print("\n📊 Performance Summary (Test Set):")
    print(f"{'Model':<12} {'Baseline IC':<12} {'Enhanced IC':<12} {'Improvement':<12}")
    print("-" * 50)
    
    improvements = []
    for model in ['Ridge', 'XGBoost', 'LightGBM']:
        if model in results['Baseline (Price Only)'] and model in results['Enhanced (Price + Reddit)']:
            baseline_ic = results['Baseline (Price Only)'][model]['rank_ic']
            enhanced_ic = results['Enhanced (Price + Reddit)'][model]['rank_ic']
            improvement = enhanced_ic - baseline_ic
            
            print(f"{model:<12} {baseline_ic:<12.4f} {enhanced_ic:<12.4f} {improvement:<+12.4f}")
            improvements.append(improvement)
    
    # Overall analysis
    if improvements:
        avg_improvement = np.mean(improvements)
        positive_improvements = sum(1 for x in improvements if x > 0)
        total_models = len(improvements)
        
        print(f"\n🚀 Reddit Data Impact:")
        print(f"   Average IC Improvement: {avg_improvement:+.4f}")
        print(f"   Models with positive improvement: {positive_improvements}/{total_models}")
        
        # Best models
        best_baseline = max(results['Baseline (Price Only)'].items(), 
                           key=lambda x: x[1]['rank_ic'])
        best_enhanced = max(results['Enhanced (Price + Reddit)'].items(), 
                           key=lambda x: x[1]['rank_ic'])
        
        print(f"\n🏆 Best Models:")
        print(f"   Baseline: {best_baseline[0]} (IC: {best_baseline[1]['rank_ic']:.4f})")
        print(f"   Enhanced: {best_enhanced[0]} (IC: {best_enhanced[1]['rank_ic']:.4f})")
        
        # Conclusions
        print(f"\n💡 Key Insights:")
        if avg_improvement > 0.01:
            print("   ✅ Strong evidence for Reddit data value!")
            recommendation = "Reddit data provides significant value - recommend inclusion"
        elif avg_improvement > 0.005:
            print("   ⚠️ Moderate evidence for Reddit data value")
            recommendation = "Reddit data provides moderate value - consider cost-benefit"
        else:
            print("   ❌ Limited evidence for Reddit data value")
            recommendation = "Limited evidence for Reddit data value in current form"
        
        # Check target performance
        best_ic = max(best_baseline[1]['rank_ic'], best_enhanced[1]['rank_ic'])
        if best_ic >= 0.03:
            print(f"   ✅ Target IC ≥ 0.03 achieved! Best IC: {best_ic:.4f}")
        else:
            print(f"   ⚠️ Target IC ≥ 0.03 not achieved. Best IC: {best_ic:.4f}")
        
        print(f"\n📋 Recommendation: {recommendation}")
        
        return {
            'avg_improvement': avg_improvement,
            'best_ic': best_ic,
            'positive_improvements': positive_improvements,
            'total_models': total_models,
            'recommendation': recommendation
        }
    
    return None

if __name__ == "__main__":
    result = quick_experiment()
    print(f"\n✅ Quick experiment completed!")