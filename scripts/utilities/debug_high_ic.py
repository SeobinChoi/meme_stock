#!/usr/bin/env python3
"""
Debug High IC Issue
==================
IC 0.07-0.08은 비현실적으로 높음. 원인 분석:

1. Data Leakage 체크
2. Overfitting 확인  
3. Feature 검증
4. Train/Test Split 문제
5. 스케일링 이슈
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb

def debug_high_ic():
    """높은 IC 원인 분석"""
    print("🔍 DEBUGGING HIGH IC (0.07-0.08)")
    print("=" * 50)
    print("💡 정상적인 IC 범위: 0.01-0.05 (0.08은 비현실적)")
    
    # 1. 데이터 로드
    print("\n📊 1. DATA INSPECTION:")
    train_df = pd.read_csv("data/colab_datasets/tabular_train_20250814_031335.csv")
    test_df = pd.read_csv("data/colab_datasets/tabular_test_20250814_031335.csv")
    
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")
    
    # 타겟 분포 확인
    print(f"\n📈 2. TARGET DISTRIBUTION:")
    print(f"   Train y1d: mean={train_df['y1d'].mean():.6f}, std={train_df['y1d'].std():.6f}")
    print(f"   Test y1d: mean={test_df['y1d'].mean():.6f}, std={test_df['y1d'].std():.6f}")
    print(f"   Train range: [{train_df['y1d'].min():.4f}, {train_df['y1d'].max():.4f}]")
    print(f"   Test range: [{test_df['y1d'].min():.4f}, {test_df['y1d'].max():.4f}]")
    
    # 3. Feature 누수 체크
    print(f"\n🚨 3. DATA LEAKAGE CHECK:")
    
    # 미래 정보가 포함된 것 같은 feature들 체크
    suspicious_features = [col for col in train_df.columns 
                          if any(word in col.lower() for word in ['future', 'next', 'lead', 'ahead'])]
    print(f"   Suspicious feature names: {suspicious_features}")
    
    # y1d와 너무 높은 상관관계를 보이는 features
    feature_cols = [col for col in train_df.columns if col not in ['y1d', 'date', 'ticker']]
    high_corr_features = []
    
    for feature in feature_cols:
        if train_df[feature].dtype in ['float64', 'int64']:
            corr = train_df[feature].corr(train_df['y1d'])
            if abs(corr) > 0.5:  # 50% 이상 상관관계
                high_corr_features.append((feature, corr))
    
    print(f"   Features with |correlation| > 0.5 with y1d:")
    for feature, corr in sorted(high_corr_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"      {feature}: {corr:.4f}")
    
    # 4. Time Series Split 문제 체크
    print(f"\n📅 4. TIME SERIES VALIDATION:")
    
    # 날짜 정보가 있는지 확인
    if 'date' in train_df.columns:
        print("   Date column found - checking chronological order...")
        # 실제 시계열 분할이 되었는지 확인하려면 원본 데이터 필요
    else:
        print("   ⚠️ No date column - cannot verify time series split")
    
    # 5. 간단한 모델로 현실적 IC 테스트
    print(f"\n🤖 5. REALISTIC MODEL TEST:")
    
    # Top features만 사용해서 간단한 모델 테스트
    top_features = [
        'log_mentions', 'price_ratio_sma20', 'price_ratio_sma10', 
        'returns_3d', 'returns_1d', 'market_sentiment'
    ]
    
    available_features = [f for f in top_features if f in train_df.columns]
    print(f"   Using {len(available_features)} conservative features")
    
    # 데이터 준비
    X_train = train_df[available_features].fillna(0).values
    y_train = train_df['y1d'].values
    X_test = test_df[available_features].fillna(0).values  
    y_test = test_df['y1d'].values
    
    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 간단한 선형 모델부터 테스트
    from sklearn.linear_model import Ridge
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)
    
    ridge_ic, _ = pearsonr(y_test, ridge_pred)
    ridge_hit_rate = np.mean(np.sign(y_test) == np.sign(ridge_pred))
    
    print(f"   Ridge Regression:")
    print(f"      IC: {ridge_ic:.4f}")
    print(f"      Hit Rate: {ridge_hit_rate:.3f}")
    
    # LightGBM with conservative parameters
    lgb_conservative = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=20,  # 매우 작게
        max_depth=3,    # 얕게
        learning_rate=0.1,
        n_estimators=50,  # 적게
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,  # 강한 정규화
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1
    )
    
    lgb_conservative.fit(X_train_scaled, y_train)
    lgb_pred = lgb_conservative.predict(X_test_scaled)
    
    lgb_ic, _ = pearsonr(y_test, lgb_pred)
    lgb_hit_rate = np.mean(np.sign(y_test) == np.sign(lgb_pred))
    
    print(f"   Conservative LightGBM:")
    print(f"      IC: {lgb_ic:.4f}")
    print(f"      Hit Rate: {lgb_hit_rate:.3f}")
    
    # 6. Cross-validation으로 안정성 체크
    print(f"\n🔄 6. CROSS-VALIDATION STABILITY:")
    
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=3)
    cv_ics = []
    
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        
        cv_model = Ridge(alpha=1.0)
        cv_model.fit(X_cv_train, y_cv_train)
        cv_pred = cv_model.predict(X_cv_val)
        
        cv_ic, _ = pearsonr(y_cv_val, cv_pred)
        cv_ics.append(cv_ic)
    
    print(f"   CV IC scores: {[f'{ic:.4f}' for ic in cv_ics]}")
    print(f"   CV IC mean: {np.mean(cv_ics):.4f}")
    print(f"   CV IC std: {np.std(cv_ics):.4f}")
    
    # 7. Feature importance 분석
    print(f"\n🔍 7. FEATURE IMPORTANCE ANALYSIS:")
    
    # Ridge 계수 확인
    feature_importance = dict(zip(available_features, ridge_model.coef_))
    top_important = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"   Top important features (Ridge coefficients):")
    for feature, coef in top_important:
        print(f"      {feature}: {coef:.4f}")
    
    # 8. 예측값 분포 확인
    print(f"\n📊 8. PREDICTION DISTRIBUTION:")
    print(f"   Ridge predictions: mean={np.mean(ridge_pred):.6f}, std={np.std(ridge_pred):.6f}")
    print(f"   LightGBM predictions: mean={np.mean(lgb_pred):.6f}, std={np.std(lgb_pred):.6f}")
    print(f"   Actual values: mean={np.mean(y_test):.6f}, std={np.std(y_test):.6f}")
    
    # 예측값이 너무 실제값과 비슷한지 체크 (오버피팅 신호)
    pred_vs_actual_corr = pearsonr(ridge_pred, y_test)[0]
    print(f"   Prediction vs Actual correlation: {pred_vs_actual_corr:.4f}")
    
    # 9. 결론 및 권장사항
    print(f"\n💡 9. CONCLUSIONS & RECOMMENDATIONS:")
    
    # 현실적인 IC 범위인지 체크
    realistic_ics = [ridge_ic, lgb_ic] + cv_ics
    max_realistic_ic = max(abs(ic) for ic in realistic_ics)
    
    print(f"   Realistic IC range: {min(realistic_ics):.4f} to {max(realistic_ics):.4f}")
    print(f"   Maximum |IC| achieved: {max_realistic_ic:.4f}")
    
    if max_realistic_ic > 0.05:
        print("   ⚠️ Still high IC - possible data issues")
    elif max_realistic_ic > 0.03:
        print("   ✅ High but reasonable IC - good model")
    elif max_realistic_ic > 0.01:
        print("   ✅ Moderate IC - acceptable performance")
    else:
        print("   ❌ Low IC - model not effective")
    
    # 원래 높은 IC 원인 추정
    print(f"\n🎯 LIKELY CAUSES OF HIGH IC (0.07-0.08):")
    if len(high_corr_features) > 0:
        print("   1. ⚠️ High correlation features detected - possible leakage")
    if max_realistic_ic < 0.03:
        print("   2. ⚠️ Conservative models show much lower IC - overfitting")
    print("   3. ⚠️ Complex models (deep trees) may be memorizing patterns")
    print("   4. ⚠️ Small dataset size can lead to unstable high IC")
    
    print(f"\n📋 RECOMMENDATIONS:")
    print("   1. Use conservative model parameters")
    print("   2. Increase regularization (alpha/lambda)")
    print("   3. Reduce model complexity (depth, leaves)")
    print("   4. Cross-validate all results")
    print("   5. Check for data leakage carefully")
    print(f"   6. Report conservative IC (~{max_realistic_ic:.3f}) in paper")

def check_data_leakage_detailed():
    """데이터 누수 상세 체크"""
    print("\n🚨 DETAILED DATA LEAKAGE CHECK")
    print("=" * 40)
    
    # Colab 데이터셋 메타데이터 확인
    try:
        with open("data/colab_datasets/dataset_metadata_20250814_031335.json", 'r') as f:
            import json
            metadata = json.load(f)
            
        print("📋 Dataset metadata found:")
        if 'feature_info' in metadata:
            print("   Feature info available")
        if 'creation_process' in metadata:
            print(f"   Creation process: {metadata['creation_process']}")
            
    except Exception as e:
        print(f"   No metadata file: {e}")
    
    # 원본 고급 features 데이터셋과 비교
    try:
        original_df = pd.read_csv("data/features/advanced_meme_features_dataset.csv", nrows=5)
        colab_df = pd.read_csv("data/colab_datasets/tabular_train_20250814_031335.csv", nrows=5)
        
        print(f"\n📊 Dataset comparison:")
        print(f"   Original features: {original_df.shape[1]}")
        print(f"   Colab features: {colab_df.shape[1]}")
        
        # 공통 features
        common_features = set(original_df.columns) & set(colab_df.columns)
        print(f"   Common features: {len(common_features)}")
        
    except Exception as e:
        print(f"   Cannot compare datasets: {e}")

if __name__ == "__main__":
    debug_high_ic()
    check_data_leakage_detailed()
    
    print(f"\n✅ Debug analysis completed!")
    print(f"💡 Use conservative IC estimates for paper")

