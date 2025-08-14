#!/usr/bin/env python3
"""
🔍 IC 값 상세 검증 - 데이터 누수 및 계산 오류 확인
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import json
from pathlib import Path

def load_actual_predictions():
    """실제 예측 결과 로드"""
    
    print("🔍 실제 예측 결과 분석...")
    
    # 평가 결과 파일 찾기
    eval_files = list(Path('models/price_prediction/predictions').glob('*.json'))
    if not eval_files:
        print("❌ 평가 결과 파일 없음")
        return
    
    latest_file = sorted(eval_files)[-1]
    print(f"📁 파일: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # 각 모델별 분석
    for model_name in ['ar', 'price_lgbm', 'reddit_lgbm']:
        if model_name not in data['predictions']:
            continue
            
        print(f"\n🔍 {model_name.upper()} 모델 분석:")
        
        predictions = data['predictions'][model_name]
        
        # 첫 번째 split만 분석
        if len(predictions) > 0:
            split_data = predictions[0]  # 첫 번째 split
            
            # 예측값과 실제값 추출
            y_pred = [item['y_pred'] for item in split_data]
            y_true = [item['y1d'] for item in split_data]
            
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            
            print(f"   샘플 수: {len(y_pred)}")
            print(f"   예측값 범위: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
            print(f"   실제값 범위: [{y_true.min():.6f}, {y_true.max():.6f}]")
            print(f"   예측값 표준편차: {y_pred.std():.6f}")
            print(f"   실제값 표준편차: {y_true.std():.6f}")
            
            # IC 재계산
            if len(y_pred) > 2:
                pearson_ic, pearson_p = pearsonr(y_pred, y_true)
                spearman_ic, spearman_p = spearmanr(y_pred, y_true)
                
                print(f"   📈 Pearson IC: {pearson_ic:.6f} (p={pearson_p:.6f})")
                print(f"   📈 Spearman IC: {spearman_ic:.6f} (p={spearman_p:.6f})")
                
                # 의심스러운 패턴 확인
                check_suspicious_patterns(y_pred, y_true, model_name)
            
            # 첫 10개 샘플 확인
            print(f"   📊 첫 10개 샘플:")
            for i in range(min(10, len(y_pred))):
                print(f"      {i+1}: pred={y_pred[i]:.6f}, true={y_true[i]:.6f}, diff={abs(y_pred[i]-y_true[i]):.6f}")

def check_suspicious_patterns(y_pred, y_true, model_name):
    """의심스러운 패턴 확인"""
    
    # 1. 동일한 값들 확인
    identical_ratio = np.mean(np.abs(y_pred - y_true) < 1e-8)
    if identical_ratio > 0.01:  # 1% 이상이 거의 동일
        print(f"   🚨 {identical_ratio:.1%}의 예측값이 실제값과 거의 동일!")
    
    # 2. 상수 예측 확인
    pred_variance = np.var(y_pred)
    if pred_variance < 1e-8:
        print(f"   🚨 모든 예측값이 거의 동일 (분산: {pred_variance:.2e})")
    
    # 3. 극단적 상관관계 확인
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    if abs(corr) > 0.8:
        print(f"   🚨 극단적으로 높은 상관관계: {corr:.6f}")
        
        # 산점도 패턴 확인
        if corr > 0.95:
            print("   🚨 거의 완벽한 양의 상관관계 - 데이터 누수 의심!")
    
    # 4. 미래 정보 사용 의심
    # 예측값이 실제값을 너무 잘 맞추는 경우
    mae = np.mean(np.abs(y_pred - y_true))
    std_true = np.std(y_true)
    
    if mae < std_true * 0.1:  # MAE가 표준편차의 10% 미만
        print(f"   🚨 예측 오차가 너무 작음: MAE={mae:.6f}, STD={std_true:.6f}")
        print("   🚨 미래 정보 사용 의심!")

def check_data_splits():
    """데이터 분할 검증"""
    
    print("\n🕐 데이터 분할 검증...")
    
    try:
        # 데이터셋 로드
        train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')  
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # 날짜 변환
        for df in [train_df, val_df, test_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        print(f"📊 Train: {train_df['date'].min()} ~ {train_df['date'].max()}")
        print(f"📊 Val: {val_df['date'].min()} ~ {val_df['date'].max()}")
        print(f"📊 Test: {test_df['date'].min()} ~ {test_df['date'].max()}")
        
        # 겹침 확인
        train_dates = set(train_df['date'])
        val_dates = set(val_df['date'])
        test_dates = set(test_df['date'])
        
        overlaps = [
            ('Train-Val', len(train_dates & val_dates)),
            ('Val-Test', len(val_dates & test_dates)),
            ('Train-Test', len(train_dates & test_dates))
        ]
        
        for name, count in overlaps:
            if count > 0:
                print(f"🚨 {name} 겹침: {count}개")
            else:
                print(f"✅ {name} 겹침 없음")
                
        # 시간 순서 확인
        if train_df['date'].max() >= val_df['date'].min():
            print(f"🚨 Train 데이터가 Val 데이터와 시간적으로 겹침!")
        if val_df['date'].max() >= test_df['date'].min():
            print(f"🚨 Val 데이터가 Test 데이터와 시간적으로 겹침!")
            
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")

def analyze_feature_target_correlation():
    """특성-타겟 상관관계 분석"""
    
    print("\n🔍 특성-타겟 상관관계 분석...")
    
    try:
        # 테스트 데이터 로드
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # 특성 컬럼 (타겟 제외)
        exclude_cols = ['date', 'ticker', 'ticker_type', 'y1d', 'y5d', 
                       'alpha_1d', 'alpha_5d', 'direction_1d', 'direction_5d']
        feature_cols = [col for col in test_df.columns if col not in exclude_cols]
        
        # 상관관계 계산
        correlations = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(test_df[col]):
                corr = test_df[col].corr(test_df['y1d'])
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
        
        # 상위 10개 특성
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("📊 타겟과 상관관계가 높은 특성들:")
        for i, (col, corr) in enumerate(correlations[:10]):
            print(f"   {i+1:2d}. {col}: {corr:.4f}")
            
            # 의심스럽게 높은 상관관계
            if corr > 0.8:
                print(f"       🚨 매우 높은 상관관계! 데이터 누수 의심")
                
    except Exception as e:
        print(f"❌ 특성 분석 실패: {e}")

def main():
    """메인 실행"""
    
    print("🔍 IC 값 상세 검증 및 데이터 누수 확인")
    print("=" * 60)
    
    # 1. 실제 예측 결과 분석
    load_actual_predictions()
    
    # 2. 데이터 분할 검증
    check_data_splits()
    
    # 3. 특성-타겟 상관관계 분석
    analyze_feature_target_correlation()
    
    print("\n" + "=" * 60)
    print("📋 검증 요약:")
    print("   1. IC 값이 0.08은 매우 높은 수준")
    print("   2. 일반적으로 금융 데이터에서 IC > 0.05는 의심스러움")
    print("   3. 데이터 누수나 미래 정보 사용 가능성 확인 필요")
    print("   4. 시계열 데이터의 정확한 분할 확인 필요")

if __name__ == "__main__":
    main()