#!/usr/bin/env python3
"""
🔍 IC 값 검증 스크립트
의심스럽게 높은 IC 값들을 검증해보자
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import json
import glob

def calculate_ic_manually(predictions, actuals, method='spearman'):
    """수동으로 IC 계산"""
    
    # NaN 제거
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    if mask.sum() < 2:
        return 0, 1
    
    pred_clean = predictions[mask]
    actual_clean = actuals[mask]
    
    if method == 'spearman':
        return spearmanr(pred_clean, actual_clean)
    else:
        return pearsonr(pred_clean, actual_clean)

def load_and_verify_data():
    """데이터 로드하고 검증"""
    
    print("🔍 IC 값 검증 시작...")
    
    # 최신 결과 파일 찾기
    result_files = glob.glob('models/price_prediction/predictions/evaluation_results_*.json')
    if not result_files:
        print("❌ 결과 파일을 찾을 수 없음")
        return
    
    latest_file = sorted(result_files)[-1]
    print(f"📁 검증할 파일: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return
    
    # 각 모델의 예측값과 실제값 확인
    print("\n📊 모델별 IC 값 검증:")
    
    for model_name, model_data in results.items():
        if not isinstance(model_data, dict) or 'predictions' not in model_data:
            continue
            
        predictions = np.array(model_data['predictions'])
        actuals = np.array(model_data['actuals'])
        
        # 기본 통계
        print(f"\n🔍 {model_name}:")
        print(f"   샘플 수: {len(predictions)}")
        print(f"   예측값 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"   실제값 범위: [{actuals.min():.4f}, {actuals.max():.4f}]")
        print(f"   예측값 표준편차: {predictions.std():.4f}")
        print(f"   실제값 표준편차: {actuals.std():.4f}")
        
        # IC 재계산
        ic_pearson, p_pearson = calculate_ic_manually(predictions, actuals, 'pearson')
        ic_spearman, p_spearman = calculate_ic_manually(predictions, actuals, 'spearman')
        
        print(f"   📈 재계산 Pearson IC: {ic_pearson:.6f} (p={p_pearson:.6f})")
        print(f"   📈 재계산 Spearman IC: {ic_spearman:.6f} (p={p_spearman:.6f})")
        
        # 원래 결과와 비교
        if 'ic' in model_data:
            orig_ic = model_data['ic']
            print(f"   📋 원래 IC: {orig_ic:.6f}")
            print(f"   🔄 차이: {abs(ic_pearson - orig_ic):.6f}")
        
        # 의심스러운 경우 경고
        if abs(ic_spearman) > 0.05:
            print(f"   ⚠️ 높은 IC 값 감지! ({ic_spearman:.6f})")
            
            # 데이터 누수 확인
            if check_data_leakage(predictions, actuals):
                print(f"   🚨 데이터 누수 의심!")

def check_data_leakage(predictions, actuals):
    """데이터 누수 확인"""
    
    # 완전히 동일한 값들이 있는지 확인
    identical_ratio = np.mean(np.abs(predictions - actuals) < 1e-10)
    if identical_ratio > 0.1:  # 10% 이상이 동일
        return True
    
    # 너무 높은 상관관계 확인 (> 0.9)
    corr = np.corrcoef(predictions, actuals)[0, 1]
    if abs(corr) > 0.9:
        return True
    
    # 순서가 거의 동일한지 확인
    rank_corr = spearmanr(predictions, actuals)[0]
    if abs(rank_corr) > 0.95:
        return True
        
    return False

def check_time_series_issues():
    """시계열 관련 문제 확인"""
    
    print("\n🕐 시계열 데이터 검증...")
    
    # 훈련/검증/테스트 데이터셋 로드
    try:
        train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # 날짜 변환
        for df in [train_df, val_df, test_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        print(f"📊 데이터셋 날짜 범위:")
        print(f"   Train: {train_df['date'].min()} ~ {train_df['date'].max()}")
        print(f"   Val: {val_df['date'].min()} ~ {val_df['date'].max()}")
        print(f"   Test: {test_df['date'].min()} ~ {test_df['date'].max()}")
        
        # 데이터 누수 확인 (날짜 겹침)
        train_dates = set(train_df['date'])
        val_dates = set(val_df['date'])
        test_dates = set(test_df['date'])
        
        overlap_train_val = len(train_dates & val_dates)
        overlap_val_test = len(val_dates & test_dates)
        overlap_train_test = len(train_dates & test_dates)
        
        if overlap_train_val > 0:
            print(f"   🚨 Train-Val 날짜 겹침: {overlap_train_val}개")
        if overlap_val_test > 0:
            print(f"   🚨 Val-Test 날짜 겹침: {overlap_val_test}개")
        if overlap_train_test > 0:
            print(f"   🚨 Train-Test 날짜 겹침: {overlap_train_test}개")
            
        if overlap_train_val == 0 and overlap_val_test == 0 and overlap_train_test == 0:
            print("   ✅ 날짜 겹침 없음 (정상)")
            
    except Exception as e:
        print(f"   ❌ 데이터셋 로드 실패: {e}")

def main():
    """메인 실행"""
    
    print("🔍 IC 값 검증 및 데이터 누수 확인")
    print("=" * 50)
    
    # 1. IC 값 재계산 및 검증
    load_and_verify_data()
    
    # 2. 시계열 데이터 누수 확인
    check_time_series_issues()
    
    print("\n" + "=" * 50)
    print("✅ 검증 완료!")
    
if __name__ == "__main__":
    main()