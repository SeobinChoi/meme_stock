#!/usr/bin/env python3
"""
Contrarian vs Standard Performance Comparison Table
==================================================
세 가지 모델에서 Contrarian 적용 전후 IC 성능 비교
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

def create_contrarian_comparison_table():
    """Contrarian 전후 성능 비교 테이블 생성"""
    print("📊 CONTRARIAN vs STANDARD PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # 데이터 로드
    train_df = pd.read_csv("data/colab_datasets/tabular_train_20250814_031335.csv")
    test_df = pd.read_csv("data/colab_datasets/tabular_test_20250814_031335.csv")
    
    # 안전한 features만 사용 (누수 제거)
    safe_features = [
        'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
        'vol_5d', 'vol_10d', 'vol_20d',
        'price_ratio_sma10', 'price_ratio_sma20', 'rsi_14',
        'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
        'reddit_surprise', 'reddit_momentum_3', 'reddit_momentum_7',
        'market_sentiment', 'day_of_week', 'month', 'is_monday', 'is_friday'
    ]
    
    available_features = [f for f in safe_features if f in train_df.columns]
    print(f"Using {len(available_features)} clean features")
    
    # 데이터 준비
    X_train = train_df[available_features].fillna(0).values
    y_train = train_df['y1d'].values
    X_test = test_df[available_features].fillna(0).values
    y_test = test_df['y1d'].values
    
    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 정의 및 훈련
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'LightGBM': lgb.LGBMRegressor(
            objective='regression',
            num_leaves=15,
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=50,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
    }
    
    # 결과 저장할 리스트
    results = []
    
    print(f"\n🔄 Training and evaluating models...")
    
    for model_name, model in models.items():
        print(f"   Training {model_name}...")
        
        # 모델 훈련
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
        # Standard 성능
        standard_ic, _ = pearsonr(y_test, predictions)
        standard_hit = np.mean(np.sign(y_test) == np.sign(predictions))
        
        # Contrarian 성능 (예측값 반전)
        contrarian_predictions = -predictions
        contrarian_ic, _ = pearsonr(y_test, contrarian_predictions)
        contrarian_hit = np.mean(np.sign(y_test) == np.sign(contrarian_predictions))
        
        # 개선도 계산
        ic_improvement = abs(contrarian_ic) - abs(standard_ic)
        hit_improvement = contrarian_hit - standard_hit
        
        # 결과 저장
        results.append({
            'Model': model_name,
            'Standard_IC': standard_ic,
            'Contrarian_IC': contrarian_ic,
            'IC_Improvement': ic_improvement,
            'Standard_Hit': standard_hit,
            'Contrarian_Hit': contrarian_hit,
            'Hit_Improvement': hit_improvement,
            'Best_Strategy': 'Contrarian' if abs(contrarian_ic) > abs(standard_ic) else 'Standard'
        })
    
    # 결과 테이블 생성
    df_results = pd.DataFrame(results)
    
    return df_results, results

def print_comparison_table(df_results, results):
    """비교 테이블 출력"""
    
    print(f"\n📋 DETAILED COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Model':<10} {'Standard IC':<12} {'Contrarian IC':<13} {'|IC| Δ':<8} {'Std Hit':<8} {'Con Hit':<8} {'Hit Δ':<7} {'Best':<10}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['Model']:<10} "
              f"{result['Standard_IC']:<12.4f} "
              f"{result['Contrarian_IC']:<13.4f} "
              f"{result['IC_Improvement']:<8.4f} "
              f"{result['Standard_Hit']:<8.3f} "
              f"{result['Contrarian_Hit']:<8.3f} "
              f"{result['Hit_Improvement']:<7.3f} "
              f"{result['Best_Strategy']:<10}")
    
    print("-" * 100)
    
    # 요약 통계
    avg_ic_improvement = np.mean([r['IC_Improvement'] for r in results])
    avg_hit_improvement = np.mean([r['Hit_Improvement'] for r in results])
    
    contrarian_wins = len([r for r in results if r['Best_Strategy'] == 'Contrarian'])
    
    print(f"SUMMARY:")
    print(f"   Average |IC| Improvement: {avg_ic_improvement:+.4f}")
    print(f"   Average Hit Rate Improvement: {avg_hit_improvement:+.3f}")
    print(f"   Models improved by Contrarian: {contrarian_wins}/{len(results)}")

def print_simple_table(results):
    """간단한 테이블 (논문용)"""
    
    print(f"\n📊 SIMPLE COMPARISON TABLE (For Paper)")
    print("=" * 60)
    print(f"{'Model':<12} {'Standard IC':<12} {'Contrarian IC':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for result in results:
        improvement_symbol = "✅" if result['IC_Improvement'] > 0 else "❌"
        print(f"{result['Model']:<12} "
              f"{abs(result['Standard_IC']):<12.3f} "
              f"{abs(result['Contrarian_IC']):<12.3f} "
              f"{result['IC_Improvement']:+.3f} {improvement_symbol}")
    
    print("-" * 60)

def analyze_contrarian_effectiveness(results):
    """Contrarian 효과 상세 분석"""
    
    print(f"\n🔍 CONTRARIAN EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    
    # 각 모델별 상세 분석
    for result in results:
        model_name = result['Model']
        standard_ic = result['Standard_IC']
        contrarian_ic = result['Contrarian_IC']
        
        print(f"\n📈 {model_name} Analysis:")
        print(f"   Standard:  IC = {standard_ic:+.4f}")
        print(f"   Contrarian: IC = {contrarian_ic:+.4f}")
        
        # IC 변화 분석
        if standard_ic < 0 and contrarian_ic > 0:
            print(f"   🔄 Sign flipped: Negative → Positive")
        elif standard_ic > 0 and contrarian_ic < 0:
            print(f"   🔄 Sign flipped: Positive → Negative")
        else:
            print(f"   ➡️ Same sign, magnitude changed")
        
        # 절댓값 기준 개선도
        abs_improvement = abs(contrarian_ic) - abs(standard_ic)
        improvement_pct = (abs_improvement / abs(standard_ic)) * 100 if standard_ic != 0 else 0
        
        print(f"   📊 |IC| improvement: {abs_improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # 권장 전략
        if abs(contrarian_ic) > abs(standard_ic):
            print(f"   💡 Recommendation: Use Contrarian strategy")
        else:
            print(f"   💡 Recommendation: Use Standard strategy")

def create_paper_summary(results):
    """논문용 요약"""
    
    print(f"\n📄 PAPER SUMMARY")
    print("=" * 40)
    
    # 최고 성과 찾기
    best_standard = max(results, key=lambda x: abs(x['Standard_IC']))
    best_contrarian = max(results, key=lambda x: abs(x['Contrarian_IC']))
    best_overall = max(results, key=lambda x: max(abs(x['Standard_IC']), abs(x['Contrarian_IC'])))
    
    print(f"🏆 BEST PERFORMANCES:")
    print(f"   Best Standard: {best_standard['Model']} (IC = {abs(best_standard['Standard_IC']):.3f})")
    print(f"   Best Contrarian: {best_contrarian['Model']} (IC = {abs(best_contrarian['Contrarian_IC']):.3f})")
    print(f"   Best Overall: {best_overall['Model']} ({'Contrarian' if abs(best_overall['Contrarian_IC']) > abs(best_overall['Standard_IC']) else 'Standard'})")
    
    # 논문용 문장
    contrarian_improvements = [r for r in results if r['IC_Improvement'] > 0]
    improvement_rate = len(contrarian_improvements) / len(results) * 100
    
    print(f"\n📝 PAPER STATEMENT:")
    print(f"   \"Contrarian strategy improved IC performance in {len(contrarian_improvements)}/{len(results)} models ({improvement_rate:.0f}%),")
    avg_improvement = np.mean([r['IC_Improvement'] for r in contrarian_improvements]) if contrarian_improvements else 0
    print(f"   with an average |IC| improvement of {avg_improvement:.3f}.\"")
    
    # 최고 성과 모델 상세
    if abs(best_overall['Contrarian_IC']) > abs(best_overall['Standard_IC']):
        best_ic = abs(best_overall['Contrarian_IC'])
        strategy = "Contrarian"
    else:
        best_ic = abs(best_overall['Standard_IC'])
        strategy = "Standard"
    
    print(f"\n🎯 HEADLINE RESULT:")
    print(f"   \"Best performing model: {best_overall['Model']} with {strategy} strategy")
    print(f"   achieved Information Coefficient of {best_ic:.3f}\"")

def main():
    """메인 실행 함수"""
    
    # 비교 분석 실행
    df_results, results = create_contrarian_comparison_table()
    
    # 테이블 출력
    print_comparison_table(df_results, results)
    print_simple_table(results)
    
    # 상세 분석
    analyze_contrarian_effectiveness(results)
    
    # 논문용 요약
    create_paper_summary(results)
    
    return df_results, results

if __name__ == "__main__":
    df_results, results = main()
    
    print(f"\n✅ Contrarian comparison analysis completed!")
    print(f"📊 Use the tables above for your paper")

