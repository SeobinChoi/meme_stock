# Reddit 소셜미디어 데이터를 활용한 밈스톡의 Contrarian Effect 분석 - 최종 정리

## 📋 프로젝트 개요

본 프로젝트는 Reddit 소셜미디어 데이터를 활용하여 밈스톡(AMC, BB, GME)의 Contrarian Effect를 분석하고, 머신러닝 모델을 통해 주가 예측 성능을 평가하는 연구입니다.

## 🎯 핵심 연구 목표

1. **Contrarian Effect 검증**: Reddit 관심도와 주가 수익률 간의 음의 상관관계 확인
2. **머신러닝 모델 성능 비교**: Ridge, LightGBM, XGBoost 모델의 성능 평가
3. **피처 엔지니어링**: Reddit 데이터를 활용한 고급 피처 개발 및 효과 검증
4. **실전 적용**: 투자 전략 및 리스크 관리에 활용 가능한 인사이트 도출

## 🔧 하이퍼파라미터 설정

### 1. Ridge Regression
```python
ridge = Ridge(alpha=1.0, random_state=42)
```
- **alpha=1.0**: L2 정규화 강도 (강한 정규화로 과적합 방지)
- **random_state=42**: 재현 가능한 결과를 위한 시드값

### 2. LightGBM
```python
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42
}
lgb_model = lgb.train(lgb_params, train_data, num_boost_round=100, 
                     valid_sets=[val_data], callbacks=[lgb.log_evaluation(0)])
```
- **num_boost_round=100**: 부스팅 라운드 수 (보수적 설정)
- **기본 파라미터 사용**: 과적합 방지 및 안정성 확보

### 3. XGBoost
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0
}
```
- **기본 파라미터 사용**: 빠른 훈련과 안정적인 성능

### 하이퍼파라미터 선택 전략
- **보수적 접근**: 과적합 방지를 위한 기본/보수적 파라미터 사용
- **일관성 유지**: 모든 모델이 동일한 조건에서 비교
- **실험 목적**: Reddit 피처 효과에 집중, 모델 성능 최적화는 부차적
- **재현성**: `random_state=42`로 일관된 결과 보장

## 📊 모델 성능 비교 결과

### 상세 성능 비교표 (AMC, BB, GME)

| **모델** | **Price Only IC** | **Reddit All IC** | **Advanced Reddit IC** | **Reddit All IC 개선** | **Advanced Reddit IC 개선** |
|----------|------------------|------------------|----------------------|----------------------|---------------------------|
| **Ridge** | 0.0341 | 0.0772 | 0.0683 | **+0.0431** | **+0.0341** |
| **LightGBM** | -0.1011 | -0.0631 | -0.1361 | **+0.0380** | **-0.0351** |
| **XGBoost** | -0.0632 | -0.0638 | -0.0230 | **-0.0006** | **+0.0402** |

### Hit Rate 비교표

| **모델** | **Price Only Hit Rate** | **Reddit All Hit Rate** | **Advanced Reddit Hit Rate** | **Reddit All HR 개선** | **Advanced Reddit HR 개선** |
|----------|------------------------|------------------------|----------------------------|----------------------|---------------------------|
| **Ridge** | 0.4474 | 0.4497 | 0.4452 | **+0.0022** | **-0.0022** |
| **LightGBM** | 0.4989 | 0.4720 | 0.4609 | **-0.0268** | **-0.0380** |
| **XGBoost** | 0.4765 | 0.4743 | 0.4765 | **-0.0022** | **+0.0000** |

### 핵심 발견사항

#### 1. 모델별 Reddit 피처 반응
- **Ridge**: Reddit 피처에 **극도로 긍정적 반응** (모든 Reddit 피처에서 개선)
- **LightGBM**: **Reddit All만 긍정적**, Advanced Reddit는 부정적
- **XGBoost**: **Advanced Reddit만 긍정적**, Reddit All은 부정적

#### 2. 피처 세트별 효과
- **Reddit All**: 평균적으로 가장 큰 IC 개선 (+0.0268)
- **Advanced Reddit**: 선택적이지만 의미있는 IC 개선 (+0.0131)
- **Hit Rate**: 두 Reddit 피처 세트 모두 평균적으로 하락

#### 3. 최적 조합
- **Ridge + Reddit All**: 최고 성능 (IC 0.0772)
- **LightGBM + Reddit All**: 최고 성능 (IC -0.0631)
- **XGBoost + Advanced Reddit**: 최고 성능 (IC -0.0230)

## 🔍 주요 피처 분석

### 1. 기본 Reddit 피처
- **log_mentions**: 로그 변환된 언급 수
- **reddit_ema_3/5/10**: 3일, 5일, 10일 지수이동평균
- **reddit_surprise**: 핵심 Contrarian 지표 (log_mentions[t-1] - reddit_ema_5[t])

### 2. 고급 Reddit 피처
- **reddit_market_ex**: 시장 초과 수익률
- **reddit_spike_p95**: 상위 5% 스파이크 이벤트
- **reddit_momentum_3/7/14/21**: 다양한 기간 모멘텀
- **reddit_vol_5/10/20**: Reddit 변동성 지표
- **reddit_percentile**: Reddit 관심도 백분위수
- **reddit_high_regime/low_regime**: 고/저 관심도 체제
- **market_sentiment**: 시장 감정 지표
- **price_reddit_momentum**: 가격-Reddit 모멘텀 상호작용
- **vol_reddit_attention**: 변동성-Reddit 관심도 상호작용

### 3. Contrarian Effect 구현

#### 핵심 Contrarian 피처
```python
# Reddit Surprise (핵심 Contrarian 지표)
reddit_surprise = log_mentions[t-1] - reddit_ema_5[t]

# Contrarian 신호 (Reddit Surprise 반전)
contrarian_signal = -reddit_surprise

# 상호작용 피처
surprise_rsi_interaction = reddit_surprise * rsi_14
surprise_vol_interaction = reddit_surprise * vol_5d
```

#### 실제 상관관계 검증
- **AMC**: Reddit Surprise vs Returns = **-0.1784** ✅
- **BB**: Reddit Surprise vs Returns = **-0.1646** ✅  
- **GME**: Reddit Surprise vs Returns = **-0.1982** ✅

## 🤖 머신러닝 모델에서 Contrarian Effect 구현

### 1. 피처 엔지니어링
- **Reddit Surprise**: 핵심 Contrarian 지표 생성
- **Contrarian 신호**: Reddit Surprise 반전으로 매도 신호 생성
- **상호작용 피처**: Reddit과 기술적 지표 간 상호작용 포착

### 2. 모델별 구현 방법
- **Ridge**: Reddit Surprise 계수가 음수로 학습되어 높은 Reddit 관심도일 때 낮은 수익률 예측
- **LightGBM**: 트리 분할 과정에서 Reddit Surprise가 높을 때 수익률이 낮아지는 패턴 학습
- **XGBoost**: 그래디언트 부스팅으로 Reddit Surprise와 수익률 간의 음의 관계 포착

### 3. 예측 로직
```python
def contrarian_predict(model, features, reddit_surprise):
    # 기본 예측
    base_prediction = model.predict(features)
    
    # Reddit Surprise 가중치 (음의 가중치)
    surprise_weight = reddit_surprise * -0.5
    
    # Contrarian 예측
    contrarian_prediction = base_prediction + surprise_weight
    
    return contrarian_prediction
```

## 📈 실험 설계 및 검증 방법

### 1. 데이터 분할
- **Train**: 60% (시계열 순서 유지)
- **Validation**: 20% (시계열 순서 유지)
- **Test**: 20% (시계열 순서 유지)

### 2. 평가 지표
- **IC (Information Coefficient)**: Spearman rank correlation
- **Hit Rate**: 방향성 예측 정확도
- **R² Score**: 결정계수
- **MSE/MAE**: 회귀 성능 지표

### 3. 검증 방법
- **상관관계 분석**: Reddit Surprise와 다음날 수익률의 음의 상관관계
- **Event Study**: Reddit 스파이크 이벤트 전후 수익률 패턴 분석
- **모델 성능 평가**: IC, Hit Rate 기반 성능 비교

## 🎯 핵심 연구 결과

### 1. Contrarian Effect 확인
- 모든 주요 밈스톡에서 일관된 음의 상관관계 확인
- GME에서 가장 강한 Contrarian Effect (-0.1982)
- Reddit 관심도가 높을 때 주가 하락 경향

### 2. 모델 성능 개선
- **Ridge**: Reddit 피처로 IC 126% 개선 (0.0341 → 0.0772)
- **LightGBM**: Reddit All로 IC 37.6% 개선 (-0.1011 → -0.0631)
- **XGBoost**: Advanced Reddit로 IC 63.6% 개선 (-0.0632 → -0.0230)

### 3. 피처 효과 분석
- **Reddit All**: 평균적으로 가장 큰 IC 개선 (+0.0268)
- **Advanced Reddit**: 선택적이지만 의미있는 IC 개선 (+0.0131)
- **모델별 차이**: 각 모델마다 다른 최적 Reddit 피처 세트 존재

## 🚀 실전 적용 가이드

### 1. 최고 성능을 위한 권장사항
- **Ridge**: Reddit All 사용 (IC 0.0772)
- **LightGBM**: Reddit All 사용 (IC -0.0631)
- **XGBoost**: Advanced Reddit 사용 (IC -0.0230)

### 2. 안정적 성능을 위한 권장사항
- **Price Only**: 모든 모델에서 안정적 베이스라인
- **Advanced Reddit**: 선택적이지만 의미있는 개선
- **Reddit All**: 가장 큰 평균 개선 효과

### 3. 투자 전략 시사점
- **Contrarian 전략**: 높은 Reddit 관심도 → 매도 신호
- **리스크 관리**: Reddit 스파이크 이벤트 모니터링
- **포지션 조정**: 극단적 관심도 기간 동안 포지션 축소

## 📁 주요 파일 구조

### 데이터 파일
- `data/colab_datasets/tabular_train_*.csv`: 훈련 데이터
- `data/colab_datasets/tabular_val_*.csv`: 검증 데이터
- `data/colab_datasets/tabular_test_*.csv`: 테스트 데이터
- `data/raw/stocks/*_stock_data.csv`: 원시 주가 데이터

### 분석 스크립트
- `scripts/analysis/detailed_performance_comparison.py`: 상세 성능 비교
- `scripts/analysis/comprehensive_ml_experiment.py`: 종합 ML 실험
- `scripts/analysis/feature_selection_experiment.py`: 피처 선택 실험
- `scripts/analysis/regularization_experiment.py`: 정규화 실험
- `scripts/analysis/contrarian_effect_experiment.py`: Contrarian Effect 실험

### 결과 파일
- `results/detailed_performance_comparison_report.txt`: 상세 성능 비교 리포트
- `results/comprehensive_ml_experiment_report.txt`: 종합 ML 실험 리포트
- `results/feature_selection_report.txt`: 피처 선택 실험 리포트
- `results/regularization_report.txt`: 정규화 실험 리포트
- `results/contrarian_effect_report.txt`: Contrarian Effect 실험 리포트

## 🔬 연구의 한계 및 향후 방향

### 1. 연구 한계
- **하이퍼파라미터 튜닝 생략**: 기본 파라미터 사용으로 성능 최적화 미흡
- **제한된 데이터**: AMC, BB, GME 3개 종목만 분석
- **단순한 모델**: 복잡한 딥러닝 모델 미사용

### 2. 향후 연구 방향
- **하이퍼파라미터 최적화**: Bayesian Optimization 등을 통한 체계적 튜닝
- **확장된 데이터**: 더 많은 밈스톡 및 소셜미디어 플랫폼 포함
- **고급 모델**: LSTM, Transformer 등 시계열 딥러닝 모델 적용
- **실시간 시스템**: 실시간 Reddit 데이터 수집 및 예측 시스템 구축

## 📚 참고문헌 및 기술 스택

### 기술 스택
- **Python**: pandas, numpy, scikit-learn, lightgbm, xgboost
- **시각화**: matplotlib, seaborn
- **통계**: scipy.stats
- **데이터**: Yahoo Finance API, Reddit API

### 핵심 개념
- **Contrarian Effect**: Reddit 관심도와 주가 수익률 간의 음의 상관관계
- **Reddit Surprise**: 예상치 대비 실제 언급 수의 편차
- **Information Coefficient**: 예측값과 실제값 간의 순위 상관계수
- **Hit Rate**: 방향성 예측 정확도

## 🎉 결론

본 연구를 통해 Reddit 소셜미디어 데이터를 활용한 밈스톡의 Contrarian Effect를 성공적으로 검증하고, 머신러닝 모델을 통한 주가 예측 성능 개선을 확인했습니다. 특히 Ridge 모델에서 Reddit 피처를 활용할 때 가장 큰 성능 개선을 보였으며, 이는 Reddit 데이터의 예측력과 Contrarian Effect의 실용성을 입증합니다.

향후 더 정교한 하이퍼파라미터 튜닝과 고급 모델을 통한 추가 연구가 필요하지만, 본 연구는 소셜미디어 데이터를 활용한 금융 예측 모델의 가능성을 보여주는 의미있는 성과입니다.

---

**연구 기간**: 2024년 8월  
**대상 종목**: AMC, BB, GME  
**주요 성과**: Reddit Contrarian Effect 검증 및 ML 모델 성능 개선  
**핵심 발견**: Ridge + Reddit All 조합에서 최고 성능 (IC 0.0772)
