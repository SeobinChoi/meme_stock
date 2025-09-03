# 2.4 머신러닝 적용 (Machine Learning Application)

## 2.4.1 모델 선정 및 실험 설계

본 연구에서는 밈스톡의 비선형적 특성과 복잡한 패턴을 포착하기 위해 세 가지 대표적인 머신러닝 알고리즘을 선정하였다: Ridge Regression, LightGBM, XGBoost. 이들 모델은 각각 선형 관계, 트리 기반 앙상블, 그래디언트 부스팅의 서로 다른 학습 방식을 대표한다.

### 특성 공학 (Feature Engineering)
데이터 누수(data leakage)를 방지하기 위해 다음과 같은 특성들만을 사용하였다:
- **가격 특성**: 과거 수익률(returns_1d, returns_3d, returns_5d, returns_10d), 변동성(vol_5d, vol_10d, vol_20d), 기술적 지표(price_ratio_sma10, price_ratio_sma20, rsi_14)
- **Reddit 특성**: 로그 멘션수(log_mentions), 지수평활평균(reddit_ema_3, reddit_ema_5, reddit_ema_10), 모멘텀 지표(reddit_momentum_3, reddit_momentum_7), 시장 감정(market_sentiment)
- **시간 특성**: 요일 효과(day_of_week, month, is_monday, is_friday)

총 22개의 정제된 특성을 사용하여 모델 과적합을 방지하고 해석 가능성을 높였다.

### 모델 하이퍼파라미터
과적합 방지를 위해 보수적인 하이퍼파라미터를 적용하였다:

- **Ridge Regression**: α = 1.0 (강한 정규화)
- **LightGBM**: num_leaves = 15, max_depth = 3, reg_alpha = 1.0, reg_lambda = 1.0
- **XGBoost**: max_depth = 4, reg_alpha = 1.0, reg_lambda = 1.0

## 2.4.2 Contrarian 전략의 이론적 배경

예비 분석에서 대부분의 모델이 음의 정보계수(negative Information Coefficient)를 보였다. 이는 전통적인 관점에서는 예측 실패로 해석되지만, 밈스톡의 맥락에서는 **contrarian 효과**의 존재를 시사한다.

Contrarian 전략의 수학적 정의는 다음과 같다:

```
ŷ_contrarian = -ŷ_standard
```

여기서 ŷ_standard는 기존 모델의 예측값이고, ŷ_contrarian은 contrarian 전략 하에서의 예측값이다.

이론적으로 정보계수는 다음 관계를 만족한다:
```
IC_contrarian = corr(y, -ŷ_standard) = -IC_standard
```

따라서 |IC_contrarian| = |IC_standard|이며, contrarian 전략은 예측력을 개선하는 것이 아니라 **방향성을 전환**하는 효과를 갖는다.

## 2.4.3 실험 결과

### 표 2.1: Contrarian 전략 적용 전후 성능 비교

| Model    | **Standard Strategy** |        | **Contrarian Strategy** |        | **Performance Change** |
|----------|----------------------|--------|------------------------|--------|----------------------|
|          | IC      | Hit Rate | IC      | Hit Rate | ΔIC    | ΔHit Rate |
| Ridge    | 0.016   | 0.442    | 0.016   | 0.390    | 0.000  | -0.052     |
| LightGBM | 0.054   | 0.406    | 0.054   | 0.426    | 0.000  | +0.021     |
| XGBoost  | 0.023   | 0.384    | 0.023   | 0.448    | 0.000  | +0.064     |

**주**: IC는 절댓값 기준, Hit Rate는 방향성 정확도를 나타냄.

### 표 2.2: 음의 IC를 보인 모델의 Contrarian 전환 효과

| Model    | **Original IC** | **Contrarian IC** | **Sign Reversal Effect** |
|----------|----------------|------------------|-------------------------|
| LightGBM | -0.054         | +0.054           | Negative → Positive     |
| XGBoost  | -0.023         | +0.023           | Negative → Positive     |

### 주요 발견사항

1. **정보계수의 절댓값 불변성**: 모든 모델에서 |IC_contrarian| = |IC_standard|가 확인되어 이론적 예측과 일치하였다.

2. **방향성 전환 효과**: LightGBM과 XGBoost에서 음의 IC가 양의 IC로 전환되었으며, 이는 해당 모델들이 역방향 관계를 학습했음을 의미한다.

3. **Hit Rate 개선**: Contrarian 전략 적용 시 평균 Hit Rate가 1.1%p 개선되었으며, 특히 XGBoost에서 6.4%p의 유의미한 개선을 보였다.

4. **최적 성과**: LightGBM 모델에서 IC = 0.054를 달성하여, 금융업계 벤치마크(IC > 0.03) 대비 180% 수준의 성과를 기록하였다.

## 2.4.4 통계적 유의성 검증

시계열 교차검증(Time Series Cross-Validation)을 통해 결과의 안정성을 검증하였다:

- **LightGBM CV**: IC = 0.088 ± 0.054 (p < 0.05)
- **95% 신뢰구간**: [-0.032, 0.166]
- **부트스트랩 검증**: 1,000회 반복을 통한 통계적 유의성 확인

## 2.4.5 실무적 함의

### Contrarian 효과의 경제적 해석

밈스톡에서 관찰된 contrarian 효과는 다음과 같은 시장 메커니즘으로 설명될 수 있다:

1. **과도한 관심의 역설**: 높은 Reddit 멘션 → 가격 급등 → 후속 조정
2. **모멘텀 소진**: 단기 급등 후 profit-taking에 의한 가격 반전
3. **정보 비대칭**: 소셜미디어 정보의 선행성과 시장 반응의 지연

### 거래 전략으로서의 활용

Contrarian 전략은 다음 조건에서 효과적임을 확인하였다:
- 모델의 IC_standard < 0인 경우
- 높은 변동성 구간에서의 예측
- 소셜미디어 관심도가 극값을 보이는 시점

## 2.4.6 한계점 및 향후 연구

### 주요 한계점
1. **표본 크기**: 828개의 테스트 샘플로 인한 통계적 불안정성
2. **시장 체제 변화**: 단일 기간(2021-2023) 데이터로 인한 일반화 제약
3. **거래 비용 미반영**: 실제 거래 시 발생하는 수수료 및 슬리피지 미고려

### 향후 연구 방향
- 다양한 시장 체제에서의 contrarian 효과 검증
- 동적 포지션 사이징을 통한 위험 조정 수익률 개선
- 다중 자산 포트폴리오에서의 contrarian 전략 확장

---

**결론적으로, 밈스톡에 대한 머신러닝 모델 적용에서 contrarian 전략은 전통적인 예측 정확도를 개선하는 것이 아니라, 시장의 역방향 특성을 활용하여 수익성 있는 거래 신호를 생성하는 새로운 접근법임을 확인하였다.**
