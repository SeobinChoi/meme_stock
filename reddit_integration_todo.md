# Reddit 데이터 통합 프로젝트 TODO

## 📋 전체 프로젝트 개요
- **목표**: Reddit 데이터를 딥러닝 모델에 효과적으로 통합하여 예측 성능 향상
- **대상**: AMC, BB, GME 3개 종목
- **평가 지표**: IC, Hit Rate, ICIR, QSR

## 🎯 Phase 1: Ridge 모델 포함 + 기본 Reddit 통합

### 1.1 Ridge 모델 추가
- [ ] 기존 고급 ML 모델 비교에 Ridge 모델 포함
- [ ] Ridge 모델 훈련 및 평가 (Price Only)
- [ ] Ridge 모델 성능 확인 (IC, Hit Rate, ICIR, QSR)

### 1.2 기본 Reddit 통합
- [ ] Price + Reddit 특성 단순 결합 (37개 특성)
- [ ] 모든 모델에 Reddit 특성 추가
- [ ] 성능 비교: Price Only vs Price + Reddit

### 1.3 1단계 결과 분석
- [ ] Ridge 모델 성능 확인
- [ ] Reddit 통합 효과 분석
- [ ] 모델별 성능 순위 재정렬

## 🧠 Phase 2: 딥러닝 Reddit 통합

### 2.1 시퀀스 기반 통합
- [ ] Price 시퀀스 (20일 × 17특성) 준비
- [ ] Reddit 시퀀스 (20일 × 20특성) 준비
- [ ] 시간적 정렬 및 결측값 처리

### 2.2 LSTM/GRU Reddit 통합
- [ ] LSTM 모델에 Reddit 시퀀스 추가
- [ ] GRU 모델에 Reddit 시퀀스 추가
- [ ] Cross-Attention 메커니즘 구현

### 2.3 CNN-LSTM Reddit 통합
- [ ] CNN-LSTM 모델에 Reddit 특성 통합
- [ ] 멀티모달 아키텍처 구현
- [ ] 특성 상호작용 레이어 추가

### 2.4 Transformer Reddit 통합
- [ ] Transformer 모델에 Reddit 어텐션 추가
- [ ] Price-Reddit Cross-Attention 구현
- [ ] 시간적 정렬 최적화

## 🚀 Phase 3: 고급 통합 모델

### 3.1 멀티모달 아키텍처
- [ ] Price Branch (CNN-LSTM) 구현
- [ ] Reddit Branch (Transformer) 구현
- [ ] Fusion Layer 구현

### 3.2 하이브리드 시퀀스 모델
- [ ] Price LSTM/GRU (시계열 패턴)
- [ ] Reddit LSTM/GRU (감정/관심도 패턴)
- [ ] Cross-Attention (상호작용)

### 3.3 계층적 통합 모델
- [ ] Level 1: Price Features (17개)
- [ ] Level 2: Reddit Features (20개)
- [ ] Level 3: Fusion Layer

## 📊 Phase 4: 종합 실험 및 평가

### 4.1 실험 1: 기본 통합
- [ ] 모든 모델에 Price + Reddit 특성 적용
- [ ] 성능 비교: IC, Hit Rate, ICIR, QSR
- [ ] 모델별 성능 순위 분석

### 4.2 실험 2: 고급 통합
- [ ] 멀티모달 아키텍처 성능 평가
- [ ] 기존 모델 대비 성능 향상 분석
- [ ] 특성 중요도 분석

### 4.3 실험 3: 시간적 정렬
- [ ] 시퀀스 기반 통합 모델 성능 평가
- [ ] 시간적 일관성과 예측력 분석
- [ ] 최적 시퀀스 길이 탐색

## 📈 Phase 5: 결과 분석 및 시각화

### 5.1 성능 비교 분석
- [ ] 모델별 성능 순위 재정렬
- [ ] Reddit 통합 효과 정량화
- [ ] 최적 모델 조합 식별

### 5.2 시각화 생성
- [ ] 성능 비교 차트 생성
- [ ] 특성 중요도 시각화
- [ ] 시간적 패턴 분석

### 5.3 최종 리포트 생성
- [ ] 종합 분석 리포트 작성
- [ ] 실전 적용 가이드 작성
- [ ] GitHub 푸시

## 🔧 기술적 세부사항

### 데이터 전처리
- [ ] 시간적 정렬 (일별 정렬, 지연 시간 고려)
- [ ] 특성 스케일링 (Price: StandardScaler, Reddit: MinMaxScaler)
- [ ] 결측값 처리 (Price: Forward Fill, Reddit: Zero Fill)
- [ ] 특성 엔지니어링 (Reddit 특성 조합, Price-Reddit 상호작용)

### 모델 아키텍처
- [ ] 시퀀스 기반 통합 모델
- [ ] 특성 기반 통합 모델
- [ ] 시간적 정렬 통합 모델
- [ ] 멀티모달 아키텍처

### 평가 메트릭
- [ ] IC (Information Coefficient)
- [ ] Hit Rate (방향성 정확도)
- [ ] ICIR (안정성)
- [ ] QSR (Quintile Spread Return)

## 📅 예상 일정
- **Phase 1**: 1-2일 (Ridge 포함 + 기본 통합)
- **Phase 2**: 2-3일 (딥러닝 통합)
- **Phase 3**: 2-3일 (고급 통합)
- **Phase 4**: 1-2일 (종합 실험)
- **Phase 5**: 1일 (결과 분석)

## 🎯 성공 기준
- **IC**: 0.05 → 0.08 (60% 향상)
- **Hit Rate**: 0.50 → 0.55 (10% 향상)
- **ICIR**: 0.15 → 0.25 (67% 향상)
- **QSR**: 0.005 → 0.010 (100% 향상)
