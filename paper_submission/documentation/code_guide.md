# 분석 코드 실행 가이드

## 코드 파일 개요

### 1. fast_correlation_analysis.py ⭐
**목적**: 핵심 Contrarian Effect 발견  
**실행시간**: ~10초  
**메모리**: 0.2MB  
**출력**: `fast_correlation_analysis.png`

**주요 기능**:
- GME, AMC, BB의 reddit_surprise vs returns 상관관계 계산
- 메모리 최적화된 데이터 로딩 (float32 사용)
- 명확한 시각화

**실행 방법**:
```bash
cd /Users/xavi/Desktop/temp_code/meme_stock
python analysis/price_correlation/fast_correlation_analysis.py
```

**핵심 결과**:
```
GME: surprise=-0.198, momentum3=-0.164, momentum7=0.055
AMC: surprise=-0.178, momentum3=0.019, momentum7=-0.020  
BB: surprise=-0.165, momentum3=0.101, momentum7=0.145
```

---

### 2. advanced_analysis_suite.py ⭐⭐
**목적**: 종합 고급 분석 (Event Study, Regime, Spillover 등)  
**실행시간**: ~30초  
**메모리**: ~50MB  
**출력**: `advanced_analysis_comprehensive.png`

**포함 분석**:
1. **Event Study**: 극한 Reddit 스파이크 전후 분석
2. **Market Regime**: Bull/Bear 시장별 contrarian effect  
3. **Cross-Stock Spillover**: 주식 간 연쇄 효과
4. **Volatility Regime**: 변동성별 패턴 차이
5. **Temporal Pattern**: 연도별, 분기별 변화

**실행 방법**:
```bash
python scripts/advanced_analysis_suite.py
```

**핵심 발견**:
- Bull 시장: contrarian effect 강화 (-0.27~-0.29)
- 113개 극한 이벤트 분석 완료
- Cross-stock spillover 모든 조합에서 확인

---

### 3. intraday_temporal_analysis.py ⭐
**목적**: 시간적 패턴 분석 (요일, 월, 계절별)  
**실행시간**: ~20초  
**출력**: `temporal_analysis_comprehensive.png`

**분석 내용**:
- 요일별 contrarian effect (화요일 최악)
- 월별/계절별 패턴 (겨울철 강화)
- 시장 타이밍 효과 (월말, 실적발표철)
- 30일 롤링 상관관계

**실행 방법**:
```bash
python scripts/intraday_temporal_analysis.py
```

**핵심 발견**:
- Tuesday 최악: GME -0.374, AMC -0.434
- Friday 역전: GME +0.307
- 겨울철 contrarian 최강: -0.38~-0.40

---

### 4. network_influence_analysis.py
**목적**: 네트워크 효과 및 영향력 분석  
**실행시간**: ~15초  
**출력**: `network_analysis_comprehensive.png`

**분석 내용**:
- 영향력 유형 시뮬레이션 (Hype vs Momentum)
- 바이럴 캐스케이드 패턴
- 주의집중 효과

**실행 방법**:
```bash
python scripts/network_influence_analysis.py
```

---

### 5. enhanced_contrarian_model.py
**목적**: 예측 모델 및 트레이딩 전략  
**실행시간**: ~45초  
**출력**: `enhanced_contrarian_analysis.png`

**기능**:
- 리스크 조정 포지션 사이징
- 트랜잭션 비용 고려
- 기간별 robustness 테스트
- Sharpe ratio 최적화

**주의**: 현재 성능이 제한적 (학술 연구용)

---

## 실행 환경 요구사항

### Python 버전:
- Python 3.8+ (권장: 3.10+)

### 필수 라이브러리:
```bash
pip install pandas numpy matplotlib scikit-learn scipy seaborn lightgbm
```

### 메모리 요구사항:
- **최소**: 4GB RAM
- **권장**: 8GB RAM (M1 Mac 기준)
- **최적**: 16GB RAM

### 실행 순서 (권장):
1. `fast_correlation_analysis.py` (핵심 결과 확인)
2. `advanced_analysis_suite.py` (종합 분석)
3. `intraday_temporal_analysis.py` (시간 패턴)
4. `network_influence_analysis.py` (네트워크 효과)

## 데이터 의존성

### 입력 데이터:
```
data/colab_datasets/
├── tabular_train_20250814_031335.csv
├── tabular_val_20250814_031335.csv  
└── tabular_test_20250814_031335.csv
```

### 데이터 규모:
- 총 5,409 샘플
- 6개 티커 (GME, AMC, BB, DOGE, ETH, SHIB)
- 2021-2023 기간

### 주요 컬럼:
- `reddit_surprise`: 핵심 predictor
- `returns_1d/5d`: target variables  
- `vol_5d`, `rsi_14`: control variables

## 성능 최적화 팁

### 메모리 절약:
1. `dtype=float32` 사용 (50% 메모리 절약)
2. 필요한 컬럼만 로딩
3. 불필요한 DataFrame 즉시 삭제

### 속도 향상:
1. 병렬 처리 (pandas groupby 최적화)
2. 벡터화 연산 활용
3. 중간 결과 캐싱

### M1 Mac 특화:
- 네이티브 NumPy/SciPy 사용
- 메모리 압박 시 swap 활용
- 백그라운드 앱 종료 권장

## 트러블슈팅

### 자주 발생하는 오류:

#### 1. Memory Error (Bus Error)
**원인**: 메모리 부족  
**해결**: 다른 앱 종료, chunk 단위 처리

#### 2. File Not Found
**원인**: 데이터 파일 경로 문제  
**해결**: 작업 디렉토리 확인
```bash
pwd  # 현재 위치 확인
ls data/colab_datasets/  # 파일 존재 확인
```

#### 3. Import Error
**원인**: 라이브러리 누락  
**해결**: 
```bash
pip install -r requirements.txt
```

#### 4. Correlation NaN
**원인**: 데이터 부족 또는 모든 값이 동일  
**해결**: 샘플 수 확인, 데이터 품질 검증

### 성능 모니터링:
```bash
# 메모리 사용량 확인
htop

# Python 메모리 프로파일링
pip install memory_profiler
python -m memory_profiler script_name.py
```

## 확장 가능성

### 새로운 분석 추가:
1. `scripts/` 폴더에 새 파일 생성
2. 기존 `load_data()` 함수 재사용
3. 표준화된 시각화 함수 활용

### 다른 데이터셋 적용:
1. 컬럼명 매핑 조정
2. 데이터 타입 변환
3. 결측치 처리 로직 수정

---

**최종 권장**: 논문 작성 시 1-3번 코드 결과 활용  
**백업 계획**: 코드 실행 불가 시 기존 이미지 파일 사용  
**발표 준비**: 실시간 코드 실행 데모 준비 (1번 코드로)
