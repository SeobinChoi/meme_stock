# 🚀 A100 GPU Colab 실행 가이드

## 🎯 **주요 수정 사항 요약**

### **1. LSTM 데이터 로딩 문제 완전 해결**
- **문자열 데이터 타입 오류**: `sequences.dtype == object` 검사 및 숫자 컬럼만 추출
- **차원 검증**: 3D 배열 구조 강제 확인
- **NaN/Inf 처리**: `np.nan_to_num()` 으로 안전한 데이터 변환

### **2. A100 GPU 최적화**
- **혼합 정밀도 훈련**: `torch.cuda.amp` 사용으로 FP16 훈련
- **메모리 활용**: 40GB+ 메모리 활용한 배치 사이즈 256
- **GPU 설정**: `cudnn.benchmark = True`, 메모리 분할 95%

### **3. 오류 처리 강화**
- **Fallback 메커니즘**: 시퀀스 모델 실패 시 테이블 모델만 실행
- **단계별 검증**: 각 단계마다 데이터 품질 확인
- **상세한 로깅**: 문제 발생 지점 명확히 파악

## 🔧 **즉시 적용할 수정사항**

### **기존 노트북에서 수정할 부분**

#### **1. 데이터 로딩 셀 수정**
```python
# ❌ 기존 코드 (오류 발생)
def prepare_sequence_data():
    # ... 기존 코드 ...

# ✅ 수정된 코드
def prepare_sequence_data_fixed(metadata):
    """A100 GPU 최적화된 시퀀스 데이터 준비 (문자열 오류 해결)"""
    
    try:
        # Load sequence data
        timestamp = metadata['timestamp']
        sequences_data = np.load(f'sequences_{timestamp}.npz')
        
        print(f"🔍 Loading sequence data: {timestamp}")
        
        all_sequences = []
        all_targets = []
        all_dates = []
        
        # 데이터 타입 강제 변환 및 문자열 제거
        for ticker in metadata['tickers']:
            if f'{ticker}_sequences' in sequences_data:
                sequences = sequences_data[f'{ticker}_sequences']
                targets = sequences_data[f'{ticker}_targets_1d']
                dates = sequences_data[f'{ticker}_dates']
                
                print(f"   Processing {ticker}: {sequences.shape}, dtype: {sequences.dtype}")
                
                # 문자열이 포함된 경우 숫자 컬럼만 선택
                if sequences.dtype == object:
                    print(f"   ⚠️ {ticker} has object dtype, cleaning...")
                    
                    numeric_cols = []
                    for i in range(sequences.shape[2]):
                        try:
                            # 각 컬럼을 float로 변환 시도
                            test_col = sequences[:, :, i].astype(float)
                            numeric_cols.append(i)
                        except:
                            continue
                    
                    if len(numeric_cols) > 0:
                        sequences = sequences[:, :, numeric_cols].astype(np.float32)
                        print(f"   ✅ {ticker}: {len(numeric_cols)} numeric columns extracted")
                    else:
                        print(f"   ❌ {ticker}: No numeric columns found, skipping")
                        continue
                else:
                    sequences = sequences.astype(np.float32)
                
                # NaN 값 처리
                if np.any(np.isnan(sequences)) or np.any(np.isinf(sequences)):
                    print(f"   🧹 {ticker}: Cleaning NaN/Inf values...")
                    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
                
                all_sequences.append(sequences)
                all_targets.extend(targets)
                all_dates.extend(dates)
        
        if not all_sequences:
            raise ValueError("❌ No valid numeric sequences found!")
        
        # A100 최적화: float32 사용
        X_seq = np.vstack(all_sequences).astype(np.float32)
        y_seq = np.array(all_targets, dtype=np.float32)
        
        print(f"✅ Sequence data prepared: {X_seq.shape}, dtype: {X_seq.dtype}")
        return X_seq, y_seq, all_dates
        
    except Exception as e:
        print(f"❌ Error preparing sequence data: {e}")
        print("🔧 Fallback to tabular models only...")
        return None, None, None
```

#### **2. A100 GPU 최적화 설정 추가**
```python
# GPU 설정 셀에 추가
import torch

# A100 GPU 최적화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # A100 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 메모리 효율성 (A100 메모리 40GB+ 활용)
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
else:
    print("💻 Using CPU")

# 혼합 정밀도 훈련 (FP16)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

#### **3. LSTM 훈련 함수 수정**
```python
# LSTM 훈련 함수 수정
def train_lstm(X_train, y_train, X_val, y_val, epochs=100, batch_size=256, lr=0.001):
    """A100 GPU 최적화된 LSTM 훈련"""
    
    input_size = X_train.shape[2]
    model = LSTMModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # A100 최적화: 혼합 정밀도 훈련
    scaler = GradScaler()
    
    # Create data loaders with larger batch size
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ... 나머지 훈련 로직 ...
```

## 🚀 **실행 순서**

### **1단계: 데이터 업로드**
```python
from google.colab import files

print("📤 Upload the following files:")
print("   - tabular_train_YYYYMMDD_HHMMSS.csv")
print("   - tabular_val_YYYYMMDD_HHMMSS.csv") 
print("   - tabular_test_YYYYMMDD_HHMMSS.csv")
print("   - sequences_YYYYMMDD_HHMMSS.npz")
print("   - dataset_metadata_YYYYMMDD_HHMMSS.json")

uploaded = files.upload()
```

### **2단계: 데이터 로딩 및 검증**
```python
# 강화된 데이터 로딩
train_df, val_df, test_df, metadata = load_data_robust()

# 시퀀스 데이터 준비 (오류 처리 포함)
X_seq, y_seq, dates_seq = prepare_sequence_data_fixed(metadata)

if X_seq is not None:
    # 차원 검증
    if validate_sequence_dimensions(X_seq, y_seq):
        USE_SEQUENCE_MODELS = True
        print("✅ Sequence models enabled")
    else:
        USE_SEQUENCE_MODELS = False
        print("⚠️ Sequence models disabled")
else:
    USE_SEQUENCE_MODELS = False
    print("⚠️ Sequence models disabled")
```

### **3단계: 모델 훈련**
```python
# MLP 모델 (항상 실행)
mlp_model = train_mlp_a100(X_train, y_train, X_val, y_val, device)

# LSTM 모델 (시퀀스 데이터가 있는 경우만)
if USE_SEQUENCE_MODELS:
    lstm_model = train_lstm_a100(X_train_seq, y_train_seq, X_val_seq, y_val_seq, device)
```

## 📊 **예상 결과**

### **성공 시나리오**
- **LSTM 데이터 로딩**: 문자열 오류 없이 정상 로딩
- **모델 훈련**: A100 GPU 활용한 빠른 훈련
- **성능 향상**: IC ≥ 0.03 달성 가능성 높음

### **실패 시나리오 (Fallback)**
- **시퀀스 모델 비활성화**: 테이블 모델만 실행
- **안정적인 실행**: 오류 없이 완료
- **기본 성능**: 기존 Traditional ML 수준 유지

## 🔍 **문제 해결 체크리스트**

### **LSTM 데이터 로딩 문제**
- [ ] `sequences.dtype == object` 검사
- [ ] 숫자 컬럼만 추출하는 로직
- [ ] NaN/Inf 값 처리
- [ ] 차원 검증 (3D 배열 확인)

### **A100 GPU 최적화**
- [ ] 혼합 정밀도 훈련 (FP16)
- [ ] 배치 사이즈 256으로 증가
- [ ] GPU 메모리 설정 (95% 활용)
- [ ] `cudnn.benchmark = True`

### **오류 처리**
- [ ] Fallback 메커니즘 구현
- [ ] 단계별 검증 로직
- [ ] 상세한 에러 로깅
- [ ] 안전한 데이터 타입 변환

## 💡 **핵심 팁**

1. **문자열 데이터**: `sequences.dtype == object` 일 때만 특별 처리
2. **메모리 관리**: A100 메모리 40GB+ 활용하여 배치 사이즈 증가
3. **혼합 정밀도**: FP16 훈련으로 속도와 메모리 효율성 향상
4. **Fallback**: 시퀀스 모델 실패 시 테이블 모델로 계속 진행

이 가이드를 따라 수정하면 A100에서 안정적으로 실행되고 LSTM 데이터 로딩 문제도 완전히 해결될 것입니다!
