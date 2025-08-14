# 🚀 Colab A100 GPU 디버깅 가이드

## 🎯 **LSTM 데이터 로딩 문제 해결**

### **문제 1: 문자열 데이터 타입 오류**
```python
# ❌ 기존 코드 (오류 발생)
sequences_data = np.load(f'sequences_{timestamp}.npz')
X_seq = np.vstack(all_sequences)  # 문자열 포함 시 오류

# ✅ 수정된 코드
def prepare_sequence_data_fixed():
    """A100 GPU 최적화된 시퀀스 데이터 준비"""
    
    all_sequences = []
    all_targets = []
    all_dates = []
    
    # 데이터 타입 강제 변환
    for ticker in metadata['tickers']:
        if f'{ticker}_sequences' in sequences_data:
            sequences = sequences_data[f'{ticker}_sequences']
            targets = sequences_data[f'{ticker}_targets_1d']
            dates = sequences_data[f'{ticker}_dates']
            
            # 문자열 제거 및 숫자만 추출
            if sequences.dtype == object:
                # 문자열이 포함된 경우 숫자 컬럼만 선택
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
                else:
                    print(f"⚠️ Warning: {ticker} has no numeric columns, skipping")
                    continue
            else:
                sequences = sequences.astype(np.float32)
            
            # NaN 값 처리
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
```

### **문제 2: 메모리 부족 해결**
```python
# A100 GPU 메모리 최적화
import torch

# GPU 메모리 정리
torch.cuda.empty_cache()

# 혼합 정밀도 훈련 (FP16)
from torch.cuda.amp import autocast, GradScaler

# 배치 사이즈 증가 (A100 메모리 활용)
BATCH_SIZE = 256  # 기존 64에서 증가
```

### **문제 3: 차원 불일치 해결**
```python
def validate_sequence_dimensions(X_seq, y_seq):
    """시퀀스 차원 검증"""
    
    print(f"🔍 Sequence validation:")
    print(f"   X shape: {X_seq.shape}")
    print(f"   y shape: {y_seq.shape}")
    print(f"   X dtype: {X_seq.dtype}")
    print(f"   y dtype: {y_seq.dtype}")
    
    # 차원 검증
    if len(X_seq.shape) != 3:
        raise ValueError(f"❌ Expected 3D array, got {len(X_seq.shape)}D")
    
    if X_seq.shape[0] != len(y_seq):
        raise ValueError(f"❌ Sample count mismatch: X={X_seq.shape[0]}, y={len(y_seq)}")
    
    # NaN/Inf 검사
    if np.any(np.isnan(X_seq)) or np.any(np.isinf(X_seq)):
        print("⚠️ Warning: NaN/Inf detected in sequences, cleaning...")
        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("✅ Sequence validation passed!")
    return X_seq, y_seq
```

## 🚀 **A100 GPU 최적화 설정**

### **GPU 설정**
```python
# A100 GPU 최적화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # A100 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 메모리 효율성
    torch.cuda.set_per_process_memory_fraction(0.95)  # 95% 메모리 사용
else:
    print("💻 Using CPU")
```

### **혼합 정밀도 훈련**
```python
# FP16 훈련으로 A100 성능 극대화
scaler = GradScaler()

def train_with_amp(model, train_loader, optimizer, criterion):
    """혼합 정밀도 훈련"""
    
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # FP16 forward pass
        with autocast():
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
        
        # FP16 backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

## 🔧 **즉시 적용할 수정사항**

### **1. 데이터 로딩 부분 수정**
```python
# 기존 prepare_sequence_data() 함수를 위의 prepare_sequence_data_fixed()로 교체
```

### **2. LSTM 모델 훈련 부분 수정**
```python
# 배치 사이즈 증가
BATCH_SIZE = 256

# 메모리 정리 추가
torch.cuda.empty_cache()

# 차원 검증 추가
X_seq, y_seq = validate_sequence_dimensions(X_seq, y_seq)
```

### **3. 오류 처리 강화**
```python
try:
    # 시퀀스 데이터 준비
    X_seq, y_seq, dates_seq = prepare_sequence_data_fixed()
    
    # 차원 검증
    X_seq, y_seq = validate_sequence_dimensions(X_seq, y_seq)
    
    print("✅ Sequence data loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading sequence data: {e}")
    print("🔧 Fallback to tabular models only...")
    
    # 시퀀스 모델 비활성화
    USE_SEQUENCE_MODELS = False
```

## 📊 **예상 성능 향상**

### **A100 vs T4 비교**
- **메모리**: 40GB vs 16GB (2.5배 증가)
- **배치 사이즈**: 256 vs 64 (4배 증가)
- **훈련 속도**: 3-5배 향상
- **모델 크기**: 더 큰 hidden layers 가능

### **권장 모델 설정**
```python
# A100 최적화된 모델 크기
LSTM_HIDDEN_SIZE = 256  # 기존 128에서 증가
TRANSFORMER_D_MODEL = 256  # 기존 128에서 증가
MLP_HIDDEN_DIMS = [1024, 512, 256, 128]  # 기존보다 큰 모델
```

## 🎯 **성공 기준 달성 전략**

### **IC ≥ 0.03 달성 방안**
1. **더 큰 모델**: A100 메모리 활용한 대형 모델
2. **혼합 정밀도**: FP16으로 안정적인 훈련
3. **데이터 품질**: 문자열 오류 완전 제거
4. **앙상블**: 여러 모델 조합으로 성능 향상

이 가이드를 따라 수정하면 A100에서 안정적으로 실행되고 성능도 크게 향상될 것입니다!
